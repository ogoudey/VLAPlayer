"""Independent target and inference loops for live Tidybot execution."""

from collections import deque
from dataclasses import dataclass
import math
import queue
import threading
import time

from schemas import PoseTarget


@dataclass
class PredictionRequest:
    observation: object
    observed_at: float
    period: float
    horizon: int
    generation: int


@dataclass
class PredictionResult:
    request: PredictionRequest
    chunk: object = None
    error: Exception | None = None
    latency_s: float = 0.0


class TidybotTargetStream:
    """Sample state and send the latest absolute target on every control tick.

    A single worker owns inference requests. The control loop owns arm RPCs;
    neither a slow prediction nor a queue underrun suspends target submission.
    Pause, rate changes, and command errors invalidate in-flight predictions.
    """

    def __init__(self, client, stop_event):
        self.client = client
        self.stop_event = stop_event
        self._lock = threading.RLock()
        self._enabled = client.predicting
        self._generation = 0
        self._period = client.connection.control_period_s
        self._targets = deque()
        self._last_target = None
        self._pending = None
        self._requests = queue.Queue(maxsize=1)
        self._results = queue.Queue()
        self._latency_s = 0.0
        self._retry_at = 0.0
        self._worker = None

    @property
    def worker_alive(self):
        return self._worker is not None and self._worker.is_alive()

    def set_enabled(self, enabled):
        # Synchronize with submission so Pause cannot return while a command
        # from the previous generation is still being submitted.
        with self._lock:
            if enabled != self._enabled:
                self._enabled = enabled
                self._invalidate()
                self._last_target = None

    def _invalidate(self):
        self._generation += 1
        self._targets.clear()
        self._retry_at = 0.0

    def stop(self):
        self.stop_event.set()
        self.set_enabled(False)

    def _predict(self):
        while not self.stop_event.is_set():
            try:
                request = self._requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.stop_event.is_set():
                return
            started = time.monotonic()
            try:
                chunk = self.client.server.make_prediction(observation=request.observation)
                result = PredictionResult(request, chunk=chunk)
            except Exception as exc:
                result = PredictionResult(request, error=exc)
            result.latency_s = time.monotonic() - started
            self._results.put(result)

    def _receive(self, now):
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                return
            if result.request is not self._pending:
                continue
            self._pending = None
            request = result.request
            if request.generation != self._generation or not self._enabled:
                continue
            self._latency_s = result.latency_s
            self.client.ui.report_client_latency(result.latency_s * 1000)
            if result.error is not None:
                self.client.ui.report_loop_event(
                    f"Prediction request failed; retaining last target: {result.error}", level="warn",
                )
                self._retry_at = now + 0.5
                continue
            if result.chunk is None or not result.chunk.actions:
                self.client.ui.report_loop_event(
                    "Prediction returned empty chunk; retaining last target", level="warn",
                )
                self._retry_at = now + 0.5
                continue
            actions = result.chunk.actions
            if any(not isinstance(action, PoseTarget) for action in actions):
                raise TypeError("Live Tidybot streaming requires absolute PoseTarget actions")

            # Outputs describe future steps relative to the request observation.
            # Replace the old plan with the part whose time has not passed.
            elapsed_steps = max(0, math.floor(
                (now - request.observed_at) / request.period + 1e-6,
            ))
            fresh = actions[elapsed_steps:elapsed_steps + request.horizon]
            if fresh:
                self._targets.clear()
                self._targets.extend(fresh)
            self.client.ui.report_loop_event(
                f"Received chunk of {len(actions)} actions; skipped {min(elapsed_steps, len(actions))} "
                f"elapsed steps, queued {len(fresh)} fresh targets"
            )

    def tick(self, now):
        """One state sample and target submission; inference never runs here."""
        with self._lock:
            if self.stop_event.is_set():
                return
            self.set_enabled(self.client.predicting)
            try:
                period = self.client.connection.control_period_s
                if not math.isfinite(period) or period <= 0:
                    raise ValueError("Control period must be finite and positive")
                if period != self._period:
                    self._period = period
                    self._invalidate()
                self._receive(now)
                if not self._enabled:
                    return

                # Like teleop recording, read measured state before issuing the
                # action. Only this thread accesses the connection for inference.
                observation = self.client.get_observation()
                if self._last_target is None:
                    pose = observation.state.target_pose
                    self._last_target = PoseTarget(
                        x=pose.x, y=pose.y, z=pose.z,
                        theta_x=pose.theta_x, theta_y=pose.theta_y,
                        theta_z=pose.theta_z, theta_w=pose.theta_w,
                        gripper_command=observation.state.gripper,
                    )

                horizon = int(self.client.server.execution_horizon)
                if horizon <= 0:
                    raise ValueError("Execution horizon must be positive")
                # Start the next request with enough targets left to cover the
                # observed latency, plus two ticks. Keep at most one in flight.
                prefetch_steps = min(horizon, max(
                    (horizon + 1) // 2, math.ceil(self._latency_s / period) + 2,
                ))
                if (self._pending is None and len(self._targets) <= prefetch_steps
                        and now >= self._retry_at):
                    request = PredictionRequest(
                        observation, now, period, horizon, self._generation,
                    )
                    self._requests.put_nowait(request)
                    self._pending = request

                action = self._targets.popleft() if self._targets else self._last_target
                self.client.connection.apply_abs_action_tidybot(action)
                self._last_target = action
                self.client.ui.report_applied_action(action)
                self.client.ui.report_queue_depth(len(self._targets))
            except Exception as exc:
                self.client.ui.report_loop_event(
                    f"Tidybot streaming failed, pausing: {exc}", level="warn",
                )
                self.client.stop_predicting()
                self.set_enabled(False)
                self.client.connection.pause()

    def run(self):
        self._worker = threading.Thread(
            target=self._predict, daemon=True, name="groot-prediction-worker",
        )
        self._worker.start()
        deadline = time.monotonic()
        period = self._period
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                self.tick(started)
                if self._period != period:
                    period = self._period
                    deadline = started
                deadline += period
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Don't send a burst of targets to catch up after an RPC
                    # overrun; resume one target per period from this point.
                    remaining = period
                    deadline = time.monotonic() + period
                self.stop_event.wait(remaining)
                self.client.ui.report_loop_timing(period, time.monotonic() - started)
        finally:
            self.stop()
            self.client.connection.pause()
            # A running model request has its own timeout. Never wait for that
            # request before stopping arm submissions.
            self._worker.join(timeout=0.2)
