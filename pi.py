from client import Client, ServerConfiguration
from typing import Optional
from schemas import Observation, Vision, State, Action, CartesianDelta
from queue import queue
import time
import threading

class Pi05Client(Client):
    def __init__(self, remote: Optional[ServerConfiguration]=None):
        if remote is None:
            # ignore for now
            pass
        else:
            self.server = remote


    def awake(self):
        # load the model, if it were local
        try:
            self.server.test_health()
        except Exception as e:
            print(f"Could not reach server: {e}")

    def start_inference_loop(self):
        self._stop_inference = threading.Event()
        action_queue: "queue.Queue[Action]" = queue.Queue()
        period = self.connection.control_period_s

        next_tick = time.monotonic()

        while not self._stop_inference.is_set():
            if not self.predicting:
                # Paused: don't busy-spin, but keep checking so we notice
                # being resumed or stopped.
                time.sleep(0.05)
                next_tick = time.monotonic()  # don't try to "catch up" on resume
                continue

            if action_queue.empty():
                try:
                    chunk = self.server.make_prediction(observation=self.get_observation())
                except Exception as e:
                    print(f"Prediction request failed, retrying: {e}")
                    time.sleep(0.5)  # back off before hammering a possibly-down server
                    continue
                for action in chunk:
                    action_queue.put(action)

            action = action_queue.get()
            try:
                self.connection.apply_action(action)
            except Exception as e:
                # Don't let one bad action kill the loop/thread silently. The
                # Twist command's built-in duration timeout means a skipped
                # action just lets the arm coast to a stop, not run away.
                print(f"apply_action failed, skipping this action: {e}")

            # Pace to the control period on a fixed schedule (not "sleep after
            # each step") so per-step overhead — the server call, apply_action
            # itself — doesn't accumulate drift over a long run.
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()  # fell behind; reset rather than spiral

        # Loop is exiting — make sure the arm isn't left coasting on a stale
        # velocity command from the last apply_action().
        self.connection.handle_cartesian_delta(CartesianDelta())

    def stop_inference_loop(self):
        self._stop_inference.set()

    def get_observation(self):
        return Observation(
            vision = self.camera_set.get_vision(),
            state = self.connection.state(),
            langauge = self.language
        )
        
