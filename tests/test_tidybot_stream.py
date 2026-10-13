from types import SimpleNamespace
import threading
import unittest
from unittest.mock import MagicMock, patch

from clients.groot import GrootN17Client
from clients.tidybot_stream import PredictionResult, TidybotTargetStream
from schemas import ActionChunk, Pose, PoseTarget, State


def target(x):
    return PoseTarget(x=x, y=0, z=.3, theta_x=0, theta_y=0, theta_z=0,
                      theta_w=1, gripper_command=.4)


class TidybotStreamTests(unittest.TestCase):
    def setUp(self):
        self.server = SimpleNamespace(execution_horizon=4, make_prediction=MagicMock())
        self.client = GrootN17Client(remote=self.server, tidybot=True)
        self.client.connection = MagicMock(control_period_s=.02)
        self.client.ui = MagicMock()
        self.measured = State(
            joint_angles=[0] * 7,
            target_pose=Pose(x=.1, y=0, z=.3, theta_x=0, theta_y=0,
                             theta_z=0, theta_w=1), gripper=.4,
        )
        self.client.get_observation = MagicMock(side_effect=lambda: SimpleNamespace(
            state=self.measured.model_copy(deep=True),
        ))
        self.stream = TidybotTargetStream(self.client, threading.Event())
        self.client._target_stream = self.stream

    def complete(self, xs, *, latency=0.0, error=None):
        request = self.stream._requests.get_nowait()
        self.stream._results.put(PredictionResult(
            request, chunk=ActionChunk(actions=[target(x) for x in xs]),
            error=error, latency_s=latency,
        ))
        return request

    def sent_x(self):
        return [call.args[0].x for call in
                self.client.connection.apply_abs_action_tidybot.call_args_list]

    def test_continues_20ms_submissions_and_state_samples_during_slow_prediction(self):
        for step in range(6):
            self.stream.tick(step * .02)
        self.assertEqual(self.sent_x(), [.1] * 6)
        self.assertEqual(self.client.get_observation.call_count, 6)
        self.assertEqual(self.stream._requests.qsize(), 1)
        self.client.connection.pause.assert_not_called()
        self.complete(range(10, 26), latency=.12)
        self.stream.tick(.12)
        self.assertEqual(self.sent_x()[-1], 16)  # six elapsed output steps skipped
        self.assertEqual(len(self.stream._targets), 3)  # horizon limits usable steps
        self.assertEqual(self.client.get_observation.call_count, 7)

    def test_prefetch_replaces_old_plan_without_pause_or_chunk_boundary_delay(self):
        self.stream.tick(0.0)
        self.complete(range(1, 9))
        self.stream.tick(0.0)
        self.stream.tick(.02)
        self.assertIsNone(self.stream._pending)
        self.stream.tick(.04)
        request = self.complete(range(101, 109), latency=.02)
        self.assertEqual(request.observed_at, .04)
        self.stream.tick(.06)
        self.assertEqual(self.sent_x(), [.1, 1, 2, 3, 102])
        self.client.connection.pause.assert_not_called()

    def test_underrun_resends_last_commanded_target_not_measured_pose(self):
        self.stream.tick(0.0)
        self.complete([.2, .25])
        self.stream.tick(0.0)
        self.stream.tick(.02)
        self.measured.target_pose.x = .11  # measured arm still trails the target
        self.stream.tick(.04)
        self.stream.tick(.06)
        self.assertEqual(self.sent_x(), [.1, .2, .25, .25, .25])
        self.client.connection.pause.assert_not_called()

    def test_prediction_older_than_entire_chunk_retains_latest_target(self):
        self.stream.tick(0.0)
        self.complete([.2, .3], latency=.1)
        self.stream.tick(.1)
        self.assertEqual(self.sent_x(), [.1, .1])
        self.assertEqual(len(self.stream._targets), 0)
        self.assertIsNotNone(self.stream._pending)  # asks for a fresh prediction

    def test_pause_stops_submissions_and_resume_discards_inflight_result(self):
        self.stream.tick(0.0)
        self.client.stop_predicting()
        self.stream.tick(.02)
        self.assertEqual(self.sent_x(), [.1])
        self.complete([.9, .95], latency=.02)
        self.measured.target_pose.x = .15
        self.client.start_predicting()
        self.stream.tick(.04)
        self.assertEqual(self.sent_x(), [.1, .15])
        request = self.stream._requests.get_nowait()
        self.assertEqual(request.observation.state.target_pose.x, .15)
        self.assertEqual(request.observed_at, .04)

    def test_rate_change_discards_predictions_from_old_dt(self):
        self.stream.tick(0.0)
        self.complete([.9] * 16)
        self.client.connection.control_period_s = .04
        self.stream.tick(.04)
        self.assertEqual(self.sent_x(), [.1, .1])
        self.assertEqual(self.stream._requests.get_nowait().period, .04)

    def test_empty_prediction_and_network_errors_keep_target_stream_running(self):
        for error in (None, RuntimeError("model timeout")):
            with self.subTest(error=error):
                self.setUp()
                self.stream.tick(0.0)
                self.complete([], error=error)
                self.stream.tick(.02)
                self.stream.tick(.04)
                self.assertEqual(self.sent_x(), [.1, .1, .1])
                self.assertTrue(self.client.predicting)
                self.assertIsNone(self.stream._pending)  # retry is delayed, control isn't
                self.client.connection.pause.assert_not_called()
                self.stream.tick(.52)
                self.assertIsNotNone(self.stream._pending)

    def test_rpc_error_pauses_discards_chunk_and_resume_requests_fresh_plan(self):
        self.stream.tick(0.0)
        self.complete([.2, .3, .4, .5])
        self.client.connection.apply_abs_action_tidybot.side_effect = RuntimeError("RPC failed")
        self.stream.tick(0.0)
        self.assertFalse(self.client.predicting)
        self.assertEqual(len(self.stream._targets), 0)
        self.client.connection.pause.assert_called_once()
        self.stream.tick(.02)
        self.assertEqual(self.sent_x(), [.1, .2])
        self.client.connection.apply_abs_action_tidybot.side_effect = None
        self.client.start_predicting()
        self.measured.target_pose.x = .12
        self.stream.tick(.04)
        self.assertEqual(self.sent_x(), [.1, .2, .12])
        self.assertIsNotNone(self.stream._pending)

    def test_horizon_changes_apply_to_next_request(self):
        self.stream.tick(0.0)
        first = self.complete(range(10), latency=0)
        self.server.execution_horizon = 2
        self.stream.tick(0.0)
        self.assertEqual(first.horizon, 4)
        self.assertEqual(len(self.stream._targets), 3)
        self.stream.tick(.02)
        self.stream.tick(.04)
        self.assertEqual(self.stream._requests.get_nowait().horizon, 2)

    def test_real_inference_worker_does_not_block_control_or_survive_stop_sending(self):
        prediction_started = threading.Event()
        release_prediction = threading.Event()
        four_targets = threading.Event()
        timestamps = []
        inference_threads = []
        control_threads = []

        def predict(**kwargs):
            inference_threads.append(threading.get_ident())
            prediction_started.set()
            release_prediction.wait(2)
            return ActionChunk(actions=[target(.2)] * 32)

        def submit(action):
            control_threads.append(threading.get_ident())
            timestamps.append(action.x)
            if len(timestamps) >= 4:
                four_targets.set()

        self.server.make_prediction.side_effect = predict
        self.client.connection.apply_abs_action_tidybot.side_effect = submit
        # Exercise the public client startup/stop path, not just the scheduler.
        runner = threading.Thread(target=self.client.start_inference_loop, daemon=True)
        runner.start()
        try:
            self.assertTrue(prediction_started.wait(1))
            self.stream = self.client._target_stream
            self.assertTrue(four_targets.wait(1))
            self.assertEqual(timestamps, [.1] * len(timestamps))
            self.assertEqual(self.server.make_prediction.call_count, 1)
            self.client.connection.pause.assert_not_called()
            self.assertNotEqual(inference_threads[0], control_threads[0])
            self.assertGreaterEqual(self.client.get_observation.call_count, 4)
            self.client.stop_inference_loop()
            calls_at_stop = len(timestamps)
        finally:
            self.stream.stop()
            release_prediction.set()
            runner.join(2)
        self.assertFalse(runner.is_alive())
        self.assertEqual(len(timestamps), calls_at_stop)
        self.assertFalse(self.stream.worker_alive)

    def test_control_overrun_does_not_burst_targets_to_catch_up(self):
        now = [0.0]
        sent_at = []

        class StopEvent:
            stopped = False

            def is_set(self):
                return self.stopped

            def set(self):
                self.stopped = True

            def wait(self, seconds):
                if not self.stopped:
                    now[0] += seconds
                return self.stopped

        stop = StopEvent()
        self.stream.stop_event = stop

        def submit(action):
            sent_at.append(now[0])
            if len(sent_at) == 1:
                now[0] += .03  # first arm RPC exceeds the 20 ms period
            if len(sent_at) == 3:
                stop.set()

        self.client.connection.apply_abs_action_tidybot.side_effect = submit
        clock = SimpleNamespace(monotonic=lambda: now[0])
        with patch("clients.tidybot_stream.time", clock), \
                patch("clients.tidybot_stream.threading.Thread"):
            self.stream.run()
        self.assertEqual(len(sent_at), 3)
        for actual, expected in zip(sent_at, (0.0, .05, .07)):
            self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
