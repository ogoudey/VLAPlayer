import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch

from clients.groot import GrootN17Client, RecordedGrootConfiguration
from replay_episode import DEFAULT_EPISODE, action_kwargs
from schemas import Pose, State


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class GrootReplayTests(unittest.TestCase):
    def make_client(self, **kwargs):
        source = RecordedGrootConfiguration(DEFAULT_EPISODE, min_z=.1, **kwargs)
        client = GrootN17Client(remote=source, tidybot=False)
        ref = source.episode["frames"][0]["observation.state"]
        import math
        state = State(joint_angles=[math.degrees(q) for q in ref[:7]],
                      target_pose=Pose(x=ref[7], y=ref[8], z=ref[9],
                                       theta_x=ref[10], theta_y=ref[11], theta_z=ref[12], theta_w=ref[13]),
                      gripper=ref[14])
        client.connection = MagicMock(control_period_s=.02, max_linear_velocity=.15)
        client.connection.ui = None
        client.connection.state.return_value = state
        client.ui = MagicMock()
        client.get_observation = MagicMock(return_value=MagicMock(state=state))
        return client, source

    def run_loop(self, client):
        clock = FakeClock()
        with patch("clients.groot.time_package", clock), contextlib.redirect_stdout(io.StringIO()):
            client.start_inference_loop()
        return clock

    def test_full_episode_uses_real_decoder_and_loop_without_server(self):
        client, source = self.make_client(response_delay_s=.025)
        with patch("clients.groot.PolicyClient") as network, \
                patch.object(source, "_from_gr00t_response", wraps=source._from_gr00t_response) as decoder:
            client.awake()
            clock = self.run_loop(client)
        network.assert_not_called()
        self.assertEqual(decoder.call_count, 53)
        self.assertEqual(decoder.call_args_list[-1].args[0]["pos_delta"].shape, (1, 5, 3))
        sent = client.connection.apply_action_waypoint.call_args_list
        self.assertEqual(len(sent), 421)
        for call, frame in zip(sent, source.episode["frames"]):
            actual = call.args[0].model_dump(exclude={"kind"})
            for key, expected in action_kwargs(frame["action"]).items():
                self.assertAlmostEqual(actual[key], expected, places=12)
        self.assertEqual(clock.sleeps.count(.1), 54)  # Includes the final EOF request.
        self.assertEqual(clock.sleeps.count(.025), 53)
        self.assertEqual(client.connection.pause.call_count, 54)
        self.assertEqual(source.cursor, 421)
        self.assertFalse(client.predicting)
        stop = client.connection.handle_cartesian_delta.call_args.args[0]
        self.assertEqual((stop.dx, stop.dy, stop.dz), (0, 0, 0))

    def test_floor_stops_loop_before_action(self):
        client, source = self.make_client()
        source.min_z = .3
        self.run_loop(client)
        client.connection.apply_action_waypoint.assert_not_called()
        client.connection.handle_cartesian_delta.assert_called_once()

    def test_custom_horizon_replay_preserves_every_frame(self):
        client, source = self.make_client()
        source.execution_horizon = 5
        self.run_loop(client)
        self.assertEqual(source.cursor, 421)
        self.assertEqual(client.connection.apply_action_waypoint.call_count, 421)

    def test_mismatched_rate_stops_loop(self):
        client, _ = self.make_client()
        client.connection.control_period_s = .1
        self.run_loop(client)
        client.connection.apply_action_waypoint.assert_not_called()
        client.connection.handle_cartesian_delta.assert_called_once()

    def test_command_error_stops_replay(self):
        client, _ = self.make_client()
        client.connection.apply_action_waypoint.side_effect = RuntimeError("RPC failed")
        self.run_loop(client)
        client.connection.apply_action_waypoint.assert_called_once()
        client.connection.handle_cartesian_delta.assert_called_once()


if __name__ == "__main__":
    unittest.main()
