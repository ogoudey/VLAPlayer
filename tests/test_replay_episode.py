import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from replay_episode import DEFAULT_EPISODE, action_kwargs, load_episode, replay
from schemas import Pose, State


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episode = load_episode(DEFAULT_EPISODE)

    def make_connection(self):
        ref = self.episode["frames"][0]["observation.state"]
        state = State(
            joint_angles=[math.degrees(q) for q in ref[:7]],
            target_pose=Pose(x=ref[7], y=ref[8], z=ref[9],
                             theta_x=ref[10], theta_y=ref[11], theta_z=ref[12], theta_w=ref[13]),
            gripper=ref[14],
        )
        conn = MagicMock()
        conn.state.return_value = state
        conn.cyclic_thread = None
        return conn

    def test_export_and_units(self):
        self.assertEqual(len(self.episode["frames"]), 421)
        self.assertEqual(self.episode["episode_index"], 2)
        self.assertEqual(self.episode["task"], "take the eggplant off the plate")
        converted = action_kwargs([.01, -.02, .03, math.pi / 2, 0, -math.pi, .7])
        self.assertEqual(converted["dz"], .03)
        self.assertEqual(converted["d_theta_x"], 90)
        self.assertEqual(converted["d_theta_z"], -180)
        self.assertEqual(converted["gripper_command"], .7)

    def test_replay_order_and_stop(self):
        episode = copy.deepcopy(self.episode)
        episode["frames"] = episode["frames"][:3]
        conn = self.make_connection()
        with tempfile.TemporaryDirectory() as tmp, \
                patch("connections.kinova.KinovaConnection", return_value=conn), \
                patch("replay_episode.time.sleep"):
            log = Path(tmp) / "replay.json"
            replay(episode, min_z=.1, log_path=log)
            self.assertTrue(log.exists())
        conn.awake.assert_called_once_with(home=False)
        self.assertEqual([call.args[0].model_dump(exclude={"kind"})
                          for call in conn.apply_action.call_args_list],
                         [action_kwargs(f["action"]) for f in episode["frames"]])
        conn.base.Stop.assert_called_once()

    def test_floor_blocks_command_and_stops(self):
        conn = self.make_connection()
        with patch("connections.kinova.KinovaConnection", return_value=conn):
            with self.assertRaisesRegex(RuntimeError, "Stopped at tool Z"):
                replay(self.episode, min_z=.3)
        conn.apply_action.assert_not_called()
        conn.base.Stop.assert_called_once()

    def test_wrong_start_blocks_command(self):
        conn = self.make_connection()
        conn.state.return_value.joint_angles[0] += 20
        with patch("connections.kinova.KinovaConnection", return_value=conn):
            with self.assertRaisesRegex(RuntimeError, "Start joints differ"):
                replay(self.episode, min_z=.1)
        conn.apply_action.assert_not_called()
        conn.base.Stop.assert_called_once()

    def test_command_failure_stops(self):
        conn = self.make_connection()
        conn.apply_action.side_effect = RuntimeError("RPC failed")
        with patch("connections.kinova.KinovaConnection", return_value=conn):
            with self.assertRaisesRegex(RuntimeError, "RPC failed"):
                replay(self.episode, min_z=.1)
        conn.base.Stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
