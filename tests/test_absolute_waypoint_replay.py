import csv
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from clients.groot import RecordedGrootConfiguration, RecordedReplayStopped
from connections.kinova import KinovaConnection
from kortex_api.autogen.messages import Base_pb2
from replay_episode import DEFAULT_EPISODE
from schemas import Pose, PoseTarget


CSV_PATH = DEFAULT_EPISODE.parent / "intended_absolute_positions.csv"


class AbsoluteWaypointReplayTests(unittest.TestCase):
    def test_all_csv_targets_are_exact_and_orientation_stays_fixed(self):
        source = RecordedGrootConfiguration(
            DEFAULT_EPISODE, min_z=0.0, absolute_targets_csv=CSV_PATH,
        )
        pose = Pose(x=0.13, y=0, z=0.21, theta_x=0, theta_y=0, theta_z=0, theta_w=1)
        observation = SimpleNamespace(state=SimpleNamespace(target_pose=pose))
        actions = []
        while source.cursor < len(source.episode["frames"]):
            actions.extend(source.make_prediction(observation).actions)
            observation.state.target_pose = pose.model_copy(update={"theta_z": 1, "theta_w": 0})
        with CSV_PATH.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(actions), 421)
        for action, row, frame in zip(actions, rows, source.episode["frames"]):
            self.assertIsInstance(action, PoseTarget)
            self.assertEqual([action.x, action.y, action.z],
                             [float(row[f"target_{axis}_m"]) for axis in "xyz"])
            self.assertEqual((action.theta_x, action.theta_y, action.theta_z, action.theta_w),
                             (0, 0, 0, 1))
            self.assertEqual(action.gripper_command, frame["action"][6])

    def test_guard_checks_absolute_target(self):
        source = RecordedGrootConfiguration(DEFAULT_EPISODE, min_z=0.1)
        source._start_checked = True
        connection = SimpleNamespace(
            control_period_s=0.02,
            state=lambda: SimpleNamespace(target_pose=SimpleNamespace(z=0.2)),
        )
        safe = PoseTarget(x=0.2, y=0, z=0.15, theta_x=0, theta_y=0, theta_z=0, theta_w=1)
        source.check_action(connection, safe)
        with self.assertRaises(RecordedReplayStopped):
            source.check_action(connection, safe.model_copy(update={"z": 0.09}))

    def test_executor_sends_exact_target_despite_gain_and_dt(self):
        connection = KinovaConnection.__new__(KinovaConnection)
        connection.control_period_s = 0.02
        connection.max_linear_velocity = 0.15
        connection.max_angular_velocity = 30.0
        connection._waypoint_command_sequence = 0
        connection.ui = MagicMock(gain=0.07)
        current = Pose(x=0.1, y=0, z=0.2, theta_x=0, theta_y=0, theta_z=0, theta_w=1)
        target = PoseTarget(x=0.3, y=0.02, z=0.12,
                            theta_x=0, theta_y=0, theta_z=0, theta_w=1)
        connection._get_current_pose = MagicMock(side_effect=[current, Pose(**target.model_dump(exclude={"gripper_command"}))])
        connection.base = MagicMock()
        connection.base.ValidateWaypointList.return_value = SimpleNamespace(
            trajectory_error_report=SimpleNamespace(trajectory_error_elements=[]),
        )
        def execute(_):
            callback = connection.base.OnNotificationActionTopic.call_args.args[0]
            callback(Base_pb2.ActionNotification(action_event=Base_pb2.ACTION_END))
        connection.base.ExecuteWaypointTrajectory.side_effect = execute
        connection.apply_action_waypoint(target)
        sent = connection.base.ExecuteWaypointTrajectory.call_args.args[0].waypoints[0].cartesian_waypoint
        for field in ("x", "y", "z"):
            self.assertAlmostEqual(getattr(sent.pose, field), getattr(target, field), places=6)
        self.assertEqual(sent.maximum_angular_velocity, 0)
        self.assertEqual(sent.reference_frame, Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE)
        self.assertEqual(connection.ui.report_waypoint_result.call_args.kwargs["position_error_m"], 0)

    def test_already_reached_target_still_logs_measured_pose(self):
        connection = KinovaConnection.__new__(KinovaConnection)
        connection.control_period_s = 0.02
        connection._waypoint_command_sequence = 0
        connection.ui = MagicMock(gain=1.0)
        connection.base = MagicMock()
        pose = Pose(x=0.13, y=0, z=0.21, theta_x=0, theta_y=0, theta_z=0, theta_w=1)
        connection._get_current_pose = MagicMock(return_value=pose)
        connection.apply_action_waypoint(PoseTarget(**pose.model_dump()))
        target = connection.ui.report_waypoint_target.call_args.kwargs
        result = connection.ui.report_waypoint_result.call_args.kwargs
        self.assertEqual(target["command_id"], result["command_id"])
        self.assertEqual(result["pose"], pose)
        self.assertEqual(result["position_error_m"], 0)
        connection.base.ExecuteWaypointTrajectory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
