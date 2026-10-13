import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from clients.groot import GrootConfig, GrootN17Client, GrootN17ServerConfiguration
from schemas import ActionChunk, CartesianDelta, Pose, PoseTarget


class GrootRot6DTests(unittest.TestCase):
    def setUp(self):
        self.server = GrootN17ServerConfiguration(
            ip="unused", port=5555, config=GrootConfig.EEF_ROT6D_ABS_IN_OUT
        )

    def test_state_matches_training_rows_and_quaternion_sign(self):
        rotation = Rotation.from_euler("xyz", [23, -41, 67], degrees=True)
        for sign in (1, -1):
            quat = sign * rotation.as_quat()
            image = np.zeros((4, 5, 3), dtype=np.uint8)
            observation = SimpleNamespace(
                vision=SimpleNamespace(views={
                    "onboard": SimpleNamespace(images=[image]),
                    "external": SimpleNamespace(images=[image]),
                }),
                state=SimpleNamespace(
                    target_pose=Pose(x=.1, y=.2, z=.3, theta_x=quat[0],
                                     theta_y=quat[1], theta_z=quat[2], theta_w=quat[3]),
                    joint_angles=[180] * 7, gripper=.4,
                ),
                language="pick up the object",
            )
            request = self.server._to_gr00t_request(observation)
            self.assertEqual(request["state"]["eef_pose"].shape, (1, 1, 9))
            np.testing.assert_allclose(request["state"]["eef_pose"][0, 0, :3], [.1, .2, .3])
            np.testing.assert_allclose(
                request["state"]["eef_pose"][0, 0, 3:],
                rotation.as_matrix()[:2, :].reshape(6), atol=1e-7,
            )
            np.testing.assert_allclose(request["state"]["arm_joints"], np.pi)
            self.assertEqual(request["video"]["wrist"].shape, (1, 1, 4, 5, 3))
            self.assertEqual(request["language"]["sub_task"], [[observation.language]])

    def test_absolute_actions_decode_rows_without_inverting_rotation(self):
        rotations = Rotation.from_euler("xyz", [[23, -41, 67], [-80, 17, 121]], degrees=True)
        positions = np.array([[.1, .2, .3], [.4, .5, .6]])
        poses = np.concatenate([
            positions, rotations.as_matrix()[:, :2, :].reshape(2, 6)
        ], axis=1)
        chunk = self.server._from_gr00t_response({
            "eef_pose_targets": poses[None],
            "gripper_targets": np.array([[[.2], [.7]]]),
        })
        self.assertEqual(len(chunk.actions), 2)
        for i, action in enumerate(chunk.actions):
            self.assertIsInstance(action, PoseTarget)
            np.testing.assert_allclose([action.x, action.y, action.z], positions[i])
            actual = Rotation.from_quat([
                action.theta_x, action.theta_y, action.theta_z, action.theta_w
            ])
            np.testing.assert_allclose(actual.as_matrix(), rotations[i].as_matrix(), atol=1e-7)
            self.assertAlmostEqual(action.gripper_command, [.2, .7][i])

    def test_live_rot6d_actions_use_waypoints(self):
        chunk = self.server._from_gr00t_response({
            "eef_pose_targets": np.array([[
                [.1, .2, .3, 1, 0, 0, 0, 1, 0],
                [.4, .5, .6, 1, 0, 0, 0, 1, 0],
            ]]),
            "gripper_targets": np.array([[[.2], [.7]]]),
        })
        client = GrootN17Client(remote=self.server, tidybot=False)
        client.connection = MagicMock(control_period_s=.02)
        client.ui = MagicMock()
        client.get_observation = MagicMock()

        def apply_waypoint(action):
            if action is chunk.actions[-1]:
                client.stop_inference_loop()

        client.connection.apply_action_waypoint.side_effect = apply_waypoint
        with patch.object(self.server, "make_prediction", return_value=chunk) as predict, \
                patch("clients.groot.time_package.sleep"):
            client.start_inference_loop()
        predict.assert_called_once()
        self.assertEqual(
            [call.args[0] for call in client.connection.apply_action_waypoint.call_args_list],
            chunk.actions,
        )
        client.connection.apply_action.assert_not_called()
        client.connection.handle_cartesian_delta.assert_called_once()

    def test_explicit_waypoints_and_twist_for_both_action_layouts(self):
        for config in (GrootConfig.THREE_TASKS_2_FIXED_LORA, GrootConfig.EEF_ROT6D_ABS_IN_OUT):
            for twist in (False, True):
                with self.subTest(config=config, twist=twist):
                    self.server.config = config
                    if config == GrootConfig.THREE_TASKS_2_FIXED_LORA:
                        action = CartesianDelta(dx=.001, gripper_command=.4)
                    else:
                        action = PoseTarget(x=.1, y=.2, z=.3, theta_x=0,
                                            theta_y=0, theta_z=0, theta_w=1)
                    client = GrootN17Client(remote=self.server, twist=twist, tidybot=False)
                    client.connection = MagicMock(control_period_s=.02)
                    client.ui = MagicMock()
                    client.get_observation = MagicMock()
                    selected = (client.connection.apply_action if twist
                                else client.connection.apply_action_waypoint)
                    other = (client.connection.apply_action_waypoint if twist
                             else client.connection.apply_action)
                    selected.side_effect = lambda _: client.stop_inference_loop()
                    with patch.object(self.server, "make_prediction", return_value=ActionChunk(actions=[action])), \
                            patch("clients.groot.time_package.sleep"):
                        client.start_inference_loop()
                    selected.assert_called_once_with(action)
                    other.assert_not_called()

    def test_dt_change_updates_running_loop_on_next_action(self):
        actions = [CartesianDelta(dx=.001), CartesianDelta(dx=.002)]
        client = GrootN17Client(remote=self.server, tidybot=False)
        client.connection = MagicMock(control_period_s=.02)
        client.ui = MagicMock()
        client.get_observation = MagicMock()

        def apply_waypoint(action):
            if action is actions[0]:
                client.connection.control_period_s = .5
            else:
                client.stop_inference_loop()

        client.connection.apply_action_waypoint.side_effect = apply_waypoint
        clock = SimpleNamespace(monotonic=lambda: 0, sleep=MagicMock())
        with patch.object(self.server, "make_prediction", return_value=ActionChunk(actions=actions)), \
                patch("clients.groot.time_package", clock):
            client.start_inference_loop()
        self.assertEqual([call.args[0] for call in clock.sleep.call_args_list], [.1, .02, .5])
        self.assertEqual([call.args[0] for call in client.ui.report_loop_timing.call_args_list], [.02, .5])

    def test_horizon_truncates_chunk_and_live_change_applies_to_next_chunk(self):
        self.server.execution_horizon = 2
        first = [CartesianDelta(dx=i * .001) for i in range(4)]
        second = [CartesianDelta(dx=.01 + i * .001) for i in range(4)]
        client = GrootN17Client(remote=self.server, tidybot=False)
        client.connection = MagicMock(control_period_s=.02)
        client.ui = MagicMock()
        client.get_observation = MagicMock()

        def apply_waypoint(action):
            if action is first[0]:
                self.server.execution_horizon = 1
            if action is second[0]:
                client.stop_inference_loop()

        client.connection.apply_action_waypoint.side_effect = apply_waypoint
        with patch.object(self.server, "make_prediction", side_effect=[
            ActionChunk(actions=first), ActionChunk(actions=second)
        ]) as predict, patch("clients.groot.time_package.sleep"):
            client.start_inference_loop()
        self.assertEqual(predict.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in client.connection.apply_action_waypoint.call_args_list],
            first[:2] + second[:1],
        )


if __name__ == "__main__":
    unittest.main()
