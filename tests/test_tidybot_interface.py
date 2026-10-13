from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from camera_set import CameraConnection
from clients.groot import GrootConfig, GrootN17Client, GrootN17ServerConfiguration
from schemas import ActionChunk, CartesianDelta, PoseTarget
from clients.tidybot_interface import TidybotInterface


class TidybotInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.rotation = Rotation.from_euler("xyz", [23, -41, 67], degrees=True)
        self.measured = {
            "arm_pos": np.array([.1, -.2, .3]),
            "arm_quat": self.rotation.as_quat(),
            "arm_joint_angles": np.linspace(-np.pi, np.pi, 7),
            "gripper_pos": np.array([.4]),
        }
        self.arm = MagicMock()
        self.arm.get_state.return_value = self.measured
        self.manager = MagicMock()
        self.manager.Arm.return_value = self.arm
        with patch("clients.tidybot_interface.ArmManager", return_value=self.manager) as factory:
            self.connection = TidybotInterface(rpc_host="robot-host", rpc_port=50002)
        factory.assert_called_once_with(
            address=("robot-host", 50002), authkey=b"secret password",
        )

    def target(self, **changes):
        quat = self.rotation.as_quat()
        values = dict(x=.4, y=-.1, z=.25, theta_x=quat[0], theta_y=quat[1],
                      theta_z=quat[2], theta_w=quat[3], gripper_command=.7)
        values.update(changes)
        return PoseTarget(**values)

    def test_awake_connects_and_resets_only_once(self):
        self.assertFalse(self.connection.is_connected())
        self.manager.connect.assert_not_called()
        self.connection.awake()
        self.connection.awake()
        self.manager.connect.assert_called_once()
        self.arm.reset.assert_called_once()
        self.assertTrue(self.connection.is_connected())

    def test_attach_to_initialized_arm_without_reset(self):
        self.connection.awake(home=False)
        self.arm.reset.assert_not_called()
        self.connection.apply_abs_action_tidybot(self.target(gripper_command=None))
        np.testing.assert_allclose(
            self.arm.execute_action.call_args.args[0]["gripper_pos"], [.4],
        )

    def test_connection_failure_explains_server_requirement(self):
        self.manager.connect.side_effect = ConnectionRefusedError()
        with self.assertRaisesRegex(ConnectionError, "arm_server.py"):
            self.connection.awake()
        self.manager.Arm.assert_not_called()
        self.assertFalse(self.connection.is_connected())

    def test_requires_awake_before_commands_or_state(self):
        for operation in (self.connection.state,
                          lambda: self.connection.apply_abs_action_tidybot(self.target())):
            with self.assertRaisesRegex(RuntimeError, "awake"):
                operation()
        self.arm.execute_action.assert_not_called()

    def test_absolute_pose_reaches_rpc_without_gain_or_dt_scaling(self):
        self.connection.awake()
        self.connection.ui = SimpleNamespace(gain=.05)
        self.connection.control_period_s = .5
        quat = 2 * self.rotation.as_quat()
        self.connection.apply_abs_action_tidybot(self.target(
            theta_x=quat[0], theta_y=quat[1], theta_z=quat[2], theta_w=quat[3],
        ))
        sent = self.arm.execute_action.call_args.args[0]
        np.testing.assert_allclose(sent["arm_pos"], [.4, -.1, .25])
        np.testing.assert_allclose(sent["arm_quat"], self.rotation.as_quat())
        np.testing.assert_allclose(sent["gripper_pos"], [.7])
        self.assertEqual(sent["gripper_pos"].shape, (1,))  # server calls .item()

    def test_gripper_is_preserved_when_omitted_and_after_failed_rpc(self):
        self.connection.awake()
        self.connection.apply_abs_action_tidybot(self.target())
        self.arm.execute_action.side_effect = RuntimeError("RPC failed")
        with self.assertRaises(RuntimeError):
            self.connection.apply_abs_action_tidybot(self.target(gripper_command=.9))
        self.arm.execute_action.side_effect = None
        self.connection.apply_abs_action_tidybot(self.target(gripper_command=None))
        np.testing.assert_allclose(
            self.arm.execute_action.call_args.args[0]["gripper_pos"], [.7],
        )

    def test_rejects_deltas_and_invalid_absolute_targets_before_rpc(self):
        self.connection.awake()
        with self.assertRaisesRegex(TypeError, "absolute PoseTarget"):
            self.connection.apply_abs_action_tidybot(CartesianDelta(dx=.01))
        for changes in (
            {"x": float("nan")}, {"z": float("inf")},
            {"theta_w": float("nan")},
            {"theta_x": 0, "theta_y": 0, "theta_z": 0, "theta_w": 0},
            {"gripper_command": float("nan")},
            {"gripper_command": float("inf")},
            {"gripper_command": float("-inf")},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.connection.apply_abs_action_tidybot(self.target(**changes))
        self.arm.execute_action.assert_not_called()

    def test_out_of_range_gripper_predictions_are_clamped_without_blocking_pose(self):
        self.connection.awake()
        self.connection.ui = MagicMock()
        for received, expected in ((-.01, 0.0), (1.01, 1.0), (.35, .35)):
            with self.subTest(received=received):
                self.connection.apply_abs_action_tidybot(self.target(gripper_command=received))
                sent = self.arm.execute_action.call_args.args[0]
                np.testing.assert_allclose(sent["arm_pos"], [.4, -.1, .25])
                np.testing.assert_allclose(sent["gripper_pos"], [expected])
                self.connection.ui.report_gripper_command.assert_called_with(
                    received=received, commanded=expected,
                )
                self.connection.apply_abs_action_tidybot(self.target(gripper_command=None))
                np.testing.assert_allclose(
                    self.arm.execute_action.call_args.args[0]["gripper_pos"], [expected],
                )

    def test_state_preserves_pose_and_converts_joint_radians_to_degrees(self):
        self.connection.awake()
        self.connection.ui = MagicMock()
        state = self.connection.state()
        np.testing.assert_allclose(state.joint_angles, [-180, -120, -60, 0, 60, 120, 180])
        np.testing.assert_allclose(
            [state.target_pose.x, state.target_pose.y, state.target_pose.z],
            self.measured["arm_pos"],
        )
        np.testing.assert_allclose(
            [state.target_pose.theta_x, state.target_pose.theta_y,
             state.target_pose.theta_z, state.target_pose.theta_w],
            self.rotation.as_quat(),
        )
        self.assertEqual(state.gripper, .4)
        self.connection.ui.report_state.assert_called_once_with(state)

    def test_pause_sends_no_targets_and_home_refreshes_gripper(self):
        self.connection.awake()
        self.connection.apply_abs_action_tidybot(self.target())
        self.arm.reset_mock()
        self.connection.pause()
        self.arm.execute_action.assert_not_called()
        self.arm.reset.assert_not_called()
        self.measured["gripper_pos"] = np.array([0.0])
        self.connection.move_to_home()
        self.arm.reset.assert_called_once()
        self.connection.apply_abs_action_tidybot(self.target(gripper_command=None))
        np.testing.assert_allclose(
            self.arm.execute_action.call_args.args[0]["gripper_pos"], [0.0],
        )

    def test_wrist_camera_uses_rpc_backend_address_without_kortex_connection(self):
        self.connection.camera_ip_address = "192.168.1.20"
        camera = CameraConnection(onboard_connection=self.connection)
        self.assertEqual(camera.rtsp_url, "rtsp://192.168.1.20/color")
        self.assertFalse(hasattr(self.connection, "tcp_connection"))

    def test_gr00t_absolute_rot6d_chunk_streams_to_rpc(self):
        from clients.tidybot_stream import PredictionResult, TidybotTargetStream
        import threading

        self.connection.awake()
        server = GrootN17ServerConfiguration(
            ip="unused", port=5555, config=GrootConfig.EEF_ROT6D_ABS_IN_OUT,
        )
        rows = self.rotation.as_matrix()[:2].reshape(6)
        chunk = server._from_gr00t_response({
            "eef_pose_targets": np.array([[np.r_[.4, -.1, .25, rows],
                                           np.r_[.45, -.1, .25, rows]]]),
            "gripper_targets": np.array([[[.2], [.7]]]),
        })
        client = GrootN17Client(remote=server, tidybot=True)
        client.connection = self.connection
        client.ui = MagicMock()
        client.get_observation = lambda: SimpleNamespace(state=self.connection.state())
        stream = TidybotTargetStream(client, threading.Event())
        client._target_stream = stream
        with patch.object(self.connection, "pause") as pause:
            stream.tick(0.0)  # seed from measured pose while inference starts
            request = stream._requests.get_nowait()
            stream._results.put(PredictionResult(request, chunk=chunk))
            stream.tick(0.0)
            stream.tick(.02)
            pause.assert_not_called()
        self.assertEqual(self.arm.execute_action.call_count, 3)
        for call, xyz, gripper in zip(self.arm.execute_action.call_args_list[1:],
                                      ([.4, -.1, .25], [.45, -.1, .25]), (.2, .7)):
            np.testing.assert_allclose(call.args[0]["arm_pos"], xyz)
            np.testing.assert_allclose(call.args[0]["arm_quat"], self.rotation.as_quat())
            np.testing.assert_allclose(call.args[0]["gripper_pos"], [gripper])

    def test_twist_and_tidybot_modes_are_mutually_exclusive(self):
        self.assertTrue(GrootN17Client().tidybot)
        self.assertFalse(GrootN17Client(twist=True).tidybot)
        self.assertFalse(GrootN17Client(tidybot=False).tidybot)
        with self.assertRaises(ValueError):
            GrootN17Client(twist=True, tidybot=True)


if __name__ == "__main__":
    unittest.main()
