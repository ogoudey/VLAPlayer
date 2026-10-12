import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from clients.groot import GrootConfig, GrootN17ServerConfiguration
from schemas import Pose


class GrootReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "reference.json"

    def write(self, data):
        self.path.write_text(json.dumps(data))
        return self.path

    def server(self, config):
        return GrootN17ServerConfiguration(ip="unused", port=5555, config=config)

    def observation(self, quat):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        return SimpleNamespace(
            vision=SimpleNamespace(views={
                "onboard": SimpleNamespace(images=[image]),
                "external": SimpleNamespace(images=[image]),
            }),
            state=SimpleNamespace(
                joint_angles=[350] * 7, gripper=.4,
                target_pose=Pose(x=.1, y=.2, z=.3, theta_x=quat[0],
                                 theta_y=quat[1], theta_z=quat[2], theta_w=quat[3]),
            ), language="pick up the object",
        )

    def test_joint_branch_applies_training_window_to_live_state(self):
        server = self.server(GrootConfig.EEF_ROT6D_ABS_IN_OUT)
        server.load_reference_files(joint_branch_path=self.write({"joint_window_lo": [-np.pi] * 7}))
        request = server._to_gr00t_request(self.observation([0, 0, 0, 1]))
        np.testing.assert_allclose(request["state"]["arm_joints"], np.deg2rad(-10), atol=3e-7)

    def test_q_ref_replaces_fixed_reference_and_fixes_sign(self):
        for config in (GrootConfig.THREE_TASKS_2_FIXED_LORA, GrootConfig.ABL6_EEFSRC_FULLSTATE,
                       GrootConfig.THREE_TASKS_2_ABS_DELTAS_LORA):
            server = self.server(config)
            server.load_reference_files(q_ref_path=self.write({"q_ref": [0, 0, 0, 2]}))
            request = server._to_gr00t_request(self.observation([0, 0, 0, -1]))
            np.testing.assert_allclose(request["state"]["arm_quat"], [[[0, 0, 0, 1]]])

    def test_extracted_q_ref_preserves_historical_sign_alignment(self):
        legacy = np.array([
            .7057592503866893, .7076099545579241,
            .024808814852628452, .024011568248638298,
        ], dtype=np.float32)
        server = self.server(GrootConfig.THREE_TASKS_2_FIXED_LORA)
        path = Path(__file__).resolve().parents[1] / "dataset_meta/three_tasks_2_fixed_lora/q_ref.json"
        stored = np.asarray(json.loads(path.read_text())["q_ref"], dtype=np.float32)
        np.testing.assert_array_equal(stored, legacy)
        rng = np.random.default_rng(23)
        for quat in [legacy, -legacy, *rng.normal(size=(30, 4))]:
            quat = quat / np.linalg.norm(quat)
            expected = -quat if quat.dot(legacy) < 0 else quat
            request = server._to_gr00t_request(self.observation(quat))
            np.testing.assert_array_equal(
                request["state"]["arm_quat"][0, 0], expected.astype(np.float32)
            )

    def test_required_file_depends_on_config(self):
        for config, flag in ((GrootConfig.EEF_ROT6D_ABS_IN_OUT, "--joint-branch"),
                             (GrootConfig.THREE_TASKS_2_FIXED_LORA, "--q-ref")):
            with self.assertRaisesRegex(ValueError, flag):
                self.server(config).load_reference_files()

    def test_invalid_reference_files(self):
        for config, keyword, invalid in (
            (GrootConfig.EEF_ROT6D_ABS_IN_OUT, "joint_branch_path", [[0] * 6, [float("nan")] * 7, {}]),
            (GrootConfig.THREE_TASKS_2_FIXED_LORA, "q_ref_path", [[0] * 4, [1] * 3, [float("inf")] * 4]),
        ):
            for data in invalid:
                with self.subTest(config=config, data=data), self.assertRaises(ValueError):
                    self.server(config).load_reference_files(**{keyword: self.write(data)})
        self.path.write_text("")
        with self.assertRaises(ValueError):
            self.server(GrootConfig.EEF_ROT6D_ABS_IN_OUT).load_reference_files(joint_branch_path=self.path)

    def test_bare_arrays_are_supported(self):
        server = self.server(GrootConfig.EEF_ROT6D_ABS_IN_OUT)
        server.load_reference_files(joint_branch_path=self.write([-np.pi] * 7))
        np.testing.assert_allclose(server.joint_branch, -np.pi)

    def test_window_lo_and_period_are_used(self):
        server = self.server(GrootConfig.EEF_ROT6D_ABS_IN_OUT)
        server.load_reference_files(joint_branch_path=self.write({
            "period": 4.0, "window_lo": [-2.0] * 7, "usage": "metadata only",
        }))
        request = server._to_gr00t_request(self.observation([0, 0, 0, 1]))
        expected = ((np.deg2rad(350) + 2) % 4) - 2
        np.testing.assert_allclose(request["state"]["arm_joints"], expected, atol=1e-7)
        self.assertEqual(server.joint_period, 4.0)

    def test_invalid_period_is_rejected(self):
        for period in (0, -1, "nan", "inf", None):
            with self.subTest(period=period), self.assertRaises(ValueError):
                self.server(GrootConfig.EEF_ROT6D_ABS_IN_OUT).load_reference_files(
                    joint_branch_path=self.write({"window_lo": [0] * 7, "period": period})
                )

    def test_rot6d_never_requires_or_loads_q_ref(self):
        for config in GrootConfig:
            if config.uses_rot6d:
                self.assertFalse(config.uses_quat)
                server = self.server(config)
                server.load_reference_files(
                    joint_branch_path=self.write([-np.pi] * 7),
                    q_ref_path=Path(self.temp.name) / "missing-q-ref.json",
                )
                self.assertIsNone(server.q_ref)

    def test_extracted_branches_match_old_inline_arithmetic_exactly(self):
        legacy = {
            GrootConfig.THREE_TASKS_ABS_DELTAS_LORA: [
                -2.559710208569662, -2.8912901023971003, -6.894980764381135,
                -5.040146803847993, -3.489536766205923, -3.6565758551107805,
                -1.1027804930978498,
            ],
            GrootConfig.THREE_TASKS_2_FIXED_LORA: [
                -3.023638133202688, -2.8912901023971003, -6.3224516868512275,
                -5.040146803847993, -3.2025910357581537, -3.610507700835363,
                -1.4728048165612897,
            ],
        }
        rng = np.random.default_rng(17)
        for config, values in legacy.items():
            server = self.server(config)
            np.testing.assert_array_equal(server.joint_branch, values)
            lower = np.array(values, dtype=np.float32)
            boundary_degrees = np.rad2deg(lower)
            samples = [np.zeros(7), [350] * 7, boundary_degrees,
                       np.nextafter(boundary_degrees, -np.inf),
                       np.nextafter(boundary_degrees, np.inf),
                       *rng.uniform(-1080, 1080, size=(30, 7))]
            for angles in samples:
                observation = self.observation([0, 0, 0, 1])
                observation.state.joint_angles = angles
                raw = np.deg2rad(np.array(angles, dtype=np.float32))
                expected = (raw - lower) % (2 * np.pi) + lower
                request = server._to_gr00t_request(observation)
                np.testing.assert_array_equal(request["state"]["arm_joints"][0, 0], expected)

    def test_explicit_joint_branch_overrides_extracted_quat_default(self):
        server = self.server(GrootConfig.THREE_TASKS_2_FIXED_LORA)
        branch_path = self.write({"window_lo": [-2] * 7, "period": 4})
        q_path = Path(self.temp.name) / "q_ref.json"
        q_path.write_text(json.dumps({"q_ref": [0, 0, 0, 1]}))
        server.load_reference_files(joint_branch_path=branch_path, q_ref_path=q_path)
        self.assertEqual(server.joint_period, 4)
        np.testing.assert_array_equal(server.joint_branch, [-2] * 7)
