"""Stream absolute poses through the controller owned by Tidybot's arm_server.py.

This client registers the existing Arm RPC without importing Tidybot or Kortex.
Run arm_server.py from the Tidybot checkout before calling awake(). Targets and
observations use Tidybot's tool frame, including its configured gripper offset.
"""

import math
from multiprocessing.managers import BaseManager
import threading

import numpy as np

from connections.connection import Connection
from schemas import Pose, PoseTarget, State


class ArmManager(BaseManager):
    pass


ArmManager.register("Arm")


class TidybotInterface(Connection):
    def __init__(self, control_period_s=0.02, *, rpc_host="localhost",
                 rpc_port=50001, authkey=b"secret password",
                 robot_ip="192.168.1.10"):
        super().__init__()
        if not math.isfinite(control_period_s) or control_period_s <= 0:
            raise ValueError("control_period_s must be finite and positive")
        self.control_period_s = control_period_s
        self.camera_ip_address = robot_ip
        self._manager = ArmManager(address=(rpc_host, rpc_port), authkey=authkey)
        self._arm = None
        self._gripper_target = None
        self._lock = threading.RLock()

    def awake(self, *, home=True):
        """Connect to the arm server; reset starts its low-level controller."""
        with self._lock:
            if self._arm is not None:
                return
            try:
                self._manager.connect()
            except OSError as exc:
                raise ConnectionError(
                    "Could not connect to Tidybot arm RPC server; start "
                    "arm_server.py in the Tidybot checkout first"
                ) from exc
            arm = self._manager.Arm()
            if home:
                arm.reset()
            # home=False requires an already initialized server/controller.
            measured = arm.get_state()
            self._gripper_target = float(np.asarray(measured["gripper_pos"]).item())
            self._arm = arm

    def _require_arm(self):
        if self._arm is None:
            raise RuntimeError("TidybotInterface.awake() must be called first")
        return self._arm

    def state(self) -> State:
        with self._lock:
            measured = self._require_arm().get_state()
        position = measured["arm_pos"]
        quat = measured["arm_quat"]  # xyzw
        state = State(
            joint_angles=np.degrees(measured["arm_joint_angles"]).tolist(),
            target_pose=Pose(
                x=float(position[0]), y=float(position[1]), z=float(position[2]),
                theta_x=float(quat[0]), theta_y=float(quat[1]),
                theta_z=float(quat[2]), theta_w=float(quat[3]),
            ),
            gripper=float(np.asarray(measured["gripper_pos"]).item()),
        )
        if self.ui is not None:
            self.ui.report_state(state)
        return state

    def apply_abs_action_tidybot(self, action: PoseTarget) -> None:
        """Submit an absolute pose without waiting for the arm to reach it.

        XYZ is in metres in the arm base frame. theta_* are quaternion
        components; gain and dt do not scale an absolute target. An omitted
        gripper command preserves the last submitted gripper target. Finite
        model predictions are clamped to the gripper's physical [0, 1] range.
        """
        if not isinstance(action, PoseTarget):
            raise TypeError(
                "apply_abs_action_tidybot requires an absolute PoseTarget; "
                f"got {type(action).__name__}"
            )
        position = np.array([action.x, action.y, action.z], dtype=float)
        quat = np.array([
            action.theta_x, action.theta_y, action.theta_z, action.theta_w,
        ], dtype=float)
        norm = np.linalg.norm(quat)
        if (not np.all(np.isfinite(position)) or
                not np.all(np.isfinite(quat)) or not np.isfinite(norm) or norm == 0):
            raise ValueError("Absolute target requires finite XYZ and a nonzero quaternion")
        quat /= norm

        with self._lock:
            arm = self._require_arm()
            gripper = (self._gripper_target if action.gripper_command is None
                       else action.gripper_command)
            if not math.isfinite(gripper):
                raise ValueError(f"Gripper target must be finite; got {gripper!r}")
            commanded_gripper = float(np.clip(gripper, 0.0, 1.0))
            if commanded_gripper != gripper:
                #print(f"[tidybot gripper] received={gripper!r}, clamped={commanded_gripper!r}")
                pass
            report = getattr(self.ui, "report_gripper_command", None)
            if callable(report):
                report(received=gripper, commanded=commanded_gripper)
            arm.execute_action({
                "arm_pos": position,
                "arm_quat": quat,
                "gripper_pos": np.array([commanded_gripper], dtype=float),
            })
            self._gripper_target = commanded_gripper

    def apply_action(self, action: PoseTarget) -> None:
        self.apply_abs_action_tidybot(action)

    def pause(self) -> None:
        """Stop submitting targets; Tidybot's command watchdog holds the arm.

        This RPC adapter does not resend targets on its own; the player's
        live target scheduler does that only while enabled. The server starts
        decelerating after 2.5 * POLICY_CONTROL_PERIOD without a new command
        (50 ms with its default 50 Hz policy period).
        """

    def move_to_home(self) -> None:
        with self._lock:
            arm = self._require_arm()
            arm.reset()
            self._gripper_target = float(np.asarray(arm.get_state()["gripper_pos"]).item())

    def is_connected(self) -> bool:
        return self._arm is not None
