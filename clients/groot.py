from dataclasses import dataclass, field
import queue
import time as time_package
from typing import Optional
from clients.client import Client
from clients.client import ServerConfiguration
from typing import override
from schemas import Action, CartesianDelta, Observation, ActionChunk, ObservationRelativeDelta, Pose, PoseTarget, JointDelta, JointAngles
from packages.groot.server_client import PolicyClient
from packages.groot.pose import EndEffectorPose
from packages.groot.types import ActionFormat
import threading
import numpy as np
from scipy.spatial.transform import Rotation
import subprocess
import paramiko
import os
import json
from pathlib import Path

from enum import Enum

class GrootConfig(Enum):
    ABL7_FULLREL_QUAT = "ABL7_FULLREL_QUAT"
    ABL7_FULLREL_ROT6D = "FULLREL_ROT6D"
    ABL6_EEFSRC_FULLSTATE = "ABL6_EEFSRC_FULLSTATE"
    ABS_ROT6D_LORA = "ABS_ROT6D_LORA"
    THREE_TASKS_REL_ROT6D_LORA = "THREE_TASKS_REL_ROT6D_LORA"
    THREE_TASKS_ABS_DELTAS_LORA = "THREE_TASKS_ABS_DELTAS_LORA"
    THREE_TASKS_2_ABS_DELTAS_LORA = "THREE_TASKS_2_ABS_DELTAS_LORA"
    THREE_TASKS_2_FIXED_LORA = "THREE_TASKS_2_FIXED_LORA"
    EEF_ROT6D_ABS_IN_OUT = "EEF_ROT6D_ABS_IN_OUT"

    @property
    def uses_rot6d(self) -> bool:
        return "ROT6D" in self.name or self == GrootConfig.THREE_TASKS_ABS_DELTAS_LORA

    @property
    def uses_quat(self) -> bool:
        return self in {
            GrootConfig.ABL7_FULLREL_QUAT,
            GrootConfig.ABL6_EEFSRC_FULLSTATE,
            GrootConfig.THREE_TASKS_2_ABS_DELTAS_LORA,
            GrootConfig.THREE_TASKS_2_FIXED_LORA,
        }

@dataclass
class GrootN17ServerConfiguration(ServerConfiguration):
    config: GrootConfig # observation/action layout must match the server checkpoint
    execution_horizon: int = 8
    timeout_ms: int = 15000
    joint_branch: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    joint_period: float = 2 * np.pi
    q_ref: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    _client: Optional[PolicyClient] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        defaults = {
            GrootConfig.THREE_TASKS_ABS_DELTAS_LORA: "three_tasks_abs_deltas_lora",
            GrootConfig.THREE_TASKS_2_FIXED_LORA: "three_tasks_2_fixed_lora",
        }
        if self.joint_branch is None and self.config in defaults:
            path = Path(__file__).resolve().parents[1] / "dataset_meta" / defaults[self.config] / "joint_branch.json"
            self._load_reference_file(path, joint=True)
        if self.q_ref is None and self.config == GrootConfig.THREE_TASKS_2_FIXED_LORA:
            path = Path(__file__).resolve().parents[1] / "dataset_meta" / "three_tasks_2_fixed_lora" / "q_ref.json"
            self._load_reference_file(path, joint=False)

    def _ensure_client(self) -> None:
        if self._client is None:
            self._client = PolicyClient(
                host=self.ip,
                port=self.port,
                timeout_ms=self.timeout_ms,
                strict=False,  # leave observation validation to the server
            )
            print(f"Connected to GrootN17 server at {self.ip}:{self.port} with embodiment tag {self.config.value}")

    @override
    def make_prediction(self, observation: Observation) -> ActionChunk:
        self._ensure_client()
        request = self._to_gr00t_request(observation)  # still needs grounding — see below
        action, info = self._client.get_action(request)
        return self._from_gr00t_response(action, observation=observation)

    @override
    def test_health(self, timeout: float = 3.0) -> bool:
        self._ensure_client()
        print(f"Testing health of GrootN17 server at {self.ip}:{self.port}")
        return self._client.ping()

    @override
    def ping(self) -> bool:
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "1", self.ip],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return result.returncode == 0
        except Exception:
            return False
    @override
    def start_server(self):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        username = os.environ.get("GROOT_SERVER_USER_NAME")
        ssh_password = os.environ.get("GROOT_SERVER_USER_PASSWORD")
        if not username or not ssh_password:
            raise ValueError("GROOT_SERVER_USER_NAME and GROOT_SERVER_USER_PASSWORD environment variables must be set. SEE HOW MUCH YOU NEED GND TRUTH")
        client.connect(
            hostname=self.ip,
            username=username,
            password=ssh_password,  # or password=self.password
        )

        command = "conda activate env; python3 xyz"

        # Run in background via nohup so the SSH session doesn't block waiting
        # for the server process to exit, and so it survives after we disconnect.
        full_command = f"nohup bash -c '{command}' > /tmp/server.log 2>&1 &"
        stdin, stdout, stderr = client.exec_command(full_command)

        # exec_command returns immediately for backgrounded processes;
        # read exit status of the launcher itself (not the server) to confirm it fired.
        exit_status = stdout.channel.recv_exit_status()

        client.close()
        return exit_status == 0

    def load_reference_files(self, *, joint_branch_path=None, q_ref_path=None):
        """Load training references for live inference before connecting to hardware.

        Joint branch is seven lower bounds in radians; q_ref is XYZW.
        Joint objects contain window_lo and period; usage is descriptive metadata.
        Also accept a JSON array or an object containing joint_window_lo / q_ref.
        """
        if not self.config.uses_rot6d and not self.config.uses_quat:
            return
        path = joint_branch_path if self.config.uses_rot6d else q_ref_path
        flag = "--joint-branch" if self.config.uses_rot6d else "--q-ref"
        if path is None:
            raise ValueError(f"{self.config.name} requires {flag}")
        self._load_reference_file(path, joint=self.config.uses_rot6d)
        if self.config.uses_quat and joint_branch_path is not None:
            self._load_reference_file(joint_branch_path, joint=True)

    def _load_reference_file(self, path, *, joint: bool):
        flag = "--joint-branch" if joint else "--q-ref"
        try:
            data = json.loads(Path(path).read_text())
            period = 2 * np.pi
            if isinstance(data, dict):
                if joint:
                    period = float(data.get("period", 2 * np.pi))
                    if not np.isfinite(period) or period <= 0:
                        raise ValueError("period must be finite and positive")
                keys = ("window_lo", "joint_window_lo", "JOINT_WINDOW_LO") if joint else ("q_ref",)
                data = next((data[key] for key in keys if key in data), None)
            values = np.asarray(data, dtype=np.float64)
            size = 7 if joint else 4
            if values.shape != (size,) or not np.all(np.isfinite(values)):
                raise ValueError(f"expected {size} finite numeric values")
            if joint:
                self.joint_branch = values
                self.joint_period = period
            else:
                norm = np.linalg.norm(values)
                if norm == 0 or not np.isfinite(norm):
                    raise ValueError("q_ref must be a nonzero quaternion")
                self.q_ref = values / norm
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid {flag} file {path}: {exc}") from exc

    def _to_gr00t_request(self, observation: Observation):
        request = self._build_gr00t_request(observation)
        if self.joint_branch is not None:
            # Preserve the inline implementation's float32 arithmetic, including
            # which side of a branch boundary a joint falls on.
            joints = np.deg2rad(np.asarray(observation.state.joint_angles, dtype=np.float32))
            lower = self.joint_branch.astype(np.float32)
            joints = (joints - lower) % self.joint_period + lower
            request["state"]["arm_joints"] = self._add_batch_and_time_dims(joints.astype(np.float32))
        if self.q_ref is not None:
            pose = observation.state.target_pose
            quat = np.array([pose.theta_x, pose.theta_y, pose.theta_z, pose.theta_w], dtype=np.float32)
            if quat.dot(self.q_ref) < 0:
                quat = -quat
            if "arm_quat" in request["state"]:
                request["state"]["arm_quat"] = self._add_batch_and_time_dims(quat)
            else:
                # Legacy ABS_ROT6D_LORA uses XYZ + WXYZ quaternion state.
                request["state"]["eef_pose"] = self._add_batch_and_time_dims(
                    np.array([pose.x, pose.y, pose.z, quat[3], *quat[:3]], dtype=np.float32)
                )
        return request

    def _build_gr00t_request(self, observation: Observation):
        views = observation.vision.views
        wrist = views.get("onboard")
        exterior = next((v for camera_id, v in views.items() if camera_id != "onboard"), None)
        if wrist is None or exterior is None:
            raise RuntimeError(f"Expected 'onboard' plus one other camera, got: {list(views)}")
        match self.config:
            case GrootConfig.THREE_TASKS_2_ABS_DELTAS_LORA:
                pose = observation.state.target_pose

                
                return {
                    "video": {
                        "third_person": self._add_batch_dim(np.stack(exterior.images)),
                        "wrist": self._add_batch_dim(np.stack(wrist.images)),
                    },
                    "state": {
                        "arm_joints": self._add_batch_and_time_dims(
                            np.deg2rad(np.array(observation.state.joint_angles, dtype=np.float32))
                        ),
                        "arm_pos": self._add_batch_and_time_dims(np.array([pose.x, pose.y, pose.z], dtype=np.float32)),
                        "arm_quat": self._add_batch_and_time_dims(np.array([pose.theta_x, pose.theta_y, pose.theta_z, pose.theta_w], dtype=np.float32)),
                        "gripper": self._add_batch_and_time_dims(np.array([observation.state.gripper], dtype=np.float32)),
                    },
                    "language": {
                        "sub_task": [[observation.language]],
                    },
                }
            case GrootConfig.ABL6_EEFSRC_FULLSTATE:
                pose = observation.state.target_pose
                return {
                    "video": {
                        "wrist": self._add_batch_dim(np.stack(wrist.images)),
                        "third_person": self._add_batch_dim(np.stack(exterior.images)),
                    },
                    "state": {
                        "arm_joints": self._add_batch_and_time_dims(
                            np.deg2rad(np.array(observation.state.joint_angles, dtype=np.float32))
                        ),
                        "arm_pos": self._add_batch_and_time_dims(np.array([pose.x, pose.y, pose.z], dtype=np.float32)),
                        "arm_quat": self._add_batch_and_time_dims(np.array([pose.theta_x, pose.theta_y, pose.theta_z, pose.theta_w], dtype=np.float32)),
                        "gripper": self._add_batch_and_time_dims(np.array([observation.state.gripper], dtype=np.float32)),
                    },
                    "language": {
                        "sub_task": [[observation.language]],
                    },
                }
            case GrootConfig.ABL7_FULLREL_ROT6D:
                pose = observation.state.target_pose
                arm_quat = Rotation.from_euler(
                    "xyz", [pose.theta_x, pose.theta_y, pose.theta_z], degrees=True
                ).as_quat().astype(np.float32)

                joints = np.deg2rad(np.array(observation.state.joint_angles, dtype=np.float32))
                joints_sine_encoded = np.sin(joints)
                joints_cosine_encoded = np.cos(joints)
                joints_sine_cosine_encoded = np.concatenate([joints_sine_encoded, joints_cosine_encoded], axis=-1)
                print(f"[groot] {joints}")
                return {
                    "video": {
                        "wrist": self._add_batch_dim(np.stack(wrist.images)),
                        "third_person": self._add_batch_dim(np.stack(exterior.images)),
                    },
                    "state": {
                        "arm_joints": self._add_batch_and_time_dims(
                            joints
                        ),
                        "eef_pose": self._add_batch_and_time_dims(np.concatenate([np.array([pose.x, pose.y, pose.z], dtype=np.float32), self._eef_6d_rot(pose)], axis=-1)),
                        "gripper": self._add_batch_and_time_dims(np.array([observation.state.gripper], dtype=np.float32)),
                    },
                    "language": {
                        "sub_task": [[observation.language]],
                    },
                }
            case GrootConfig.ABS_ROT6D_LORA:
                pose = observation.state.target_pose

                joints = np.deg2rad(np.array(observation.state.joint_angles, dtype=np.float32))
                #joints_sine_encoded = np.sin(joints)
                #joints_cosine_encoded = np.cos(joints)
                #joints_sine_cosine_encoded = np.concatenate([joints_sine_encoded, joints_cosine_encoded], axis=-1)
                print(f"[groot] {joints}")
                return {
                    "video": {
                        "third_person": self._add_batch_dim(np.stack(exterior.images)),
                        "wrist": self._add_batch_dim(np.stack(wrist.images)),
                    },
                    "state": {
                        "arm_joints": self._add_batch_and_time_dims(
                            joints
                        ),
                        "eef_pose": self._add_batch_and_time_dims(np.array([pose.x, pose.y, pose.z, pose.theta_w, pose.theta_x, pose.theta_y, pose.theta_z], dtype=np.float32)),
                        "gripper": self._add_batch_and_time_dims(np.array([observation.state.gripper], dtype=np.float32)),
                    },
                    "language": {
                        "sub_task": [[observation.language]],
                    },
                }
            case GrootConfig.EEF_ROT6D_ABS_IN_OUT:
                pose = observation.state.target_pose
                eef_pose = np.concatenate([
                    np.array([pose.x, pose.y, pose.z], dtype=np.float32),
                    self._eef_6d_rot_first_two_rows(pose),
                ])
                return {
                    "video": {
                        "third_person": self._add_batch_dim(np.stack(exterior.images)),
                        "wrist": self._add_batch_dim(np.stack(wrist.images)),
                    },
                    "state": {
                        "arm_joints": self._add_batch_and_time_dims(
                            np.deg2rad(np.array(observation.state.joint_angles, dtype=np.float32))
                        ),
                        "eef_pose": self._add_batch_and_time_dims(eef_pose),
                        "gripper": self._add_batch_and_time_dims(
                            np.array([observation.state.gripper], dtype=np.float32)
                        ),
                    },
                    "language": {"sub_task": [[observation.language]]},
                }
            case GrootConfig.THREE_TASKS_REL_ROT6D_LORA:
                joints = np.deg2rad(np.array(observation.state.joint_angles, dtype=np.float32))

                pose = observation.state.target_pose
                rot6d = self._eef_6d_rot(pose)
                position = np.array([pose.x, pose.y, pose.z], dtype=np.float32)
                pose_9d = np.concatenate([position, rot6d])
                return_ = {
                    "video": {
                        "third_person": self._add_batch_dim(np.stack(exterior.images)),
                        "wrist": self._add_batch_dim(np.stack(wrist.images)),
                    },
                    "state": {
                        "arm_joints": self._add_batch_and_time_dims(
                            joints
                        ),
                        "eef_pose": self._add_batch_and_time_dims(pose_9d),
                        "gripper": self._add_batch_and_time_dims(np.array([observation.state.gripper], dtype=np.float32)),
                    },
                    "language": {
                        "sub_task": [[observation.language]],
                    },
                }
                return return_
            case GrootConfig.ABS_ROT6D_LORA:
                pose = observation.state.target_pose

                joints = np.deg2rad(np.array(observation.state.joint_angles, dtype=np.float32))
                #joints_sine_encoded = np.sin(joints)
                #joints_cosine_encoded = np.cos(joints)
                #joints_sine_cosine_encoded = np.concatenate([joints_sine_encoded, joints_cosine_encoded], axis=-1)
                print(f"[groot] {joints}")
                return {
                    "video": {
                        "third_person": self._add_batch_dim(np.stack(exterior.images)),
                        "wrist": self._add_batch_dim(np.stack(wrist.images)),
                    },
                    "state": {
                        "arm_joints": self._add_batch_and_time_dims(
                            joints
                        ),
                        "eef_pose": self._add_batch_and_time_dims(np.array([pose.x, pose.y, pose.z, pose.theta_w, pose.theta_x, pose.theta_y, pose.theta_z], dtype=np.float32)),
                        "gripper": self._add_batch_and_time_dims(np.array([observation.state.gripper], dtype=np.float32)),
                    },
                    "language": {
                        "sub_task": [[observation.language]],
                    },
                }
            case GrootConfig.THREE_TASKS_ABS_DELTAS_LORA:
                pose = observation.state.target_pose

                joints = np.deg2rad(np.array(observation.state.joint_angles, dtype=np.float32))
                rot6d = self._eef_6d_rot_first_two_rows(pose)
                position = np.array([pose.x, pose.y, pose.z], dtype=np.float32)
                pose_9d = np.concatenate([position, rot6d])
                return {
                    "video": {
                        "third_person": self._add_batch_dim(np.stack(exterior.images)),
                        "wrist": self._add_batch_dim(np.stack(wrist.images)),
                    },
                    "state": {
                        "arm_joints": self._add_batch_and_time_dims(
                            joints
                        ),
                        "eef_pose": self._add_batch_and_time_dims(pose_9d),
                        "gripper": self._add_batch_and_time_dims(np.array([observation.state.gripper], dtype=np.float32)),
                    },
                    "language": {
                        "sub_task": [[observation.language]],
                    },
                }
            case GrootConfig.THREE_TASKS_2_FIXED_LORA:
                pose = observation.state.target_pose

                joints = np.deg2rad(np.array(observation.state.joint_angles, dtype=np.float32))


                quat = np.array([pose.theta_x, pose.theta_y, pose.theta_z, pose.theta_w])
                print(f"[torequest] {observation.language}")
                return {
                    "video": {
                        "third_person": self._add_batch_dim(np.stack(exterior.images)),
                        "wrist": self._add_batch_dim(np.stack(wrist.images))
                    },
                    "state": {
                        "arm_joints": self._add_batch_and_time_dims(
                            joints
                        ),
                        "arm_pos": self._add_batch_and_time_dims(np.array([pose.x, pose.y, pose.z], dtype=np.float32)),
                        "arm_quat": self._add_batch_and_time_dims(np.array(quat, dtype=np.float32)),
                        "gripper": self._add_batch_and_time_dims(np.array([observation.state.gripper], dtype=np.float32)),
                    },
                    "language": {
                        "sub_task": [[observation.language]],
                    },
                }

    def _eef_6d_rot_first_two_columns(self, pose: Pose) -> np.ndarray:
        rot = Rotation.from_quat([pose.theta_x, pose.theta_y, pose.theta_z, pose.theta_w])
        R = rot.as_matrix()
        # First two columns, laid out [col0, col1] to mirror rot6d_to_quaternion
        return R[:2, :].reshape(6).astype(np.float32)
    
    def _eef_6d_rot_first_two_rows(self, pose: Pose) -> np.ndarray:
        rot = Rotation.from_quat([pose.theta_x, pose.theta_y, pose.theta_z, pose.theta_w])
        R = rot.as_matrix()
        # First two rows, laid out [row0, row1] to mirror rot6d_to_quaternion
        return np.concatenate([R[0, :], R[1, :]]).astype(np.float32)
    
    def _eef_9d(self, pose: Pose) -> np.ndarray:
        ee_pose = EndEffectorPose(
            translation=[pose.x, pose.y, pose.z],
            rotation=[pose.theta_x, pose.theta_y, pose.theta_z],
            rotation_type="euler",
            rotation_order="xyz",   # confirms what we assumed for TwistCommand's frame too — scipy's lowercase "xyz" is extrinsic/fixed-axis, matching Kinova's own convention
            degrees=True,
        )
        return ee_pose.xyz_rot6d.astype(np.float32)

    def _add_batch_dim(self, array: np.ndarray) -> np.ndarray:
        """Adds the leading B=1 axis every modality needs — GR00T processes a
        'batch of examples' even for one live observation."""
        return array[np.newaxis, ...]

    def _add_batch_and_time_dims(self, array: np.ndarray) -> np.ndarray:
        """For a modality with no real history (T=1, like state) — adds both
        B=1 and T=1. Don't use this on video: its T axis is real (from the
        frame history buffer), so it only needs _add_batch_dim."""
        return array[np.newaxis, np.newaxis, ...]

    def _from_gr00t_response(self, response: dict, observation: Optional[Observation] = None) -> ActionChunk:
        match self.config:
            case GrootConfig.EEF_ROT6D_ABS_IN_OUT:
                poses = np.asarray(response["eef_pose_targets"])[0]  # (T, 9), absolute XYZ + row rot6D
                grippers = np.asarray(response["gripper_targets"])[0]  # (T, 1)
                actions = []
                for pose, gripper in zip(poses, grippers):
                    eef = EndEffectorPose.from_action_format(pose, ActionFormat.XYZ_ROT6D)
                    quat = eef.to_rotation("quat", "xyzw")
                    actions.append(PoseTarget(
                        x=float(pose[0]), y=float(pose[1]), z=float(pose[2]),
                        theta_x=float(quat[0]), theta_y=float(quat[1]),
                        theta_z=float(quat[2]), theta_w=float(quat[3]),
                        gripper_command=float(gripper[0]),
                    ))
                return ActionChunk(actions=actions)
            case GrootConfig.THREE_TASKS_2_ABS_DELTAS_LORA | GrootConfig.THREE_TASKS_2_FIXED_LORA:
                delta_pos = np.asarray(response["pos_delta"])[0]
                #delta_rot = np.asarray(response["rot_delta"])[0]
                delta_rot = np.degrees(np.asarray(response["rot_delta"])[0])
                gripper = np.asarray(response["gripper"])[0]
                return ActionChunk(actions=[
                    CartesianDelta(
                        dx=float(pos[0]), dy=float(pos[1]), dz=float(pos[2]),
                        d_theta_x=float(rot[0]), d_theta_y=float(rot[1]), d_theta_z=float(rot[2]),
                        gripper_command=float(gripper[0]),
                    )
                    for pos, rot, gripper in zip(delta_pos, delta_rot, gripper)
                ])
            case GrootConfig.THREE_TASKS_ABS_DELTAS_LORA:
                deltas = np.asarray(response["eef_pose_deltas"])[0]  # (T, 6)
                gripper = np.asarray(response["gripper_targets"])[0]         # (T, 1)
                return ActionChunk(actions=[
                    CartesianDelta(
                        dx=float(d[0]), dy=float(d[1]), dz=float(d[2]),
                        d_theta_x=float(d[3]), d_theta_y=float(d[4]), d_theta_z=float(d[5]),
                        gripper_command=float(g[0]),
                    )
                    for d, g in zip(deltas, gripper)
                ])
            case GrootConfig.THREE_TASKS_REL_ROT6D_LORA:
                print(f"[groot] Response: {response}")
                d9 = response['eef_pose_targets'][0]  # shape (T, 9)
                gripper = response["gripper_targets"][0]  # shape (T,)

                ac = ActionChunk(actions=[])
                for step, g in zip(d9, gripper):
                    pos = step[:3]      # (3,)
                    rot6d = step[3:9]   # (6,)
                    
                    eef_abs_quat = self.rot6d_to_quaternion(rot6d)
                    ac.actions.append(PoseTarget(
                        x=pos[0],
                        y=pos[1],
                        z=pos[2],
                        theta_w=eef_abs_quat[0],
                        theta_x=eef_abs_quat[1],
                        theta_y=eef_abs_quat[2],
                        theta_z=eef_abs_quat[3],
                        gripper_command=g[0]
                    ))
                return ac
            case GrootConfig.ABL7_FULLREL_QUAT:
                print(f"[groot] Response: {response}")
                
                eef_obs_relative_delta = np.asarray(response["eef_delta"])[0]
                gripper_delta = np.asarray(response["gripper_delta"])[0]

                ac = ActionChunk(actions=[])

                observation_pose = observation.state.target_pose
                observation_gripper = observation.state.gripper
                projected_pose = observation.state.target_pose
                projected_gripper = observation.state.gripper
                for eef_d, g in zip(eef_obs_relative_delta, gripper_delta):
                    next_pose = PoseTarget(
                        x=observation_pose.x + eef_d[0],
                        y=observation_pose.y + eef_d[1],
                        z=observation_pose.z + eef_d[2],
                        theta_x=observation_pose.theta_x + eef_d[3],
                        theta_y=observation_pose.theta_y + eef_d[4],
                        theta_z=observation_pose.theta_z + eef_d[5],
                        gripper_command=observation_gripper + g[0]
                    )
                    
                    ac.actions.append(
                        ObservationRelativeDelta(
                            dx=next_pose.x - projected_pose.x,
                            dy=next_pose.y - projected_pose.y,
                            dz=next_pose.z - projected_pose.z,
                            d_theta_x=next_pose.theta_x - projected_pose.theta_x,
                            d_theta_y=next_pose.theta_y - projected_pose.theta_y,
                            d_theta_z=next_pose.theta_z - projected_pose.theta_z,
                            gripper_obs_rel_delta=g[0],
                        )
                    )

                    projected_pose.x += eef_d[0]
                    projected_pose.y += eef_d[1]
                    projected_pose.z += eef_d[2]
                    projected_pose.theta_x += eef_d[3]
                    projected_pose.theta_y += eef_d[4]
                    projected_pose.theta_z += eef_d[5]
                    projected_gripper += g[0]
                    
                
                return ac
            
            case GrootConfig.ABL7_FULLREL_ROT6D:
                d9 = response['eef_pose']
                eef_obs_relative_pos_delta = np.asarray(d9[:, :3])[0]
                rotv6d_delta = np.asarray(d9[:, 3:9])[0]
                eef_obs_relative_rot_delta = self._rot6d_to_euler(rotv6d_delta)

                print(f"[groot] Response: {response}")
                                
                gripper_delta = np.asarray(response["gripper_delta"])[0]

                ac = ActionChunk(actions=[])

                observation_pose = observation.state.target_pose
                observation_gripper = observation.state.gripper
                projected_pose = observation.state.target_pose
                projected_gripper = observation.state.gripper
                for eef_pos_d, eef_rot_d, g in zip(eef_obs_relative_pos_delta, eef_obs_relative_rot_delta, gripper_delta):
                    next_pose = PoseTarget(
                        x=observation_pose.x + eef_pos_d[0],
                        y=observation_pose.y + eef_pos_d[1],
                        z=observation_pose.z + eef_pos_d[2],
                        theta_x=observation_pose.theta_x + eef_rot_d[3],
                        theta_y=observation_pose.theta_y + eef_rot_d[4],
                        theta_z=observation_pose.theta_z + eef_rot_d[5],
                        gripper_command=observation_gripper + g[0]
                    )
                    
                    ac.actions.append(
                        ObservationRelativeDelta(
                            dx=next_pose.x - projected_pose.x,
                            dy=next_pose.y - projected_pose.y,
                            dz=next_pose.z - projected_pose.z,
                            d_theta_x=next_pose.theta_x - projected_pose.theta_x,
                            d_theta_y=next_pose.theta_y - projected_pose.theta_y,
                            d_theta_z=next_pose.theta_z - projected_pose.theta_z,
                            gripper_obs_rel_delta=g[0],
                        )
                    )

                    projected_pose.x += eef_pos_d[0]
                    projected_pose.y += eef_pos_d[1]
                    projected_pose.z += eef_pos_d[2]
                    projected_pose.theta_x += eef_rot_d[3]
                    projected_pose.theta_y += eef_rot_d[4]
                    projected_pose.theta_z += eef_rot_d[5]
                    projected_gripper += g[0]
                    
                
                return ac
            case GrootConfig.ABL6_EEFSRC_FULLSTATE:
                pos_delta = np.asarray(response["pos_delta"])[0]
                rot_delta = np.asarray(response["rot_delta"])[0]
                gripper = np.asarray(response["gripper"])[0]
                return ActionChunk(actions=[
                    CartesianDelta(
                        dx=float(pos[0]), dy=float(pos[1]), dz=float(pos[2]),
                        d_theta_x=float(rot[0]), d_theta_y=float(rot[1]), d_theta_z=float(rot[2]),
                        gripper_command=float(g[0] * 2), # IDK WHY
                    )
                    for pos, rot, g in zip(pos_delta, rot_delta, gripper)
                ])
            case GrootConfig.ABS_ROT6D_LORA:
                d9 = response['eef_pose_ik_targets']
                eef_abs_pos = np.asarray(d9[:, :3])[0]
                eef_abs_rotv6d = np.asarray(d9[:, 3:9])[0]
                
                gripper = np.asarray(response["gripper_targets"])[0]

                ac = ActionChunk(actions=[])

                for pos, rot6d, g in zip(eef_abs_pos, eef_abs_rotv6d, gripper):

                    eef_abs_quat = self.rot6d_to_quaternion(rot6d)
                    ac.append(PoseTarget(
                        x=float(pos[0]),
                        y=float(pos[1]),
                        z=float(pos[2]),
                        theta_x=float(eef_abs_quat[1]),
                        theta_y=float(eef_abs_quat[2]),
                        theta_z=float(eef_abs_quat[3]),
                        theta_w=float(eef_abs_quat[0]),
                        gripper=g
                    ))
            case _:
                raise ValueError(f"Setting not implemented for {self.config}")

    def rot6d_to_quaternion(self, rot6d: np.ndarray) -> np.ndarray:
        x_raw, y_raw = rot6d[:3], rot6d[3:]

        # Gram-Schmidt
        x = x_raw / np.linalg.norm(x_raw)
        z = np.cross(x, y_raw)
        z = z / np.linalg.norm(z)
        y = np.cross(z, x)

        rot_matrix = np.column_stack((x, y, z))

        # Convert matrix to quaternion [x, y, z, w] using SciPy, then reorder to [w, x, y, z]
        quat_xyzw = Rotation.from_matrix(rot_matrix).as_quat()
        return np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        )  # [w, x, y, z]

    def _rot6d_to_euler(self, d6):
        rot_matrix = self._rot6d_to_matrix(d6)
        r = Rotation.from_matrix(rot_matrix)
        return r.as_euler("xyz", degrees=True)

    def _rot6d_to_matrix(self, d6):
        a1, a2 = d6[:3], d6[3:6]
        b1 = a1 / np.linalg.norm(a1)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / np.linalg.norm(b2)
        b3 = np.cross(b1, b2)
        return np.stack([b1, b2, b3], axis=-1)

    def _eef_to_pose_target(self, eef: EndEffectorPose, gripper_command: float) -> PoseTarget:
        x, y, z = eef.translation
        theta_x, theta_y, theta_z = eef.to_rotation("euler", "xyz", degrees=True)
        return PoseTarget(
            x=float(x), y=float(y), z=float(z),
            theta_x=float(theta_x), theta_y=float(theta_y), theta_z=float(theta_z),
            gripper_command=gripper_command,
        )

class RecordedReplayStopped(Exception):
    """A recorded episode finished, or its replay checks stopped execution."""


class RecordedGrootConfiguration(GrootN17ServerConfiguration):
    """Supply consecutive dataset rows at the normal prediction boundary."""

    def __init__(
        self, episode_path, *, min_z: float, response_delay_s: float = 0.0,
        allow_control_period_override: bool = False,
        absolute_targets_csv=None,
    ):
        from replay_episode import load_episode

        super().__init__(ip="recorded", port=0, config=GrootConfig.THREE_TASKS_2_FIXED_LORA)
        if not np.isfinite(min_z):
            raise ValueError("Replay min_z must be finite")
        if not np.isfinite(response_delay_s) or response_delay_s < 0:
            raise ValueError("Replay response delay must be finite and nonnegative")
        self.episode = load_episode(episode_path)
        self.min_z = min_z
        self.response_delay_s = response_delay_s
        self.allow_control_period_override = allow_control_period_override
        self.cursor = 0
        self._start_checked = False
        self.absolute_targets = None
        self._fixed_orientation = None
        if absolute_targets_csv is not None:
            import csv
            from pathlib import Path

            with Path(absolute_targets_csv).open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != len(self.episode["frames"]):
                raise ValueError("Absolute target CSV must contain one row per episode frame")
            self.absolute_targets = []
            for i, row in enumerate(rows):
                xyz = [float(row[f"target_{axis}_m"]) for axis in "xyz"]
                if int(row["frame_index"]) != i or not np.all(np.isfinite(xyz)):
                    raise ValueError(f"Invalid absolute target CSV frame {i}")
                self.absolute_targets.append(xyz)

    def ping(self) -> bool:
        return True

    def test_health(self, timeout: float = 3.0) -> bool:
        return True

    def start_server(self):
        return True

    def make_prediction(self, observation: Observation) -> ActionChunk:
        frames = self.episode["frames"]
        if self.cursor >= len(frames):
            raise RecordedReplayStopped(f"Recorded episode complete ({len(frames)} frames)")
        if self.response_delay_s:
            time_package.sleep(self.response_delay_s)
        end = min(self.cursor + self.execution_horizon, len(frames))
        if self.absolute_targets is not None:
            if self._fixed_orientation is None:
                pose = observation.state.target_pose
                self._fixed_orientation = {
                    key: getattr(pose, key)
                    for key in ("theta_x", "theta_y", "theta_z", "theta_w")
                }
            chunk = ActionChunk(actions=[
                PoseTarget(
                    x=self.absolute_targets[i][0], y=self.absolute_targets[i][1],
                    z=self.absolute_targets[i][2], **self._fixed_orientation,
                    gripper_command=frames[i]["action"][6],
                ) for i in range(self.cursor, end)
            ])
        else:
            values = np.asarray([f["action"] for f in frames[self.cursor:end]], dtype=np.float64)
            # Feed the same raw response fields through the production decoder.
            response = {"pos_delta": values[None, :, :3],
                        "rot_delta": values[None, :, 3:6],
                        "gripper": values[None, :, 6:7]}
            chunk = self._from_gr00t_response(response, observation=observation)
        print(f"[recorded] episode={self.episode['episode_index']} frames={self.cursor}:{end}")
        self.cursor = end
        return chunk

    def check_action(self, connection, action):
        dt = connection.control_period_s
        if not self.allow_control_period_override and not np.isclose(dt, 1 / self.episode["fps"]):
            raise RecordedReplayStopped("Replay control period does not match dataset fps")
        state = connection.state()
        if not self._start_checked:
            expected = np.degrees(self.episode["frames"][0]["observation.state"][:7])
            measured = np.asarray(state.joint_angles)
            if measured.shape != (7,) or not np.all(np.isfinite(measured)):
                raise RecordedReplayStopped("Replay requires valid joint feedback")
            error = np.abs((measured - expected + 180) % 360 - 180)
            if np.max(error) > 5:
                raise RecordedReplayStopped(f"Replay start joints differ by {np.max(error):.1f} degrees")
            self._start_checked = True
        z = state.target_pose.z
        if isinstance(action, PoseTarget):
            next_z = action.z
        else:
            gain = connection.ui.gain if connection.ui is not None else 1.0
            vz = np.clip(gain * action.dz / dt,
                         -connection.max_linear_velocity, connection.max_linear_velocity)
            next_z = z + vz * dt
        if not np.isfinite(z) or not np.isfinite(next_z) or min(z, next_z) <= self.min_z:
            raise RecordedReplayStopped(
                f"Replay stopped at tool Z={z:.4f} m, target Z={next_z:.4f} m (min-z={self.min_z})"
            )


class GrootN17Client(Client):
    """
    Same as PI client actually
    """
    def __init__(self, remote: Optional[ServerConfiguration]=None, *, twist: bool = False,
                 tidybot: Optional[bool] = None):
        super().__init__()
        if tidybot is None:
            tidybot = not twist
        if twist and tidybot:
            raise ValueError("Choose either twist or Tidybot execution")
        self.twist = twist
        self.tidybot = tidybot
        self._target_stream = None
        self._tidybot_loop_lock = threading.Lock()
        self._stop_inference = threading.Event()
        if remote is None:
            # ignore for now
            pass
        else:
            self.server = remote

    @property
    def setting(self) -> str:
        return self.server.config
    
    def start_inference_loop(self):
        if self.tidybot and not isinstance(self.server, RecordedGrootConfiguration):
            from clients.tidybot_stream import TidybotTargetStream

            if not self._tidybot_loop_lock.acquire(blocking=False):
                self.ui.report_loop_event("Tidybot target loop is already running")
                return
            try:
                if self._target_stream is not None and self._target_stream.worker_alive:
                    self.ui.report_loop_event("Previous prediction worker is still finishing", level="warn")
                    return
                self._stop_inference = threading.Event()
                self._target_stream = TidybotTargetStream(self, self._stop_inference)
                self._target_stream.run()
            finally:
                self._tidybot_loop_lock.release()
            return

        self._stop_inference = threading.Event()
        action_queue: "queue.Queue[Action]" = queue.Queue()
        period = self.connection.control_period_s

        next_tick = time_package.monotonic()

        while not self._stop_inference.is_set():
            if not self.predicting:
                action_queue.queue.clear()
                time_package.sleep(0.05)
                next_tick = time_package.monotonic()
                continue

            #if action_queue.empty():
            if action_queue.qsize() < 1:
                self.connection.pause()  # pause the arm while we wait for a prediction
                time_package.sleep(0.1)
                self.ui.report_loop_event("Action queue empty — requesting prediction")
                try:
                    predict_start = time_package.monotonic()
                    chunk = self.server.make_prediction(observation=self.get_observation())
                    self.ui.report_client_latency((time_package.monotonic() - predict_start) * 1000)
                except RecordedReplayStopped as e:
                    self.ui.report_loop_event(str(e))
                    self.predicting = False
                    break
                except Exception as e:
                    self.ui.report_loop_event(f"Prediction request failed, retrying: {e}", level="warn")
                    time_package.sleep(0.5)
                    continue
                if chunk is None:
                    self.ui.report_loop_event("Prediction returned empty chunk", level="warn")
                    time_package.sleep(0.5)
                    continue
                action_queue.queue.clear()  # clear any stale actions from the queue
                for action in chunk.actions[:self.server.execution_horizon]:
                    action_queue.put(action)
                self.ui.report_loop_event(f"Received chunk of {len(chunk.actions)} actions")
                self.ui.report_queue_depth(action_queue.qsize())
                if self.tidybot:
                    # Prediction latency must not cause streamed targets to
                    # catch up to a deadline from before the inference gap.
                    next_tick = time_package.monotonic()

            action = action_queue.get()
            current_period = self.connection.control_period_s
            if current_period != period:
                period = current_period
                next_tick = time_package.monotonic()
            self.ui.report_queue_depth(action_queue.qsize())
            self.ui.report_applied_action(action)
            try:
                if not self.predicting:
                    action_queue.queue.clear()
                    continue
                if isinstance(self.server, RecordedGrootConfiguration):
                    self.server.check_action(self.connection, action)
                if self.tidybot:
                    self.connection.apply_abs_action_tidybot(action)
                elif self.twist:
                    self.connection.apply_action(action)
                else:
                    self.connection.apply_action_waypoint(action)
            except RecordedReplayStopped as e:
                self.ui.report_loop_event(str(e), level="warn")
                self.predicting = False
                break
            except Exception as e:
                stopping = self.tidybot or isinstance(self.server, RecordedGrootConfiguration)
                pausing = self.tidybot and not isinstance(self.server, RecordedGrootConfiguration)
                self.ui.report_loop_event(
                    f"apply_action failed, {'pausing' if pausing else 'stopping' if stopping else 'skipping'}: {e}",
                    level="warn",
                )
                if stopping:
                    self.predicting = False
                    if pausing:
                        # Keep the live loop available for Pause/Resume, but
                        # discard the failed chunk and wait for user input.
                        action_queue.queue.clear()
                        self.connection.pause()
                        continue
                    break
            
            tick_start = time_package.monotonic()
            next_tick += period
            sleep_for = next_tick - time_package.monotonic()
            
            if sleep_for > 0:
                time_package.sleep(sleep_for)
                
            else:
                next_tick = time_package.monotonic()
            self.ui.report_loop_timing(period, time_package.monotonic() - tick_start + max(sleep_for, 0))

        # Loop is exiting — make sure the arm isn't left coasting on a stale
        # velocity command from the last apply_action().
        if self.tidybot:
            self.connection.pause()
        else:
            self.connection.handle_cartesian_delta(CartesianDelta())

    def stop_inference_loop(self):
        self._stop_inference.set()
        if self._target_stream is not None:
            self._target_stream.stop()

    def stop_predicting(self):
        super().stop_predicting()
        if self._target_stream is not None:
            self._target_stream.set_enabled(False)

    def start_predicting(self):
        super().start_predicting()
        if self._target_stream is not None:
            self._target_stream.set_enabled(True)

    def get_observation(self):
        observation = Observation(
            vision = self.camera_set.get_vision(),
            state = self.connection.state(),
            language = self.language
        )
        
        return observation
