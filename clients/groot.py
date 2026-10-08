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

@dataclass
class GrootN17ServerConfiguration(ServerConfiguration):
    config: GrootConfig # must match --embodiment-tag on the server
    EXECUTION_HORIZON = 8 # This should be editable in the GUI
    timeout_ms: int = 15000

    _client: Optional[PolicyClient] = field(default=None, init=False, repr=False, compare=False)

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

    def _to_gr00t_request(self, observation: Observation):
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
                JOINT_WINDOW_LO = np.array([-2.559710208569662, -2.8912901023971003, -6.894980764381135, -5.040146803847993, -3.489536766205923, -3.6565758551107805, -1.1027804930978498], dtype=np.float32)
                joints = (joints - JOINT_WINDOW_LO) % (2 * np.pi) + JOINT_WINDOW_LO
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
                JOINT_WINDOW_LO = np.array([
                    -3.023638133202688,
                    -2.8912901023971003,
                    -6.3224516868512275,
                    -5.040146803847993,
                    -3.2025910357581537,
                    -3.610507700835363,
                    -1.4728048165612897
                ], dtype=np.float32)
                joints = (joints - JOINT_WINDOW_LO) % (2 * np.pi) + JOINT_WINDOW_LO


                q_ref = np.array([
                    0.7057592503866893,
                    0.7076099545579241,
                    0.024808814852628452,
                    0.024011568248638298
                ], dtype=np.float32)
                quat = np.array([pose.theta_x, pose.theta_y, pose.theta_z, pose.theta_w])
                adjusted_quat = -quat if quat.dot(q_ref) < 0 else quat
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
                        "arm_quat": self._add_batch_and_time_dims(np.array(adjusted_quat, dtype=np.float32)),
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

class GrootN17Client(Client):
    """
    Same as PI client actually
    """
    def __init__(self, remote: Optional[ServerConfiguration]=None):
        super().__init__()
        if remote is None:
            # ignore for now
            pass
        else:
            self.server = remote

    @property
    def setting(self) -> str:
        return self.server.config
    
    def start_inference_loop(self):
        self._stop_inference = threading.Event()
        action_queue: "queue.Queue[Action]" = queue.Queue()
        period = self.connection.control_period_s

        next_tick = time_package.monotonic()

        while not self._stop_inference.is_set():
            if not self.predicting:
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
                except Exception as e:
                    self.ui.report_loop_event(f"Prediction request failed, retrying: {e}", level="warn")
                    time_package.sleep(0.5)
                    continue
                if chunk is None:
                    self.ui.report_loop_event("Prediction returned empty chunk", level="warn")
                    time_package.sleep(0.5)
                    continue
                action_queue.queue.clear()  # clear any stale actions from the queue
                for action in chunk.actions[:GrootN17ServerConfiguration.EXECUTION_HORIZON]:
                    action_queue.put(action)
                self.ui.report_loop_event(f"Received chunk of {len(chunk.actions)} actions")
                self.ui.report_queue_depth(action_queue.qsize())

            action = action_queue.get()
            self.ui.report_queue_depth(action_queue.qsize())
            self.ui.report_applied_action(action)
            try:
                if not self.predicting:
                    continue
                self.connection.apply_action(action)
            except Exception as e:
                self.ui.report_loop_event(f"apply_action failed, skipping: {e}", level="warn")
            
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
        self.connection.handle_cartesian_delta(CartesianDelta())

    def stop_inference_loop(self):
        self._stop_inference.set()

    def get_observation(self):
        observation = Observation(
            vision = self.camera_set.get_vision(),
            state = self.connection.state(),
            language = self.language
        )
        
        return observation