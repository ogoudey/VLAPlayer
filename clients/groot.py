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
            case "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT":
                return {
                    "video": {
                        "exterior_image_1_left": self._add_batch_dim(np.stack(exterior.images)),  # (1, T, H, W, C)
                        "wrist_image_left": self._add_batch_dim(np.stack(wrist.images)),
                    },
                    "state": {
                        "eef_9d": self._add_batch_and_time_dims(
                            self._eef_9d(observation.state.target_pose)
                        ),
                        "gripper_position": self._add_batch_and_time_dims(
                            np.array([observation.state.gripper], dtype=np.float32)
                        ),
                        "joint_position": self._add_batch_and_time_dims(
                            np.array(observation.state.joint_angles, dtype=np.float32)
                        ),
                    },
                    "language": {
                        "annotation.language.language_instruction": [[observation.language]],  # (B=1, T=1) as plain nested lists — not an ndarray
                    },
                }
            case "OLD_NEW_EMBODIMENT": # something like HRILAB_DELTAS_TO_FROM_UT
                pose = observation.state.target_pose
                arm_quat = Rotation.from_euler(
                    "xyz", [pose.theta_x, pose.theta_y, pose.theta_z], degrees=True
                ).as_quat().astype(np.float32)

                return {
                    "video": {
                        "wrist_image": self._add_batch_dim(np.stack(wrist.images)),
                        "third_person_image": self._add_batch_dim(np.stack(exterior.images)),
                    },
                    "state": {
                        "arm_joint_angles": self._add_batch_and_time_dims(
                            np.deg2rad(np.array(observation.state.joint_angles, dtype=np.float32))
                        ),
                        "arm_pos": self._add_batch_and_time_dims(np.array([pose.x, pose.y, pose.z], dtype=np.float32)),
                        "arm_quat": self._add_batch_and_time_dims(arm_quat),
                        "gripper_pos": self._add_batch_and_time_dims(np.array([observation.state.gripper], dtype=np.float32)),
                    },
                    "language": {
                        "annotation.human.task_description": [[observation.language]],
                    },
                }
            case GrootConfig.ABL6_EEFSRC_FULLSTATE:
                pose = observation.state.target_pose
                arm_quat = Rotation.from_euler(
                    "xyz", [pose.theta_x, pose.theta_y, pose.theta_z], degrees=True
                ).as_quat().astype(np.float32)

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
                        "arm_quat": self._add_batch_and_time_dims(arm_quat),
                        "gripper": self._add_batch_and_time_dims(np.array([observation.state.gripper], dtype=np.float32)),
                    },
                    "language": {
                        "sub_task": [[observation.language]],
                    },
                }
            case GrootConfig.ABL7_FULLREL_QUAT:
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
                        "third_person": self._add_batch_dim(np.stack(exterior.images)),
                        "wrist": self._add_batch_dim(np.stack(wrist.images)),
                    },
                    "state": {
                        "arm_joints": self._add_batch_and_time_dims(
                            joints
                        ),
                        "eef_pose": self._add_batch_and_time_dims(np.concatenate([np.array([pose.x, pose.y, pose.z], dtype=np.float32), arm_quat], axis=-1)),
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

    def _eef_6d_rot(self, pose: Pose) -> np.ndarray:
        ee_pose = EndEffectorPose(
            translation=[pose.x, pose.y, pose.z],
            rotation=[pose.theta_x, pose.theta_y, pose.theta_z],
            rotation_type="euler",
            rotation_order="xyz",   # confirms what we assumed for TwistCommand's frame too — scipy's lowercase "xyz" is extrinsic/fixed-axis, matching Kinova's own convention
            degrees=True,
        )
        return ee_pose.rot6d.astype(np.float32)

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
            case "NEW_EMBODIMENT": # something like HRILAB_DELTAS_TO_FROM_UT
                try:
                    joint_angles = np.asarray(response["joint_target"])[0]
                    gripper = np.asarray(response["gripper"])[0]
                except KeyError as e:
                    print(f"[groot] KeyError: {e} in response: {response}")
                return ActionChunk(actions=[
                    CartesianDelta(
                        
                    )
                    for angles, g in zip(joint_angles, gripper)
                ])
            case _:
                raise ValueError(f"Setting not implemented for {self.config}")

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

            if action_queue.empty():
                self.connection.pause()  # pause the arm while we wait for a prediction
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