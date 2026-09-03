from dataclasses import dataclass, field
import queue
import time as time_package
from typing import Optional
from clients.client import Client
from clients.client import ServerConfiguration
from typing import override
from schemas import Action, CartesianDelta, Observation, ActionChunk, Pose
from packages.groot.server_client import PolicyClient
from packages.groot.pose import EndEffectorPose
from packages.groot.types import ActionFormat
import threading
import numpy as np

@dataclass
class GrootN17ServerConfiguration(ServerConfiguration):
    embodiment_tag: str = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"  # must match --embodiment-tag on the server
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
            print(f"Connected to GrootN17 server at {self.ip}:{self.port} with embodiment tag {self.embodiment_tag}")

    @override
    def make_prediction(self, observation: Observation) -> ActionChunk:
        self._ensure_client()
        request = self._to_gr00t_request(observation)  # still needs grounding — see below
        action, info = self._client.get_action(request)
        # `info` carries server-side diagnostics (timing etc.), same spirit
        # as openpi's server_timing/policy_timing — worth logging rather
        # than silently discarding, once you know what's actually in it.
        return self._from_gr00t_response(action)

    @override
    def test_health(self, timeout: float = 3.0) -> bool:
        self._ensure_client()
        print(f"Testing health of GrootN17 server at {self.ip}:{self.port}")
        return self._client.ping()

    def _to_gr00t_request(self, observation: Observation):
        views = observation.vision.views
        wrist = views.get("onboard")
        exterior = next((v for camera_id, v in views.items() if camera_id != "onboard"), None)
        if wrist is None or exterior is None:
            raise RuntimeError(f"Expected 'onboard' plus one other camera, got: {list(views)}")

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

    def _from_gr00t_response(self, response: dict) -> ActionChunk:
        eef_9d = np.asarray(response["eef_9d"])[0]                     # (T, 9) — drop batch dim
        gripper_position = np.asarray(response["gripper_position"])[0]  # (T, 1)
        # "joint_position" deliberately unused — see note below.

        if eef_9d.shape[0] != gripper_position.shape[0]:
            raise RuntimeError(
                f"eef_9d horizon ({eef_9d.shape[0]}) != gripper_position horizon "
                f"({gripper_position.shape[0]})"
            )

        return ActionChunk(actions=[
            self._row_to_action(eef_row, gripper_row[0])
            for eef_row, gripper_row in zip(eef_9d, gripper_position)
        ])

    def _row_to_action(self, eef_9d_row: np.ndarray, gripper_value: float) -> CartesianDelta:
        relative_pose = EndEffectorPose.from_action_format(eef_9d_row, ActionFormat.XYZ_ROT6D)
        dx, dy, dz = relative_pose.translation
        d_theta_x, d_theta_y, d_theta_z = relative_pose.to_rotation("euler", "xyz", degrees=True)

        return CartesianDelta(
            dx=float(dx), dy=float(dy), dz=float(dz),
            d_theta_x=float(d_theta_x), d_theta_y=float(d_theta_y), d_theta_z=float(d_theta_z),
            gripper_command=float(gripper_value),  # see gripper note below — do NOT reuse PI's remap on this
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


    def awake(self):
        # load the model, if it were local
        try:
            self.server.test_health()
        except Exception as e:
            print(f"Could not reach server: {e}")

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
                for action in chunk.actions:
                    action_queue.put(action)
                self.ui.report_loop_event(f"Received chunk of {len(chunk.actions)} actions")
                self.ui.report_queue_depth(action_queue.qsize())

            action = action_queue.get()
            self.ui.report_queue_depth(action_queue.qsize())
            self.ui.report_applied_action(action)
            try:
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