import math

from client import Client, ServerConfiguration
from typing import Optional
from schemas import Observation, Vision, ActionChunk, State, Action, CartesianDelta, JointVelocities7DOF
import queue
import time
import threading
from dataclasses import dataclass, field
from openpi_client import msgpack_numpy
import websockets.exceptions
import websockets.sync.client
from typing import override
import numpy as np
from openpi_client import image_tools
import http
import math

@dataclass
class Pi05ServerConfiguration(ServerConfiguration):
    # Connection state — not constructor args, so declared as init=False fields.
    _ws: Optional[websockets.sync.client.ClientConnection] = field(default=None, init=False, repr=False, compare=False)
    _server_metadata: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _packer: msgpack_numpy.Packer = field(default_factory=msgpack_numpy.Packer, init=False, repr=False, compare=False)

    @property
    def _uri(self) -> str:
        return f"ws://{self.ip}:{self.port}"

    def _ensure_connected(self):
        if self._ws is not None:
            return
        conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
        # Server sends a metadata dict as the very first message on connect.
        self._server_metadata = msgpack_numpy.unpackb(conn.recv())
        self._ws = conn
    
    @override
    def make_prediction(self, observation: Observation) -> ActionChunk:
        print(f"Ensuring connection...")
        self._ensure_connected()
        
        # Possibly convert to OpenPI client's schema.
        pi_request = self._to_pi_request(observation)
        payload = self._packer.pack(pi_request)
        try:
            self._ws.send(payload)
            response = self._ws.recv()
        except websockets.exceptions.ConnectionClosed:
            # Reconnect once and retry, rather than let one dropped
            # connection permanently break the inference loop.
            self._ws = None
            self._ensure_connected()
            self._ws.send(payload)
            response = self._ws.recv()

        if isinstance(response, str):
            # Server sends a plain string (the traceback) instead of a
            # packed action on internal error — see the server's `except
            # Exception` branch.
            raise RuntimeError(f"Inference server error:\n{response}")
        response_dict = msgpack_numpy.unpackb(response)
        return self._to_action_chunk(response_dict)

    def _to_action_chunk(self, response: dict) -> ActionChunk:
        actions = np.asarray(response["actions"])  # (chunk_size, 7)
        return ActionChunk(actions=[self._row_to_action(row) for row in actions])

    def _row_to_action(self, row: np.ndarray) -> CartesianDelta:
        match self.setting:
            case "DROID":
                return JointVelocities7DOF(
                    j0=float(row[0]), 
                    j1=float(row[1]), 
                    j2=float(row[2]),
                    j3=float(row[3]), 
                    j4=float(row[4]), 
                    j5=float(row[5]),
                    j6=float(row[6]),
                    gripper_command=float(row[7]),
                )
            case "LIBERO":
                return CartesianDelta(
                    dx=float(row[0]), dy=float(row[1]), dz=float(row[2]),
                    d_theta_x=float(row[3]), d_theta_y=float(row[4]), d_theta_z=float(row[5]),

                    gripper_command=float(row[6]),
                )

    def _to_pi_request(self, observation: Observation) -> dict:
        views = observation.vision.views
        # The Gen3's onboard camera sits at the wrist/end-effector — that's the
        # model's "wrist" view. The D435 is your exterior/scene view.
        wrist = views.get("onboard")
        exterior = next((v for camera_id, v in views.items() if camera_id != "onboard"), None)
        if wrist is None or exterior is None:
            raise RuntimeError(f"Expected 'onboard' plus one other camera, got: {list(views)}")

        match self.setting:
            case "DROID":
                joint_position = np.array(
                    [math.radians(angle) for angle in observation.state.joint_angles],
                    dtype=np.float32,
                )
                gripper_position = np.array([observation.state.gripper], dtype=np.float32)

                return {
                    "observation/exterior_image_1_left": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(exterior.image, 224, 224)
                    ),
                    "observation/wrist_image_left": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist.image, 224, 224)
                    ),
                    "observation/joint_position": joint_position,
                    "observation/gripper_position": gripper_position,
                    "prompt": observation.language,
                }
            case "LIBERO":
                state = np.array(observation.state.joint_angles + [observation.state.gripper], dtype=np.float32)
                return {
                    "observation/state": state,
                    "observation/image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(exterior.image, 224, 224)
                    ),
                    "observation/wrist_image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist.image, 224, 224)
                    ),
                    "prompt": observation.language,
                }
            case "BASE":
                state = np.array(observation.state.joint_angles + [observation.state.gripper], dtype=np.float32)
                return {
                    "observation/state": state,
                    "observation/image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(exterior.image, 224, 224)
                    ),
                    "observation/wrist_image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist.image, 224, 224)
                    ),
                    "prompt": observation.language,
                }
    
    @override
    def test_health(self, timeout: float = 3.0) -> bool:
        conn = http.client.HTTPConnection(self.ip, self.port, timeout=timeout)
        try:
            conn.request("GET", "/healthz")
            return conn.getresponse().status == 200
        except OSError:
            return False
        finally:
            conn.close()

    def close(self):
        if self._ws is not None:
            self._ws.close()
            self._ws = None

class Pi05Client(Client):
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

        next_tick = time.monotonic()

        while not self._stop_inference.is_set():
            if not self.predicting:
                # Paused: don't busy-spin, but keep checking so we notice
                # being resumed or stopped.
                time.sleep(0.05)
                next_tick = time.monotonic()  # don't try to "catch up" on resume
                continue

            if action_queue.empty():
                try:
                    print(f"Action queue empty! Making prediction...")
                    chunk = self.server.make_prediction(observation=self.get_observation())
                except Exception as e:
                    print(f"Prediction request failed, retrying: {e}")
                    time.sleep(0.5)  # back off before hammering a possibly-down server
                    continue
                if chunk is None:
                    print(f"Prediction returned empty chunk: {chunk}")
                    time.sleep(0.5)
                    continue
                for action in chunk.actions:
                    action_queue.put(action)

            action = action_queue.get()
            try:
                self.connection.apply_action(action)
            except Exception as e:
                # Don't let one bad action kill the loop/thread silently. The
                # Twist command's built-in duration timeout means a skipped
                # action just lets the arm coast to a stop, not run away.
                print(f"apply_action failed, skipping this action: {e}")

            # Pace to the control period on a fixed schedule (not "sleep after
            # each step") so per-step overhead — the server call, apply_action
            # itself — doesn't accumulate drift over a long run.
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()  # fell behind; reset rather than spiral

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
        
