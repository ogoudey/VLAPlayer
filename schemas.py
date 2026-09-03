from pydantic import BaseModel, ConfigDict, field_serializer
from typing import List, Literal, Optional
import cv2
import numpy as np
import base64

class Schema(BaseModel):
    pass


class Action(Schema):
    pass

class ActionChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")
    actions: List[Action]

class CartesianDelta(Action):
    kind: Literal["cartesian_delta"] = "cartesian_delta"
    dx: float = 0.0         # meters
    dy: float = 0.0         # meters
    dz: float = 0.0         # meters
    d_theta_x: float = 0.0  # degrees, in tool's local space
    d_theta_y: float = 0.0  # degrees
    d_theta_z: float = 0.0  # degrees
    gripper_command: Optional[float] = None

class JointVelocities7DOF(Action):
    kind: Literal["joint_velocities_7dof"] = "joint_velocities_7dof"
    j0: float = 0.0 # radians/sec
    j1: float = 0.0
    j2: float = 0.0
    j3: float = 0.0
    j4: float = 0.0
    j5: float = 0.0
    j6: float = 0.0
    gripper_command: Optional[float] = None

class Vision(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    camera_id: str
    images: List[np.ndarray]     # List of BGR, HxWx3 uint8 — one per historical_indices entry
    timestamp: float
    historical_indices: List[int]

    @field_serializer("images", when_used="json")
    def _encode_images(self, images: List[np.ndarray]) -> List[str]:
        encoded = []
        for i, image in enumerate(images):
            ok, buffer = cv2.imencode(".jpg", image)
            if not ok:
                raise ValueError(
                    f"Failed to JPEG-encode frame {i} (offset {self.historical_indices[i]}) "
                    f"from camera {self.camera_id!r}"
                )
            encoded.append(base64.b64encode(buffer).decode("ascii"))
        return encoded

class VisionBundle(BaseModel):
    views: dict[str, Vision]

class Pose(BaseModel):
    x: float          # meters
    y: float          # meters
    z: float           # meters
    theta_x: float     # degrees, Tait-Bryan
    theta_y: float     # degrees
    theta_z: float     # degrees

class State(BaseModel):
    joint_angles: Optional[List[float]]  # degrees, one per actuator, base->wrist order
    target_pose: Optional[Pose]
    gripper: Optional[float]

class Observation(Schema):
    vision: VisionBundle
    state: State
    language: str
    