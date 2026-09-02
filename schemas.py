from pydantic import BaseModel, ConfigDict, field_serializer
from typing import List, Literal
import cv2
import numpy as np
import base64

class Schema(BaseModel):
    pass

class Action(Schema):
    pass

class CartesianDelta(Action):
    kind: Literal["cartesian_delta"] = "cartesian_delta"
    dx: float = 0.0         # meters
    dy: float = 0.0         # meters
    dz: float = 0.0         # meters
    d_theta_x: float = 0.0  # degrees, in tool's local space
    d_theta_y: float = 0.0  # degrees
    d_theta_z: float = 0.0  # degrees

class Vision(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # needed for the np.ndarray field

    camera_id: str
    image: np.ndarray   # BGR, HxWx3 uint8 — raw frame, used as-is for local/in-process calls
    timestamp: float

    @field_serializer("image", when_used="json")
    def _encode_image(self, image: np.ndarray) -> str:
        # Only triggers for .model_dump_json() / .model_dump(mode="json") —
        # a local caller doing .model_dump() (python mode) or just reading
        # `.image` directly still gets the raw array.
        ok, buffer = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError(f"Failed to JPEG-encode frame from camera {self.camera_id!r}")
        return base64.b64encode(buffer).decode("ascii")

class State(BaseModel):
    joint_angles: List[float]  # degrees, one per actuator, base->wrist order

class Observation(Schema):
    vision: Vision
    state: State
    language: str
    