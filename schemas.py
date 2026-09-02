from pydantic import BaseModel
from typing import List, Literal

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
    pass

class State(BaseModel):
    joint_angles: List[float]  # degrees, one per actuator, base->wrist order

class Observation(Schema):
    vision: Vision
    state: State
    language: str
    