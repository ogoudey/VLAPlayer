import math
import os
import subprocess
import threading
import time
import numpy as np

import collections
import collections.abc
collections.MutableMapping = collections.abc.MutableMapping
collections.MutableMapping = collections.abc.MutableMapping
collections.MutableSequence = collections.abc.MutableSequence
collections.Mapping = collections.abc.Mapping

from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import ActuatorCyclic_pb2, ActuatorConfig_pb2, Base_pb2, BaseCyclic_pb2, Common_pb2, ControlConfig_pb2, Session_pb2
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport

from .connection import Connection
from schemas import JointAngles, JointDelta, PoseTarget, State, Action, CartesianDelta, JointVelocities7DOF, Pose
from scipy.spatial.transform import Rotation



class KortexConnection:
    """
    Kinova is X forward, Y left, Z up. Right-hand-rule.
    """
    IP_ADDRESS = '192.168.1.10'
    TCP_PORT = 10000
    UDP_PORT = 10001

    @staticmethod
    def createTcpConnection():
        return KortexConnection(port=KortexConnection.TCP_PORT)

    @staticmethod
    def createUdpConnection():
        return KortexConnection(port=KortexConnection.UDP_PORT)

    def __init__(self, ip_address=IP_ADDRESS, port=TCP_PORT, credentials=('admin', 'admin')):
        self.ip_address = ip_address
        self.port = port
        self.credentials = credentials
        self.session_manager = None
        self.transport = TCPTransport() if port == KortexConnection.TCP_PORT else UDPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)

    def __enter__(self):
        self.transport.connect(self.ip_address, self.port)
        if self.credentials[0] != '':
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000   # (milliseconds)
            session_info.connection_inactivity_timeout = 2000 # (milliseconds)
            self.session_manager = SessionManager(self.router)
            print('Logging as', self.credentials[0], 'on device', self.ip_address)
            self.session_manager.CreateSession(session_info)
        return self.router

    def __exit__(self, *_):
        if self.session_manager is not None:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000
            self.session_manager.CloseSession(router_options)
        self.transport.disconnect()

"""
[REACH_JOINT_ANGLES] 'Retract'
[REACH_JOINT_ANGLES] 'Home'
[REACH_JOINT_ANGLES] 'Packaging'
[REACH_JOINT_ANGLES] 'Zero'
[REACH_POSE] 'pick'
"""
class KinovaConnection(Connection):
    HOME_ACTION_NAME = 'Retract'    # factory default — check the Kinova web app's Actions list if this errors
    FEEDBACK_POLL_HZ = 100.0
    HOME_ACTION_TIMEOUT_S = 20.0

    def __init__(self, control_period_s=0.02, max_linear_velocity=0.15,
                 max_angular_velocity=30.0, max_joint_velocity_deg_s=30.0, **kwargs):
        super().__init__()
        # Check whether arm is connected
        try:
            subprocess.run(['ping', '-c', '1',  '192.168.1.10'], check=True, timeout=1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as e:
            raise Exception('Could not communicate with arm') from e
 
        # General Kortex setup
        self.tcp_connection = KortexConnection.createTcpConnection()
        self.udp_connection = KortexConnection.createUdpConnection()
        self.base = BaseClient(self.tcp_connection.__enter__())
        self.base_cyclic = BaseCyclicClient(self.udp_connection.__enter__())
        self.actuator_config = ActuatorConfigClient(self.base.router)
        self.actuator_count = self.base.GetActuatorCount().count
        self.control_config = ControlConfigClient(self.base.router)
        device_manager = DeviceManagerClient(self.base.router)
        device_handles = device_manager.ReadAllDevices()
        self.actuator_device_ids = [
            handle.device_identifier for handle in device_handles.device_handle
            if handle.device_type in [Common_pb2.BIG_ACTUATOR, Common_pb2.SMALL_ACTUATOR]
        ]
        self.send_options = RouterClientSendOptions()
        self.send_options.timeout_ms = 3
 
        # Command and feedback setup
        self.base_command = BaseCyclic_pb2.Command()
        for _ in range(self.actuator_count):
            self.base_command.actuators.add()
        self.motor_cmd = self.base_command.interconnect.gripper_command.motor_cmd.add()
        self.base_feedback = BaseCyclic_pb2.Feedback()
 
        # Make sure actuators are in position mode
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('POSITION')
        for device_id in self.actuator_device_ids:
            self.actuator_config.SetControlMode(control_mode_message, device_id)
 
        # Make sure arm is in high-level servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)
 
        # Cyclic thread setup
        self.cyclic_thread = None
        self.kill_the_thread = False
        self.cyclic_running = False

        self._feedback_lock = threading.Lock()
        self._latest_feedback = self.base_feedback
        self.control_period_s = control_period_s        # expected seconds between apply_action() calls
        self.max_linear_velocity = max_linear_velocity   # m/s safety clamp
        self.max_angular_velocity = max_angular_velocity # deg/s safety clamp
        self.max_joint_velocity_deg_s = max_joint_velocity_deg_s  # deg/s safety clamp — Kinova's own default is 30 deg/s, but the arm can do more if you want to risk it

    def awake(self):
        self.move_to_home()
        self.kill_the_thread = False
        self.cyclic_thread = threading.Thread(target=self._feedback_poll_loop, daemon=True)
        self.cyclic_thread.start()
        self.cyclic_running = True
        

    def _feedback_poll_loop(self):
        period = 1.0 / self.FEEDBACK_POLL_HZ
        while not self.kill_the_thread:
            try:
                feedback = self.base_cyclic.RefreshFeedback()
            except Exception as e:
                print("Feedback refresh failed:", e)
                time.sleep(period)
                continue
            with self._feedback_lock:
                self._latest_feedback = feedback
            time.sleep(period)
        self.cyclic_running = False

    def _get_gripper_position(self):
        with self._feedback_lock:
            feedback = self._latest_feedback

        gripper_motors = feedback.interconnect.gripper_feedback.motor
        if len(gripper_motors) == 0:
            # Empty until the first real RefreshFeedback() comes back (or if
            # no gripper is attached/configured). Don't silently report 0.0
            # here — that's indistinguishable from "gripper fully open" and
            # would feed a fabricated value into whatever policy reads it.
            print(
                "No gripper feedback available yet — has awake() run and "
                "completed at least one feedback refresh?"
            )
            return 0.0

        # Kinova reports gripper position as a percentage: 0 = fully open,
        # 100 = fully closed. Keeping that native convention here — DROID's
        # and the base checkpoint's own conventions (which may define
        # open/closed the opposite way, or expect [0, 1] instead of
        # [0, 100]) belong in the per-checkpoint transform, not baked in here.
        gripper_position = gripper_motors[0].position  / 100.0
        return gripper_position

    def state(self) -> State:
        """
        Check against tidybot2's get_state() and get_tool_pose()
        """
        with self._feedback_lock:
            feedback = self._latest_feedback

        result = State(
            joint_angles=[actuator.position for actuator in feedback.actuators],
            target_pose=self._get_current_pose(),
            gripper=self._get_gripper_position(),
        )

        if self.ui is not None:
            self.ui.report_state(result)

        return result
    
    def _get_current_pose(self):
        with self._feedback_lock:
            feedback = self._latest_feedback
        target_pose = Pose(
            x=feedback.base.tool_pose_x,
            y=feedback.base.tool_pose_y,
            z=feedback.base.tool_pose_z,
            theta_x=feedback.base.tool_pose_theta_x,
            theta_y=feedback.base.tool_pose_theta_y,
            theta_z=feedback.base.tool_pose_theta_z,
        )
        return target_pose
    
    def apply_action(self, action: Action):
        if isinstance(action, PoseTarget):
            current_pose = self._get_current_pose()

            current_rotation = Rotation.from_euler(
                "xyz", [current_pose.theta_x, current_pose.theta_y, current_pose.theta_z], degrees=True
            )
            target_rotation = Rotation.from_euler(
                "xyz", [action.theta_x, action.theta_y, action.theta_z], degrees=True
            )
            delta_rotation = target_rotation * current_rotation.inv()
            d_theta_x, d_theta_y, d_theta_z = delta_rotation.as_euler("xyz", degrees=True)

            delta = CartesianDelta(
                dx=action.x - current_pose.x,
                dy=action.y - current_pose.y,
                dz=action.z - current_pose.z,
                d_theta_x=float(d_theta_x),
                d_theta_y=float(d_theta_y),
                d_theta_z=float(d_theta_z),
                gripper_command=action.gripper_command,
            )
            self.handle_cartesian_delta(delta)      
        elif isinstance(action, CartesianDelta):
            self.handle_cartesian_delta(action)
        elif isinstance(action, JointAngles):
            self.handle_joint_angles(action)
        elif isinstance(action, JointDelta):
            self.handle_joint_deltas(action)
        elif isinstance(action, JointVelocities7DOF):
            self.handle_joint_velocities_7dof(action)
        else:
            # No duck-typing fallback on purpose: an action type we don't
            # have an explicit handler for should fail loudly, not silently
            # no-op or guess at what it means.
            raise TypeError(
                f"KinovaConnection has no handler for action type {type(action).__name__!r}"
            )
    
    def pause(self) -> None:
        """Halt motion without touching the connection — safe to call
        repeatedly, and safe to follow with more apply_action() calls."""
        try:
            self.handle_cartesian_delta(CartesianDelta())
        except Exception as e:
            print(f"Failed to zero out twist during pause(): {e}")

    def handle_joint_angles(self, action: JointAngles) -> None:
        angles_rad = [action.j0, action.j1, action.j2, action.j3,
                      action.j4, action.j5, action.j6]  # length validated once at init

        kortex_action = Base_pb2.Action()
        kortex_action.name = "reach_joint_angles"
        kortex_action.application_data = ""

        for joint_id, angle_rad in enumerate(angles_rad):
            ja = kortex_action.reach_joint_angles.joint_angles.joint_angles.add()
            ja.joint_identifier = joint_id
            ja.value = math.degrees(angle_rad) % 360.0  # Kortex uses [0, 360)

        self.base.ExecuteAction(kortex_action)

        if action.gripper_command is not None:
            self.handle_gripper_command(action.gripper_command)
    def handle_joint_deltas(self, action: JointDelta) -> None:
        deltas_rad = [action.j0, action.j1, action.j2, action.j3,
                      action.j4, action.j5, action.j6]  # length validated once at init

        gain = self.ui.gain if self.ui is not None else 1.0
        
        dt = self.control_period_s
        print(f"Applying joint deltas: {deltas_rad} rad over {dt} s")
        vel_deg_s = [math.degrees(d / dt) * gain for d in deltas_rad]


        joint_speeds = Base_pb2.JointSpeeds()
        for joint_id, v in enumerate(vel_deg_s):
            js = joint_speeds.joint_speeds.add()
            js.joint_identifier = joint_id
            js.value = v
        self.base.SendJointSpeedsCommand(joint_speeds)

        if action.gripper_command is not None:
            self.handle_gripper_command(action.gripper_command)

    def handle_joint_velocities_7dof(self, action: JointVelocities7DOF) -> None:
        velocities_rad_s = [action.j0, action.j1, action.j2, action.j3,
                            action.j4, action.j5, action.j6]

        if len(velocities_rad_s) != self.actuator_count:
            # Real check, not boilerplate — Gen3 ships in both 6-DOF and 7-DOF
            # variants, so this catches a policy/hardware mismatch immediately
            # instead of silently sending 7 commands to a 6-joint arm.
            raise ValueError(
                f"Expected {self.actuator_count} joint velocities, got {len(velocities_rad_s)}"
            )

        dt = self.control_period_s
        joint_speeds = Base_pb2.JointSpeeds()
        for joint_id, velocity_rad_s in enumerate(velocities_rad_s):
            speed = joint_speeds.joint_speeds.add()
            speed.joint_identifier = joint_id
            speed.value = self._clamp(math.degrees(velocity_rad_s), self.max_joint_velocity_deg_s)
            speed.duration = int(dt * 1.0)

        self.base.SendJointSpeedsCommand(joint_speeds)

        if action.gripper_command is not None:
            self.handle_gripper_command(action.gripper_command)

    def handle_cartesian_delta(self, action: CartesianDelta):
        dt = self.control_period_s
        gain = self.ui.gain if self.ui is not None else 1.0
        print(f"Gain: {gain}, dt: {dt}, action: {action}")
        twist_cmd = Base_pb2.TwistCommand()
        twist_cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        twist_cmd.twist.linear_x = self._clamp(gain * action.dx / dt, self.max_linear_velocity)
        twist_cmd.twist.linear_y = self._clamp(gain * action.dy / dt, self.max_linear_velocity)
        twist_cmd.twist.linear_z = self._clamp(gain * action.dz / dt, self.max_linear_velocity)
        twist_cmd.twist.angular_x = self._clamp(gain * action.d_theta_x / dt, self.max_angular_velocity)
        twist_cmd.twist.angular_y = self._clamp(gain * action.d_theta_y / dt, self.max_angular_velocity)
        twist_cmd.twist.angular_z = self._clamp(gain * action.d_theta_z / dt, self.max_angular_velocity)
        twist_cmd.duration = int(dt * 3.0)

        self.base.SendTwistCommand(twist_cmd)

        if action.gripper_command is not None:

            self.handle_gripper_command(action.gripper_command)

    def _clamp(self, value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def get_gripper_position(self) -> float:
        """Measured gripper position, 0.0 = open, 1.0 = closed."""
        req = Base_pb2.GripperRequest()
        req.mode = Base_pb2.GRIPPER_POSITION
        meas = self.base.GetMeasuredGripperMovement(req)
        return meas.finger[0].value if len(meas.finger) else 0.0

    def handle_gripper_delta(self, gripper_delta: float, scale: float = 1.0,
                            deadband: float = 1e-3) -> None:
        # Initialize the target from the real gripper state the first time
        if getattr(self, "_gripper_target", None) is None:
            self._gripper_target = self.get_gripper_position()

        if abs(gripper_delta) < deadband:
            return  # avoid spamming tiny commands

        self._gripper_target = self._clamp(self._gripper_target + scale * gripper_delta, 1.0)
        self.handle_gripper_command(self._gripper_target)

    def handle_gripper_command(self, value_0_1: float) -> None:
        """value_0_1: Kinova's own convention — 0.0 = fully open, 1.0 = fully closed."""
        cmd = Base_pb2.GripperCommand()
        cmd.mode = Base_pb2.GRIPPER_POSITION  # position mode, as opposed to speed/force
        finger = cmd.gripper.finger.add()
        finger.finger_identifier = 0
        finger.value = self._clamp(value_0_1, 1.0)  # Kinova's own convention — 0 = fully open, 100 = fully closed
        self.base.SendGripperCommand(cmd)

    def list_actions(self) -> None:
        for action_type in (Base_pb2.REACH_JOINT_ANGLES, Base_pb2.REACH_POSE):
            request = Base_pb2.RequestedActionType()
            request.action_type = action_type
            for action in self.base.ReadAllActions(request).action_list:
                print(f"[{Base_pb2.ActionType.Name(action_type)}] {action.name!r}")

    def move_to_home(self):
        self.base.ClearFaults()
        self._wait_until_ready(timeout=3)
        self._gripper_position_command(0.0)
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        home_action = next(
            (a for a in action_list.action_list if a.name == self.HOME_ACTION_NAME), None
        )
        if home_action is None:
            raise RuntimeError(f"No factory action named {self.HOME_ACTION_NAME!r} found on this arm.")

        finished = threading.Event()

        def _on_notification(notification, finished=finished):
            if notification.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
                finished.set()

        handle = self.base.OnNotificationActionTopic(_on_notification, Base_pb2.NotificationOptions())
        self.base.ExecuteAction(home_action)
        reached = finished.wait(self.HOME_ACTION_TIMEOUT_S)
        self.base.Unsubscribe(handle)
        if not reached:
            raise RuntimeError("Timed out waiting for arm to reach Home position.")

    def _gripper_position_command(self, value):
        # Send gripper command
        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_command.gripper.finger.add()
        finger.value = value
        self.base.SendGripperCommand(gripper_command)

        # Wait for reported position to match value
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        while True:
            gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
            if abs(value - gripper_measure.finger[0].value) < 0.01:
                break
            time.sleep(0.01)

    def _feedback_poll_loop(self):
        period = 1.0 / self.FEEDBACK_POLL_HZ
        while not self.kill_the_thread:
            try:
                feedback = self.base_cyclic.RefreshFeedback()
            except Exception as e:
                print("Feedback refresh failed:", e)
                time.sleep(period)
                continue
            with self._feedback_lock:
                self._latest_feedback = feedback
            time.sleep(period)
        self.cyclic_running = False

    def _wait_until_ready(self, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        last_state = None
        while time.time() < deadline:
            arm_state = self.base.GetArmState()
            last_state = arm_state.active_state
            if last_state == Common_pb2.ARMSTATE_SERVOING_READY:
                return
            time.sleep(0.05)

        state_name = Common_pb2.ArmState.Name(last_state) if last_state is not None else "UNKNOWN"
        