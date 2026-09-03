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
from schemas import State, Action, CartesianDelta, JointVelocities7DOF, Pose
class KortexConnection:
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

    def __init__(self, control_period_s=0.05, max_linear_velocity=0.15,
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
        self._move_to_home()
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
        gripper_position = gripper_motors[0].position
        return gripper_position

    def state(self) -> State:
        with self._feedback_lock:
            feedback = self._latest_feedback

        result = State(
            joint_angles=[actuator.position for actuator in feedback.actuators],
            target_pose=self._get_target_pose(),
            gripper=self._get_gripper_position(),
        )

        if self.ui is not None:
            self.ui.report_robot_state(result)

        return result
    def _get_target_pose(self):
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
        if isinstance(action, CartesianDelta):
            self.handle_cartesian_delta(action)
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
        twist_cmd = Base_pb2.TwistCommand()
        twist_cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_MIXED
        twist_cmd.twist.linear_x = self._clamp(action.dx / dt, self.max_linear_velocity)
        twist_cmd.twist.linear_y = self._clamp(action.dy / dt, self.max_linear_velocity)
        twist_cmd.twist.linear_z = self._clamp(action.dz / dt, self.max_linear_velocity)
        twist_cmd.twist.angular_x = self._clamp(action.d_theta_x / dt, self.max_angular_velocity)
        twist_cmd.twist.angular_y = self._clamp(action.d_theta_y / dt, self.max_angular_velocity)
        twist_cmd.twist.angular_z = self._clamp(action.d_theta_z / dt, self.max_angular_velocity)
        twist_cmd.duration = int(dt * 1.0)

        self.base.SendTwistCommand(twist_cmd)

        if action.gripper_command is not None:
            # PLACEHOLDER mapping, not confirmed: assumes the raw ~[-1, 1]
            # model output is a tanh-style signal and linearly rescales it to
            # Kinova's [0, 1] (open->closed) convention. Polarity (does the
            # model's +1 mean open or closed?) is still unverified — test this
            # with the gripper clear of anything before trusting it in a real
            # grasp sequence.
            gripper_value = (action.gripper_command + 1.0) / 2.0
            self.handle_gripper_command(gripper_value)

    def _clamp(self, value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, value))
    
    def handle_gripper_command(self, value_0_1: float) -> None:
        """value_0_1: Kinova's own convention — 0.0 = fully open, 1.0 = fully closed."""
        cmd = Base_pb2.GripperCommand()
        cmd.mode = Base_pb2.GRIPPER_POSITION  # position mode, as opposed to speed/force
        finger = cmd.gripper.finger.add()
        finger.finger_identifier = 0
        finger.value = self._clamp01(value_0_1)
        self.base.SendGripperCommand(cmd)

    def list_actions(self) -> None:
        for action_type in (Base_pb2.REACH_JOINT_ANGLES, Base_pb2.REACH_POSE):
            request = Base_pb2.RequestedActionType()
            request.action_type = action_type
            for action in self.base.ReadAllActions(request).action_list:
                print(f"[{Base_pb2.ActionType.Name(action_type)}] {action.name!r}")

    def _move_to_home(self):
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

