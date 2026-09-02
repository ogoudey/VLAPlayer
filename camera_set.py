from typing import List, Optional
from connections import Connection

class CameraConnection:
    def __init__(self, auto_detect: bool=False, onboard_connection: Optional[Connection]=None):
        if auto_detect:
            # run through normal ports, dev/sda1, whatever... its a usb connection 
            pass
        if onboard_connection:
            if hasattr(onboard_connection, "onboard_camera"):
                # onboard_connection.onboard_camera is probably a rstp stream
                pass

    def awake(self):
        # self.camera_stream.start_capture...
        pass



class CameraSet:
    def __init__(self, camera_connections: List[CameraConnection]):
        self.camera_connections = camera_connections
        self.recording = False
    def awake(self):
        for camera_connection in self.camera_connections:
            camera_connection.awake()

    def start_recording(self):
        self.recording = True
    def stop_recording(self):
        self.recording = False
