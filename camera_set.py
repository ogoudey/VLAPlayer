from typing import List, Optional
from connections import Connection
from schemas import Vision
import threading
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from schemas import Vision, VisionBundle

class CameraConnection:
    POLL_HZ = 30.0

    def __init__(self, auto_detect: bool = False, onboard_connection: Optional["Connection"] = None):
        if auto_detect and onboard_connection is not None:
            raise ValueError("Pass exactly one of auto_detect or onboard_connection, not both.")

        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_capture = threading.Event()

        if auto_detect:
            self.backend = "realsense"
            self.id = self._detect_realsense_serial()
            self._pipeline: Optional[rs.pipeline] = None

        elif onboard_connection is not None:
            self.backend = "rtsp"
            # No duck-typed "onboard_camera" attribute — go straight to the
            # real, already-known field on KinovaConnection.
            ip_address = onboard_connection.tcp_connection.ip_address
            self.rtsp_url = f"rtsp://{ip_address}/color"
            self.id = "onboard"
            self._capture: Optional[cv2.VideoCapture] = None

        else:
            raise ValueError("Must pass either auto_detect=True or an onboard_connection.")

    def _detect_realsense_serial(self) -> str:
        devices = rs.context().query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense device detected over USB.")
        # First device found. If you ever have more than one D435, pass its
        # serial explicitly instead of relying on auto-detect picking one.
        return devices[0].get_info(rs.camera_info.serial_number)

    def awake(self):
        if self.backend == "realsense":
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.id)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self._pipeline.start(config)

        else:  # rtsp
            # This exact GStreamer pipeline is the one a Kinova engineer
            # posted as working: github.com/Kinovarobotics/kortex/issues/88
            gst_pipeline = (
                f"rtspsrc location={self.rtsp_url} latency=0 ! "
                "decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false"
            )
            
            self._capture = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if not self._capture.isOpened():
                # Falls back to OpenCV's default FFmpeg-based RTSP handling,
                # in case this OpenCV build wasn't compiled with GStreamer.
                self._capture = cv2.VideoCapture(self.rtsp_url)
            if not self._capture.isOpened():
                raise RuntimeError(f"Could not open onboard camera stream at {self.rtsp_url}.")

        self._stop_capture.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def _capture_loop(self):
        period = 1.0 / self.POLL_HZ
        while not self._stop_capture.is_set():
            frame = self._read_frame()
            if frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
            time.sleep(period)
    """
    def _read_frame(self) -> Optional[np.ndarray]:
        if self.backend == "realsense":
            frames = self._pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            return np.asanyarray(color_frame.get_data()) if color_frame else None
        else:
            ok, frame = self._capture.read()
            return frame if ok else None
    """
    def _read_frame(self) -> Optional[np.ndarray]:
        if self.backend == "realsense":
            frames = self._pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            frame_bgr = np.asanyarray(color_frame.get_data())
        else:
            ok, frame_bgr = self._capture.read()
            if not ok:
                return None

        # Both backends hand back BGR (rs.format.bgr8 for the D435, OpenCV's
        # default for the RTSP path) — convert once here so GUI display and
        # the model request transforms can uniformly assume RGB.
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        General public method to get the frame. Consumed by recorders and GUI, e.g.
        """
        with self._frame_lock:
            return self._latest_frame

    def get_vision(self) -> Optional[Vision]:
        frame = self.get_frame()
        if frame is None:
            return None
        return Vision(camera_id=self.id, image=frame, timestamp=time.time())

    def is_connected(self) -> bool:
        return self._latest_frame is not None

    def stop(self):
        self._stop_capture.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2)
        if self.backend == "realsense" and self._pipeline is not None:
            self._pipeline.stop()
        elif self.backend == "rtsp" and self._capture is not None:
            self._capture.release()

class CameraSet:
    def __init__(self, camera_connections: List[CameraConnection]):
        self.camera_connections = camera_connections
        self.recording = False

    def awake(self):
        for camera_connection in self.camera_connections:
            print(f"[CameraSet] Awakening camera {camera_connection.id}...")
            camera_connection.awake()
            print(f"[CameraSet] Camera {camera_connection.id} awake.")

    def get_vision(self) -> VisionBundle:
        views = {}
        for camera in self.camera_connections:
            vision = camera.get_vision()
            if vision is not None:
                views[camera.id] = vision
        return VisionBundle(views=views)

    

    def start_recording(self):
        self.recording = True
    def stop_recording(self):
        self.recording = False
