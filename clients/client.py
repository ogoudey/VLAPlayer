from exceptions import ConnectionIssue, CameraIssue
from abc import ABC, abstractmethod
from dataclasses import dataclass
from schemas import Observation, ActionChunk
from typing import Optional

@dataclass
class ServerConfiguration(ABC):
    ip: str
    port: int
    setting: Optional[str] = None
    @abstractmethod
    def make_prediction(self, observation: Observation) -> ActionChunk:
        raise NotImplementedError()

    @abstractmethod
    def test_health(self, timeout: float = 3.0) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def ping(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def start_server(self):
        raise NotImplementedError()
    
class Client:
    """
    A client of a VLA.
    """
    server: ServerConfiguration
    def __init__(self):
        self.connection = None
        self.camera_set = None
        self.language = None

        self.predicting = True

    
    def awake(self):
        # load the model, if it were local
        self.server.ping()
        if not self.server.test_health():
            self.server.start_server()
            while not self.server.test_health():
                print(f"Waiting for server to start...")
                import time
                time.sleep(1)
        
        try:
            self.server.test_health()
        except Exception as e:
            print(f"Could not reach server: {e}")


    @abstractmethod
    def start_inference_loop(self):
        pass

    def start(self):
        """
        Called from player
        """
        if self.connection is None:
            raise ConnectionIssue("Connection is None at Start")
        if self.camera_set is None:
            raise CameraIssue("Camera set is None at Start")

        # start inference loop
        print(f"Starting inference loop...")
        self.start_inference_loop()
    
    def stop_predicting(self):
        self.predicting = False

    def start_predicting(self):
        self.predicting = True

    

