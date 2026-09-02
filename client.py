from exceptions import ConnectionIssue, CameraIssue
from abc import abstractmethod

class Client:
    """
    A client of a VLA.
    """
    def __init__(self):
        self.connection = None
        self.camera_set = None
        self.language = None

        self.predicting = True

    @abstractmethod
    def awake(self):
        # load the model
        pass

    def start(self):
        if self.connection is None:
            raise ConnectionIssue("Connection is None at Start")
        if self.camera_set is None:
            raise CameraIssue("Camera set is None at Start")

        # start inference loop
    
    def stop_predicting(self):
        self.predicting = False

    def start_predicting(self):
        self.predicting = True

    

