from abc import abstractmethod

class Connection:
    def __init__(self):
        pass

    @abstractmethod
    def awake(self):
        pass

    @abstractmethod
    def start(self):
        pass

# -------- Inference -------- #
    @abstractmethod
    def state(self):
        pass

    @abstractmethod
    def apply_action(self):
        pass

    @abstractmethod
    def pause(self):
        pass