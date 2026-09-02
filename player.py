from client import Client
from ui import UI
from typing import Optional
from connections import Connection
from camera_set import CameraSet

from utilities import bind_camera_set_to_client, bind_connection_to_client, bind_client_to_ui, bind_connection_to_ui, bind_camera_set_to_ui

class Player:
    def __init__(self, client: Client, connection: Connection, camera_set: CameraSet, ui: Optional[UI]):
        self.client = client
        self.connection = connection
        self.camera_set = camera_set
        self.ui = ui

    def awake(self):
        print(f"[Player] Awakening...")
        self.connection.awake()
        print(f"[Player] Connection awake.")
        self.client.awake()
        print(f"[Player] Client awake.")
        self.camera_set.awake()
        print(f"[Player] Cameras awake.")
        bind_camera_set_to_client(self.camera_set, self.client)
        bind_connection_to_client(self.connection, self.client)
        bind_client_to_ui(self.client, self.ui)
        bind_connection_to_ui(self.connection, self.ui)
        bind_camera_set_to_ui(self.camera_set, self.ui)
        print(f"[Player] Everything awaken is bound.")
        
    def start(self):
        print(f"[Player] Starting...")
        self.camera_set.start_recording()
        self.client.language = self.ui.language
        self.client.start()
        print(f"[Player] Started.")

    def pause(self):
        print(f"[Player] Pausing.")
        self.connection.pause()
        print(f"[Player] Connection paused.")
        self.client.stop_predicting()
        print(f"[Player] Client stopped making predictions.")
        self.camera_set.stop_recording()
        print(f"[Player] Cameras stopped recording.")
        print(f"[Player] Paused.")

    def unpause(self):
        print(f"[Player] Unpausing.")
        self.client.start_predicting()
        print(f"[Player] Client restarted prediction.")
        self.camera_set.start_recording()
        print(f"[Player] Cameras restarted recording.")
        print(f"[Player] Unpaused.")