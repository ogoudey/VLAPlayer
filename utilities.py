from camera_set import CameraSet
from connections.connection import Connection
from clients.client import Client
from ui import UI

def bind_camera_set_to_client(camera_set: CameraSet, client: Client):
    client.camera_set = camera_set

def bind_connection_to_client(connection: Connection, client: Client):
    client.connection = connection

def bind_client_to_ui(client: Client, ui: UI):
    ui.client = client

def bind_connection_to_ui(connection: Connection, ui: UI):
    ui.connection = connection

def bind_camera_set_to_ui(camera_set: CameraSet, ui: UI):
    ui.camera_set = camera_set

def bind_ui_to_connection(connection: Connection, ui: UI):
    connection.ui = ui

def bind_ui_to_client(client: Client, ui: UI):
    client.ui = ui