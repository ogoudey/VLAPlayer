from pi import Pi05Client
from client import ServerConfiguration
client = Pi05Client(remote=ServerConfiguration(ip="0.0.0.0", port=8000))

from kinova import KinovaConnection
connection = KinovaConnection(auto_detect=True)


from camera_set import CameraConnection, CameraSet
camera_set = CameraSet([
    CameraConnection(auto_detect=True),
    CameraConnection(onboard_connection=connection)
])

#from utilities import verify
#verify(client=client, connection=connection, camera_set=camera_set)

from gui import GUI
ui = GUI(headless=False)

from player import Player
player = Player(client, connection, camera_set, ui)

ui.bind_player(player)

print("Open http://127.0.0.1:8000/ and press Start.")
ui.wait_for_start()  # optional — only if this thread wants to block too

# Keep the process (and telemetry) alive for a bit so you can watch it.
import time
try:
    time.sleep(120)
except KeyboardInterrupt:
    pass
finally:
    ui.stop()
   

