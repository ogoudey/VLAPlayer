#from clients.pi import Pi05Client, Pi05ServerConfiguration
#client = Pi05Client(remote=Pi05ServerConfiguration(ip="192.168.0.177", port=8000, setting="LIBERO"))

from clients.groot import GrootN17Client, GrootN17ServerConfiguration
client = GrootN17Client(remote=GrootN17ServerConfiguration(ip="192.168.0.156", port=5555, embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT", timeout_ms=15000))


from connections.kinova import KinovaConnection
connection = KinovaConnection(auto_detect=True)


from camera_set import CameraConnection, CameraSet
camera_set = CameraSet([
    CameraConnection(auto_detect=True, historical_indices=[-15, 0]),
    CameraConnection(onboard_connection=connection, historical_indices=[-15, 0])
])

#from utilities import verify
#verify(client=client, connection=connection, camera_set=camera_set)
#connection.list_actions()





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
   

