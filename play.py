
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("recording_path", nargs="?", help="Rerun recording output")
parser.add_argument("--replay-episode", nargs="?", type=Path,
                    const=Path(__file__).parent / "data/replay/eggplant_episode_002/episode.json",
                    help="Feed recorded actions into the GR00T loop (default: eggplant episode 2)")
parser.add_argument("--min-z", type=float, help="Minimum Kortex tool Z in metres for replay")
parser.add_argument("--replay-delay", type=float, default=0.0,
                    help="Simulated server latency in seconds per recorded chunk")
args = parser.parse_args()
if args.replay_episode is not None and args.min_z is None:
    parser.error("--replay-episode requires --min-z in Kortex tool coordinates")

# ------- GROOT_CONFIGS ------- #
from clients.groot import GrootConfig

from clients.groot import GrootN17Client, GrootN17ServerConfiguration, RecordedGrootConfiguration
if args.replay_episode is not None:
    remote = RecordedGrootConfiguration(args.replay_episode, min_z=args.min_z,
                                         response_delay_s=args.replay_delay)
    print(f"GR00T loop replay: episode {remote.episode['episode_index']}, {remote.episode['task']}")
else:
    remote = GrootN17ServerConfiguration(ip="192.168.0.196", port=5555, config=GrootConfig.THREE_TASKS_2_FIXED_LORA, timeout_ms=15000)
client = GrootN17Client(remote=remote)

"""
IF the state pose does not follow the target pose, expose the twists.
"""

"""
Server:
uv run python gr00t/eval/run_gr00t_server.py --model-path /home/hrilab/Isaac-GR00T/Models/gr00t_tabletop_3tasks_rot6d_LoRA  --embodiment-tag NEW_EMBODIMENT --device cuda:0
"""



#from clients.pi import Pi05Client, Pi05ServerConfiguration
#client = Pi05Client(remote=Pi05ServerConfiguration(ip="192.168.0.177", port=8000, setting="LIBERO"))


from connections.kinova import KinovaConnection
connection = KinovaConnection(auto_detect=True)


from camera_set import CameraConnection, CameraSet
camera_set = CameraSet([
    CameraConnection(auto_detect=True, historical_indices=[0]),
    CameraConnection(onboard_connection=connection, historical_indices=[0])
])

#from utilities import verify
#verify(client=client, connection=connection, camera_set=camera_set)
#connection.list_actions()





from gui import GUI
ui = GUI(headless=False, client_setting=client.setting, recording_path=args.recording_path)

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
    client.stop_predicting()
    connection.pause()
    ui.stop()
   
