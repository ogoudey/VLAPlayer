# VLA Player

## Running
```
python3 play.py --joint-branch /path/to/joint_branch.json
```

GR00T actions use `apply_abs_action_tidybot` by default. Run
`python3 play.py --execution waypoint` to use `apply_action_waypoint`,
waiting for each waypoint to complete, or `--execution delta` to use `apply_action`.
These options also apply to recorded episodes played through `play.py`.

To use the same compliant controller as Tidybot teleop, start its arm server
in one terminal, then run the player in another:

```bash
cd /home/hri/tidybot2
python arm_server.py
```

```bash
python play.py --config EEF_ROT6D_ABS_IN_OUT \
  --joint-branch /path/to/joint_branch.json --dt 0.02
```

`clients/tidybot_interface.py` exposes `TidybotInterface.apply_abs_action_tidybot(PoseTarget)`.
It sends exact absolute XYZ in metres, XYZW quaternions, and gripper targets in
`[0, 1]` to `Arm.execute_action`; Tidybot handles IK, trajectory generation, and
1 kHz current control. Finite gripper predictions outside `[0, 1]` are clamped
to that range and logged; NaN and infinite predictions are rejected.
Gain does not scale these absolute targets. An omitted
gripper command keeps the last gripper target. State comes from the same server,
with joint radians converted to the player's degree-based state schema.
Awake resets/retracts the arm and starts its controller. The wrist camera reads
RTSP directly without opening another Kortex control connection.

`--execution` accepts `tidybot`, `waypoint`, or `delta`, and defaults to `tidybot`.
Tidybot accepts absolute
`PoseTarget` outputs (`EEF_ROT6D_ABS_IN_OUT` or `THREE_TASKS_REL_ROT6D_LORA`);
delta outputs must first be converted to absolute targets. Recorded replay needs
`--replay-targets-csv`. Use `--tidybot-host` and `--tidybot-port` for a remote arm
server, and `--tidybot-robot-ip` for a different wrist-camera address.
Use poses calibrated to Tidybot's tool frame (currently a 0.12 m gripper offset).
Live Tidybot execution samples measured state and submits a target every `--dt`
without waiting for the arm to reach it. A separate inference worker requests
the next chunk before the target queue empties. There is no automatic pause or
100 ms delay between live chunks. When inference falls behind, the latest target
is resent on every tick, like enabled teleop with an unchanged target. The arm
continues tracking that target instead of triggering the command watchdog.

Predictions are aligned to their observation time: elapsed steps are skipped,
and the remaining fresh targets replace the old plan, up to `--execution-horizon`.
If inference takes longer than the whole predicted trajectory, the last target
continues to be sent while a fresh prediction is requested. Matching the target
rate and controller does not guarantee identical physical lag or hard real-time
deadlines; model latency, target trajectories, and RPC overruns still matter.

Manual Pause stops all submissions and discards queued targets and predictions
from before the pause. Resume seeds its initial target from measured state.
Changing dt also invalidates predictions made at the old rate. The server's
watchdog starts holding/decelerating after 50 ms without commands at its current
default policy period. Periods longer than that timeout interrupt trajectories.
Recorded replay retains chronological playback and its existing chunk pauses;
it does not skip frames to compensate for prediction latency. RPC or target
validation errors pause live Tidybot inference and discard the failed chunk.
Pause/Resume requests a fresh prediction; recorded replay stops on errors.

The `dt (seconds)` field below Gain starts at the launch-time `--dt` value.
Changing it updates the control period for the next action without restarting.
An in-progress waypoint or sleep finishes first. Recorded replay still checks
the dataset period unless launched with an explicit `--dt` override.
Use `--execution-horizon 16` to execute up to 16 actions per prediction (default:
8). The Execution horizon field below dt changes this limit for the next chunk;
already queued actions finish with the previous limit. If fewer actions are
returned, all available actions are executed. This setting also controls the
chunk size for recorded replay through `play.py`.
## Configuring
Re-write `play.py`

Live inference requires the training reference matching the selected config:

```bash
python play.py --config EEF_ROT6D_ABS_IN_OUT --joint-branch /path/to/joint_branch.json --dt 0.02
python play.py --execution waypoint --config THREE_TASKS_2_FIXED_LORA --q-ref /path/to/q_ref.json
```

`joint_branch.json` contains seven joint window lower bounds in **radians**,
either a JSON array or `{"joint_window_lo": [...]}` (`JOINT_WINDOW_LO` and
`window_lo` are also accepted). The training export uses
`{"period": 6.283185307179586, "window_lo": [...] , "usage": "..."}`.
Joints are wrapped with `((raw - window_lo) % period) + window_lo`; `usage` is
descriptive metadata. If `period` is omitted, it defaults to 2π.
`q_ref.json` contains a nonzero quaternion in **XYZW** order, either a JSON array
or `{"q_ref": [...]}`. State quaternions are sign-aligned to that reference.
Use the values exported by training. Files are validated before hardware setup.
Configs with `ROT6D` in their name, plus `THREE_TASKS_ABS_DELTAS_LORA`, require
the joint branch. Quaternion configs (`ABL7_FULLREL_QUAT`, `ABL6_EEFSRC_FULLSTATE`,
`THREE_TASKS_2_ABS_DELTAS_LORA`, and `THREE_TASKS_2_FIXED_LORA`) require `q_ref`.
Rot6D configs do not require `q_ref`. Replay does not require either file.

The former inline joint bounds live in config-specific files:

- `THREE_TASKS_ABS_DELTAS_LORA`: `dataset_meta/three_tasks_abs_deltas_lora/joint_branch.json`
- `THREE_TASKS_2_FIXED_LORA`: `dataset_meta/three_tasks_2_fixed_lora/joint_branch.json`

These configs load their historical bounds on construction. Live rot6D launches
still require an explicit `--joint-branch`, which overrides the default. The
quaternion config still requires only `--q-ref`; optionally supply
`--joint-branch` to override its historical joint bounds. Wrapping uses the same
float32 arithmetic as the former inline implementation, including at boundaries.
The historical quaternion reference for `THREE_TASKS_2_FIXED_LORA` is stored in
`dataset_meta/three_tasks_2_fixed_lora/q_ref.json` and loaded on construction.
Live launches still require `--q-ref`, which overrides that historical default.

Aiming for Kinova...

## Replay a recorded episode

`data/replay/eggplant_episode_002/` contains `episode_index=2` (zero-based)
from `olingoudey/tabletop_3tasks_deltas_2`, revision
`5cac0290168a8466300b74777cfc5ec9bced36ee`: **take the eggplant off the plate**.
There are 421 frames at 50 Hz (8.42 seconds).

- `actions.csv`: original position deltas in metres, world-frame rotation vectors
  in radians, and absolute gripper targets in [0, 1].
- `states.csv`: recorded joint angles in radians, tool XYZ, XYZW quaternion,
  and gripper position.
- `episode.json`: both streams, frame indices, timestamps, and source provenance.

Inspect without connecting to the robot:

```bash
python replay_episode.py
```

For real replay, use the `kinova-groot` environment. Manually prepare the printed
starting joint configuration and gripper position first. The script does not home
the arm; it rejects a starting joint error above 5 degrees. Set `MIN_TOOL_Z` to
your measured clearance limit in metres, in **Kortex's configured tool frame**:

```bash
python replay_episode.py --execute --min-z "$MIN_TOOL_Z"
```

Replay sends every recorded delta once through `KinovaConnection.apply_action`,
using the same radians-to-degrees conversion as the active GR00T configuration,
gain 1, and the existing velocity limits. It runs continuously at a nominal 50 Hz;
it does not reproduce policy inference or chunk-boundary pauses. Slow RPCs can
lengthen playback. A timestamped `replay_*.json` alongside the episode records
command timing and measured state before each action. Completion, Ctrl+C, and
errors call `base.Stop()`.

The minimum-Z check stops if measured tool Z, or its next nominal linear step,
reaches the limit. It is a software check, not a collision model: allow clearance
for the actual gripper geometry, feedback delay, and stopping distance. Recorded
tool Z may differ from Kortex tool Z because of the tool-offset issue.

For this episode, ideal integration of the Z commands at gain 1 with the existing
0.15 m/s clamp reaches **-0.2965 m** from the recorded starting Z of **0.2087 m**.
The recorded trajectory only reaches **0.0448 m**. This assumes exact velocity
tracking and matching tool frames; it is not a prediction of the real trajectory.
The exported deltas remain unchanged so replay can test this controller mismatch.

## Absolute EEF rot6D inference

Set `config=GrootConfig.EEF_ROT6D_ABS_IN_OUT` in the remote configuration in
`play.py` for a checkpoint with absolute XYZ + rot6D state and action poses.
The state key is `eef_pose` with shape `(1, 1, 9)`; the server response keys
are `eef_pose_targets` with shape `(1, T, 9)` and `gripper_targets` with shape
`(1, T, 1)`. Both rotations use the first two **rows** of the rotation matrix,
flattened, matching the training conversion. State joints are sent in radians.
Returned poses are decoded into absolute quaternion `PoseTarget` actions;
gripper targets are passed through as absolute commands. The inference loop
executes these poses through `apply_abs_action_tidybot` by default. Use
`--execution waypoint` to wait for each Kinova waypoint to complete, or `--execution delta` to use
`apply_action`.

Select the checkpoint separately using the server's `--model-path`, and use
the embodiment tag saved in its processor. This client config selects the
wire layout; it does not select weights or send an embodiment tag.

## Replay through the GR00T inference loop

To debug the actual queue and timing in `clients/groot.py`, run in `kinova-groot`:

```bash
python play.py --execution waypoint --replay-episode --min-z "$MIN_TOOL_Z"
```

This defaults to eggplant `episode_index=2`. An explicit JSON path may follow
`--replay-episode`. Use the normal GUI Start/Pause controls. This follows the normal
player startup, including moving to Retract and opening the gripper; it also uses
the normal cameras and telemetry. The recorded starting joints are checked before
the first dataset action. The minimum-Z check uses the current GUI gain and
connection velocity limit.

The action period defaults to the episode FPS. Override it with `--dt` (or
`--control-period`) in seconds, for example `--dt 0.01`. An explicit override
allows replaying at a period different from the dataset FPS and changes the
per-action displacement cap used by waypoint execution. Live inference defaults
to `0.02` seconds.

`RecordedGrootConfiguration` replaces the remote prediction source. It bypasses
server health checks and sends each group of eight recorded rows through the same
GR00T response decoder, then through `GrootN17Client.start_inference_loop()`.
The loop retains its existing 100 ms pause at each prediction boundary and its
existing action scheduling. The final chunk contains five actions. EOF stops the
loop and sends zero velocity; restart the program to replay again.

For a simulated 200 ms server response time on each chunk:

```bash
python play.py --execution waypoint --replay-episode --min-z "$MIN_TOOL_Z" --replay-delay 0.2
```

With the default delay of zero, there is no server/network latency. This is not
continuous 8.42-second playback: the chunk pauses and optional delay extend it.
Like the live loop, pausing clears queued actions, so a mid-chunk pause discards
the remaining rows of that chunk and resumes at the next chunk. Console messages
show the loaded frame ranges. Supply the usual positional recording path before
the flags to save the GUI telemetry, e.g. `python play.py replay.rrd
--execution waypoint --replay-episode --min-z "$MIN_TOOL_Z"`. Without `--replay-episode`, normal server
inference is selected.

## Replay absolute CSV positions through the waypoint executor

```bash
python play.py absolute_replay.rrd --execution waypoint --replay-targets-csv --min-z "$MIN_TOOL_Z" --dt 0.02
```

This uses all 421 `target_x_m`, `target_y_m`, and `target_z_m` rows from
`data/replay/eggplant_episode_002/intended_absolute_positions.csv`. An explicit
CSV path can follow `--replay-targets-csv`; use `--replay-episode PATH` to select
its corresponding episode for start-joint checks and recorded gripper commands.
Targets are sent as exact absolute positions in the Kortex base frame, with
orientation held at the first replay observation. GUI gain and the delta
displacement clamp do not alter these positions. The minimum-Z guard checks both
the measured Z and the absolute target Z. The default CSV's lowest target Z is
0.022361 m; a higher minimum-Z setting stops before those targets.

Each target uses `KinovaConnection.apply_action_waypoint`, validates speed
limits, and waits for completion before the next one. `--dt` controls the nominal
loop period; waypoint completion may make replay slower than the dataset. Target,
achieved pose, error, and duration remain logged under `inference/waypoint`.
