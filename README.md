# VLA Player

## Running
```
python3 play.py
```

GR00T actions use `apply_action_waypoint` by default, waiting for each waypoint
to complete. Run `python3 play.py --twist` to use `apply_action` instead.
This option also applies to recorded episodes played through `play.py`.
The `dt (seconds)` field below Gain starts at the launch-time `--dt` value.
Changing it updates the control period for the next action without restarting.
An in-progress waypoint or sleep finishes first. Recorded replay still checks
the dataset period unless launched with an explicit `--dt` override.
## Configuring
Re-write `play.py`

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
executes these poses through `apply_action_waypoint` by default, waiting for each
waypoint to complete before sending the next. Use `--twist` to use `apply_action`.

Select the checkpoint separately using the server's `--model-path`, and use
the embodiment tag saved in its processor. This client config selects the
wire layout; it does not select weights or send an embodiment tag.

## Replay through the GR00T inference loop

To debug the actual queue and timing in `clients/groot.py`, run in `kinova-groot`:

```bash
python play.py --replay-episode --min-z "$MIN_TOOL_Z"
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
python play.py --replay-episode --min-z "$MIN_TOOL_Z" --replay-delay 0.2
```

With the default delay of zero, there is no server/network latency. This is not
continuous 8.42-second playback: the chunk pauses and optional delay extend it.
Like the live loop, pausing clears queued actions, so a mid-chunk pause discards
the remaining rows of that chunk and resumes at the next chunk. Console messages
show the loaded frame ranges. Supply the usual positional recording path before
the flags to save the GUI telemetry, e.g. `python play.py replay.rrd
--replay-episode --min-z "$MIN_TOOL_Z"`. Without `--replay-episode`, normal server
inference is selected.

## Replay absolute CSV positions through the waypoint executor

```bash
python play.py absolute_replay.rrd --replay-targets-csv --min-z "$MIN_TOOL_Z" --dt 0.02
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
