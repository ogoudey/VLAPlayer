"""Replay exported dataset deltas through KinovaConnection; inspect by default.

Run in the same Python environment as play.py. No cameras or policy server needed.
"""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time


DEFAULT_EPISODE = Path(__file__).parent / "data/replay/eggplant_episode_002/episode.json"


def load_episode(path):
    episode = json.loads(Path(path).read_text())
    fps = episode["fps"]
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Episode fps must be positive and finite")
    frames = episode["frames"]
    if not frames:
        raise ValueError("Episode has no frames")
    for i, frame in enumerate(frames):
        if frame["frame_index"] != i:
            raise ValueError("Frames must be contiguous and ordered from zero")
        if frame["episode_index"] != episode["episode_index"]:
            raise ValueError("Mixed episode indices")
        for key, size in (("action", 7), ("observation.state", 15)):
            values = frame[key]
            if len(values) != size or not all(math.isfinite(v) for v in values):
                raise ValueError(f"Invalid {key} at frame {i}")
        if not 0 <= frame["action"][6] <= 1:
            raise ValueError(f"Invalid gripper target at frame {i}")
        if abs(frame["timestamp"] - i / fps) > 1e-4:
            raise ValueError(f"Unexpected timestamp at frame {i}")
    return episode


def action_kwargs(values):
    """Match THREE_TASKS_2_FIXED_LORA: metres, rotvec degrees, absolute gripper."""
    return dict(zip(
        ("dx", "dy", "dz", "d_theta_x", "d_theta_y", "d_theta_z", "gripper_command"),
        [*values[:3], *(math.degrees(v) for v in values[3:6]), values[6]],
    ))


def replay(episode, *, min_z, log_path=None):
    # Lazy imports keep inspection independent of robot libraries/hardware.
    from connections.kinova import KinovaConnection
    from schemas import CartesianDelta

    dt = 1 / episode["fps"]
    frames = episode["frames"]
    if not math.isfinite(min_z):
        raise ValueError("min_z must be finite")
    records = []
    connection = KinovaConnection(control_period_s=dt)
    try:
        # Preserve the manually prepared start pose; do not run Retract or open gripper.
        connection.awake(home=False)
        deadline = time.monotonic() + 3
        while True:
            state = connection.state()
            if state.joint_angles and len(state.joint_angles) == 7:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("No valid robot feedback within 3 seconds")
            time.sleep(0.01)

        expected = [math.degrees(q) for q in frames[0]["observation.state"][:7]]
        errors = [abs((q - ref + 180) % 360 - 180)
                  for q, ref in zip(state.joint_angles, expected)]
        if max(errors) > 5:
            raise RuntimeError(
                f"Start joints differ from recording by up to {max(errors):.1f} degrees. "
                "Position the arm at the printed start joints before replay."
            )

        start = time.monotonic()
        next_tick = start
        for frame in frames:
            time.sleep(max(0, next_tick - time.monotonic()))
            state = connection.state()
            dz = frame["action"][2]
            next_z = state.target_pose.z + max(-0.15, min(0.15, dz / dt)) * dt
            if min(state.target_pose.z, next_z) <= min_z:
                raise RuntimeError(f"Stopped at tool Z={state.target_pose.z:.4f} m (min-z={min_z})")
            before = time.monotonic()
            connection.apply_action(CartesianDelta(**action_kwargs(frame["action"])))
            after = time.monotonic()
            records.append({
                "frame_index": frame["frame_index"],
                "send_started_s": before - start,
                "send_finished_s": after - start,
                "recorded_state": frame["observation.state"],
                "measured_state": state.model_dump(),
                "action": frame["action"],
            })
            # No catch-up bursts after a slow RPC; record actual timing for analysis.
            next_tick = max(next_tick + dt, before + dt)
        time.sleep(max(0, next_tick - time.monotonic()))
    finally:
        # Also stop on Ctrl+C, command failures, or a failed start-pose check.
        try:
            connection.base.Stop()
        finally:
            connection.kill_the_thread = True
            if connection.cyclic_thread is not None:
                connection.cyclic_thread.join(timeout=2)
            try:
                connection.udp_connection.__exit__()
            finally:
                connection.tcp_connection.__exit__()
                if log_path is not None:
                    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(log_path).write_text(json.dumps({
                        "repo_id": episode["repo_id"], "revision": episode["revision"],
                        "episode_index": episode["episode_index"], "fps": episode["fps"],
                        "gain": 1.0, "min_z": min_z, "frames": records,
                    }, indent=2) + "\n")
                    print(f"Saved {len(records)} replay records to {log_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", nargs="?", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--execute", action="store_true", help="Connect to the real arm and send actions")
    parser.add_argument("--min-z", type=float, help="Stop at this Kortex-reported tool Z in metres")
    parser.add_argument("--log", type=Path, help="Measured-state and command-timing JSON output")
    args = parser.parse_args()
    if args.min_z is not None and not math.isfinite(args.min_z):
        parser.error("--min-z must be finite")
    episode = load_episode(args.episode)
    frames = episode["frames"]
    state = frames[0]["observation.state"]
    print(f"{episode['repo_id']} @ {episode['revision']}")
    print(f"Episode index {episode['episode_index']}: {episode['task']}")
    print(f"{len(frames)} actions, {episode['fps']} Hz, {len(frames)/episode['fps']:.2f} seconds")
    print("Start joints (degrees):", [round(math.degrees(q) % 360, 3) for q in state[:7]])
    print("Recorded start tool XYZ (metres):", state[7:10])
    print("Recorded start gripper (0=open, 1=closed):", state[14])
    print("Raw deltas; gain=1; existing Kinova velocity limits; continuous playback.")
    if not args.execute:
        print("Inspection only. Use --execute to replay from the manually prepared start pose.")
        return
    if args.min_z is None:
        parser.error("--execute requires --min-z in the robot's configured tool frame")
    log_path = args.log or args.episode.parent / f"replay_{datetime.now():%Y%m%dT%H%M%S%f}.json"
    try:
        replay(episode, min_z=args.min_z, log_path=log_path)
    except KeyboardInterrupt:
        print("Replay interrupted.")


if __name__ == "__main__":
    main()
