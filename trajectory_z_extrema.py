"""Print the minimum and maximum intended Z in an absolute-position CSV."""

import argparse
import csv
from pathlib import Path


DEFAULT_CSV = (
    Path(__file__).parent
    / "data/replay/eggplant_episode_002/intended_absolute_positions.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    with args.csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        parser.error(f"No data rows in {args.csv_path}")

    try:
        z_values = [(int(row["frame_index"]), float(row["target_z_m"])) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(f"Expected numeric frame_index and target_z_m columns: {exc}")

    min_frame, min_z = min(z_values, key=lambda item: item[1])
    max_frame, max_z = max(z_values, key=lambda item: item[1])
    print(f"Min intended Z: {min_z:.6f} m (frame {min_frame})")
    print(f"Max intended Z: {max_z:.6f} m (frame {max_frame})")


if __name__ == "__main__":
    main()
