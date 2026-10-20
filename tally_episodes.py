"""Tally episode annotations (success / failure) across a directory of episode folders.

Usage:
    python tally_episodes.py episodes/
    python tally_episodes.py episodes/ --by prompt
    python tally_episodes.py episodes/ --by client_setting
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_episodes(root: Path):
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        meta_path = folder / "metadata.json"
        if not meta_path.exists():
            yield folder, None  # never saved (or still recording)
            continue
        try:
            yield folder, json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            yield folder, "unreadable"


def rate(counts: Counter) -> str:
    annotated = counts["success"] + counts["failure"]
    return f"{counts['success'] / annotated:.0%}" if annotated else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="Folder containing episode folders")
    parser.add_argument("--by", metavar="FIELD", help="Also break down by a metadata field, e.g. prompt or client_setting")
    args = parser.parse_args()

    totals = Counter()
    groups = defaultdict(Counter)
    missing_rrd = []

    for folder, meta in load_episodes(args.directory):
        if meta is None:
            totals["no metadata"] += 1
            continue
        if meta == "unreadable":
            totals["unreadable metadata"] += 1
            continue
        label = meta.get("episode_annotation", "<no annotation>")
        totals[label] += 1
        if args.by:
            groups[str(meta.get(args.by, "<missing>"))][label] += 1
        if meta.get("rrd_present") is False or not (folder / "episode.rrd").exists():
            missing_rrd.append(folder.name)

    print(f"Episodes in {args.directory}: {sum(totals.values())}")
    for label, count in totals.most_common():
        print(f"  {label:20s} {count}")
    print(f"  {'success rate':20s} {rate(totals)}  (of annotated episodes)")

    if args.by:
        print(f"\nBy {args.by}:")
        for key, counts in sorted(groups.items()):
            print(f"  {key}")
            print(f"      success {counts['success']}, failure {counts['failure']}, rate {rate(counts)}")

    if missing_rrd:
        print(f"\nAnnotated but missing episode.rrd ({len(missing_rrd)}): {', '.join(missing_rrd)}")


if __name__ == "__main__":
    main()
