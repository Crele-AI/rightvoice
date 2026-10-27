#!/usr/bin/env python3
"""Positive controls: three systems with known answers, run before trusting any number.

    control_clean    the clean target stem returned as the output: words lost near the
                     recogniser's floor, wrong-voice words near zero
    control_mixture  the mixture returned unchanged: high wrong-voice words
    control_silence  nothing returned: zero words per minute on target-absent items

The first two are symbolic links to files the build already holds, so they cost no disk; the
third is an empty folder, because the scorer treats a missing output as empty output.

    python runners/controls.py --build data/build/v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CONTROLS = {"control_clean": "target.wav", "control_mixture": "mixture.wav", "control_silence": None}


def write_controls(build: Path, outputs: Path | None = None) -> dict[str, int]:
    outputs = outputs or build / "outputs"
    rows = [json.loads(line) for line in (build / "manifest.jsonl").read_text().splitlines() if line.strip()]
    counts: dict[str, int] = {}
    for system, stem in CONTROLS.items():
        folder = outputs / system
        folder.mkdir(parents=True, exist_ok=True)
        n = 0
        for row in rows:
            if stem is None:
                continue
            source = build / "items" / row["id"] / stem
            link = folder / f"{row['id']}.wav"
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(os.path.relpath(source, folder), link)
            n += 1
        counts[system] = n
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--outputs", type=Path, default=None, help="default <build>/outputs")
    args = parser.parse_args(argv)
    counts = write_controls(args.build, args.outputs)
    for system, n in counts.items():
        print(f"{system}: {n} outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
