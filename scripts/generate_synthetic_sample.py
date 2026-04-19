# -*- coding: utf-8 -*-
"""
Regenerate sample_data/processed/*_metrics.csv with random data + weak synthetic signal.

Run from repo root:  python scripts/generate_synthetic_sample.py

Output is NOT real NBA statistics; it exists only so the public pipeline can run without licensed data.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sample_data" / "processed"


def main() -> None:
    random.seed(42)
    teams = [f"T{i}" for i in range(1, 9)]
    years = list(range(2014, 2026))
    metrics = [f"metric_{i}" for i in range(1, 13)]
    OUT.mkdir(parents=True, exist_ok=True)
    for y in years:
        path = OUT / f"{y}_metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Team", "Win Rate"] + metrics)
            for t in teams:
                mvals = [random.uniform(-1.5, 1.5) for _ in metrics]
                signal = 0.5 + 0.08 * mvals[0] - 0.05 * mvals[1] + 0.04 * mvals[2]
                signal += random.gauss(0, 0.05)
                wr = max(0.2, min(0.8, signal))
                w.writerow([t, round(wr, 3)] + [round(x, 4) for x in mvals])
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
