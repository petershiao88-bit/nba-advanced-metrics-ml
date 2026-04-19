# -*- coding: utf-8 -*-
"""
Entry point: load processed team-season CSVs and run the full ML pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_layout import resolve_processed_data_dir  # noqa: E402
from nba_win_rate_pipeline import load_metrics_from_csvs, run_full_pipeline  # noqa: E402


def main() -> int:
    plot_dir = ROOT / "outputs" / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        csv_dir = resolve_processed_data_dir(ROOT)
    except FileNotFoundError as e:
        print(e)
        return 1

    print(f"Loading data from: {csv_dir}\n")
    data = load_metrics_from_csvs(str(csv_dir))
    if "Team" in data.columns and "team" not in data.columns:
        data = data.rename(columns={"Team": "team"})

    run_full_pipeline(
        data,
        target_col="win_rate",
        split_mode="year_holdout",
        save_extra_plots=True,
        plot_dir=str(plot_dir),
        show_plots=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
