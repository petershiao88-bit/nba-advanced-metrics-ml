# -*- coding: utf-8 -*-
"""
Resolve where team-season CSV panels live (local licensed data vs bundled demo sample).
"""
from __future__ import annotations

from pathlib import Path


def resolve_processed_data_dir(repo_root: Path) -> Path:
    """
    Prefer ``data/processed/`` (your own licensed CSVs, not tracked in git).
    Fall back to ``sample_data/processed/`` (synthetic demo only).
    """
    candidates = [
        repo_root / "data" / "processed",
        repo_root / "sample_data" / "processed",
    ]
    for d in candidates:
        if d.is_dir() and list(d.glob("*_metrics.csv")):
            return d
    raise FileNotFoundError(
        "No *_metrics.csv found. Add licensed files under data/processed/ "
        "or rely on sample_data/processed/ (see data/README.md)."
    )
