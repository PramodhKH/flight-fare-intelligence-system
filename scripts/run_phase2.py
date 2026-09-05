#!/usr/bin/env python
"""Run Phase 2 analytics, EDA figure generation, and leakage-aware splitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flight_fare_intelligence.analytics import build_market_summary, generate_eda_figures
from flight_fare_intelligence.data import load_raw_dataset, validate_dataset
from flight_fare_intelligence.splitting import create_split_assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", default="data/raw/Flight_Booking.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-dir", default="reports/metrics")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--processed-dir", default="data/processed")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = load_raw_dataset(args.data)
    validate_dataset(df, strict=True, path=args.data)

    market_summary = build_market_summary(df)
    assignments, split_summary = create_split_assignments(df, random_seed=args.seed)
    figures = generate_eda_figures(df, args.figures_dir)

    metrics_dir = Path(args.metrics_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    write_json(metrics_dir / "phase2_market_summary.json", market_summary)
    write_json(metrics_dir / "phase2_split_summary.json", split_summary.to_dict())
    assignments.to_csv(processed_dir / "phase2_split_assignments.csv", index=False)

    output = {
        "market_summary": str(metrics_dir / "phase2_market_summary.json"),
        "split_summary": str(metrics_dir / "phase2_split_summary.json"),
        "split_assignments": str(processed_dir / "phase2_split_assignments.csv"),
        "figures": [str(path) for path in figures],
        "rows": len(df),
        "scenario_overlap_count": split_summary.scenario_overlap_count,
        "split_rows": split_summary.rows,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
