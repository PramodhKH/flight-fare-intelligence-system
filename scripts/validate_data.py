#!/usr/bin/env python
"""Validate the raw Flight Booking dataset against the project contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flight_fare_intelligence.data import load_raw_dataset, validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="data/raw/Flight_Booking.csv")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", default="reports/metrics/phase1_data_validation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_raw_dataset(args.path)
    summary = validate_dataset(df, strict=args.strict, path=args.path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary.to_dict(), indent=2))
    print(f"Validation report written to {output}")


if __name__ == "__main__":
    main()
