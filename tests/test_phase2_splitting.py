from __future__ import annotations

import pandas as pd

from flight_fare_intelligence.splitting import create_split_assignments


def _synthetic_dataset() -> pd.DataFrame:
    rows = []
    record_id = 0
    airlines = ["Vistara", "Indigo"]
    routes = [("Delhi", "Mumbai"), ("Mumbai", "Delhi")]
    classes = ["Economy", "Business"]
    horizons = [2, 5, 10, 12, 18, 20, 25, 30, 40, 45]

    for source, destination in routes:
        for cabin_class in classes:
            for days_left in horizons:
                for scenario_index in range(6):
                    airline = airlines[scenario_index % 2]
                    row = {
                        "Unnamed: 0": record_id,
                        "airline": airline,
                        "source_city": source,
                        "destination_city": destination,
                        "departure_time": "Morning" if scenario_index < 3 else "Evening",
                        "stops": "zero" if scenario_index % 3 else "one",
                        "class": cabin_class,
                        "duration": 2.0 + scenario_index / 10,
                        "days_left": days_left,
                        "price": 5000 + record_id,
                    }
                    rows.append(row)
                    record_id += 1
                    # Duplicate some exact deployed scenarios to ensure grouping is respected.
                    if scenario_index == 0:
                        duplicate = row.copy()
                        duplicate["Unnamed: 0"] = record_id
                        duplicate["price"] += 250
                        rows.append(duplicate)
                        record_id += 1
    return pd.DataFrame(rows)


def test_split_is_complete_and_scenario_grouped() -> None:
    df = _synthetic_dataset()
    assignments, summary = create_split_assignments(df, random_seed=42)

    assert len(assignments) == len(df)
    assert set(assignments["split"]) == {"train", "validation", "test"}
    assert assignments.groupby("scenario_id")["split"].nunique().max() == 1
    assert summary.scenario_overlap_count == 0


def test_split_is_reproducible() -> None:
    df = _synthetic_dataset()
    first, _ = create_split_assignments(df, random_seed=42)
    second, _ = create_split_assignments(df, random_seed=42)
    pd.testing.assert_frame_equal(first, second)
