import pandas as pd
import pytest

from flight_fare_intelligence.data import DataValidationError, validate_dataset


def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "airline": ["Vistara", "Indigo"],
            "flight": ["UK-995", "6E-200"],
            "source_city": ["Delhi", "Mumbai"],
            "departure_time": ["Morning", "Evening"],
            "stops": ["zero", "one"],
            "arrival_time": ["Afternoon", "Night"],
            "destination_city": ["Mumbai", "Delhi"],
            "class": ["Economy", "Economy"],
            "duration": [2.25, 4.5],
            "days_left": [12, 20],
            "price": [7350, 8200],
        }
    )


def test_valid_sample_passes_non_strict_validation():
    summary = validate_dataset(sample_df())
    assert summary.rows == 2
    assert summary.missing_values == 0


def test_missing_required_column_fails():
    df = sample_df().drop(columns=["price"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_dataset(df)


def test_invalid_category_fails():
    df = sample_df()
    df.loc[0, "class"] = "First"
    with pytest.raises(DataValidationError, match="Unexpected values in class"):
        validate_dataset(df)


def test_invalid_days_left_fails():
    df = sample_df()
    df.loc[0, "days_left"] = 0
    with pytest.raises(DataValidationError, match="days_left must be between"):
        validate_dataset(df)


def test_same_source_destination_fails():
    df = sample_df()
    df.loc[0, "destination_city"] = "Delhi"
    with pytest.raises(DataValidationError, match="identical source and destination"):
        validate_dataset(df)
