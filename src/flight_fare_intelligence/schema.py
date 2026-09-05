"""Dataset and model feature contracts."""

from __future__ import annotations

RAW_REQUIRED_COLUMNS = [
    "airline",
    "flight",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
    "duration",
    "days_left",
    "price",
]

OPTIONAL_EXPORT_COLUMNS = ["Unnamed: 0"]

MODEL_FEATURES = [
    "airline",
    "source_city",
    "destination_city",
    "departure_time",
    "stops",
    "class",
    "duration",
    "days_left",
]

ANALYSIS_ONLY_COLUMNS = ["flight", "arrival_time"]
TARGET_COLUMN = "price"

EXPECTED_CATEGORIES = {
    "airline": {"AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"},
    "source_city": {"Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"},
    "destination_city": {"Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"},
    "departure_time": {"Afternoon", "Early_Morning", "Evening", "Late_Night", "Morning", "Night"},
    "arrival_time": {"Afternoon", "Early_Morning", "Evening", "Late_Night", "Morning", "Night"},
    "stops": {"one", "two_or_more", "zero"},
    "class": {"Business", "Economy"},
}

EXPECTED_STRICT_ROWS = 300_153
EXPECTED_STRICT_ROUTES = 30
EXPECTED_DAYS_LEFT_RANGE = (1, 49)
