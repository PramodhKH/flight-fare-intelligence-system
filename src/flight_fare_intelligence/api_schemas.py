"""Pydantic contracts for the Phase 8 production API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Airline = Literal["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"]
City = Literal["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
DeparturePeriod = Literal[
    "Afternoon",
    "Early_Morning",
    "Evening",
    "Late_Night",
    "Morning",
    "Night",
]
Stops = Literal["zero", "one", "two_or_more"]
CabinClass = Literal["Business", "Economy"]
WhatIfFeature = Literal["airline", "departure_time", "stops", "days_left"]


class FlightScenario(BaseModel):
    """Validated deployed feature contract for one fare prediction."""

    model_config = ConfigDict(populate_by_name=True)

    airline: Airline
    source_city: City
    destination_city: City
    departure_time: DeparturePeriod
    stops: Stops
    cabin_class: CabinClass = Field(alias="class")
    duration: float = Field(gt=0.0, le=50.0)
    days_left: int = Field(ge=1, le=49)

    @model_validator(mode="after")
    def validate_route(self) -> FlightScenario:
        if self.source_city == self.destination_city:
            raise ValueError("source_city and destination_city must be different")
        return self

    def as_model_dict(self) -> dict[str, str | float | int]:
        """Return the exact eight-feature dictionary expected by the model."""
        return self.model_dump(by_alias=True)


class BatchPredictionRequest(BaseModel):
    """Bounded batch inference contract."""

    scenarios: list[FlightScenario] = Field(min_length=1, max_length=100)
    include_explanations: bool = False


class WhatIfRequest(BaseModel):
    """One-feature-at-a-time counterfactual request."""

    scenario: FlightScenario
    feature: WhatIfFeature
    values: list[str | int | float] = Field(min_length=1, max_length=49)
