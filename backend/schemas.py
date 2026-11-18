from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=64)


class TripOption(BaseModel):
    trip_id: str
    observations: int


class PredictResponse(BaseModel):
    trip_id: str
    partition: str
    observation_count: int
    predicted_arc_ids: list[int]
    predicted_route_coordinates: list[list[float]]
    gps_coordinates: list[list[float]]
    metrics: dict[str, float | int | None]
    note: str
