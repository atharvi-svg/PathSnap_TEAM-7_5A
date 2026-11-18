"""Validated loader for the raw PathSnap trip-file format.

This module deliberately keeps the reference route separate from inference
inputs.  Callers can request it while training/analysing, but deployed
prediction code should use ``include_route=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TripData:
    """All raw data associated with one PathSnap trip."""

    trip_id: str
    gps: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: pd.DataFrame | None


def discover_trip_ids(data_root: str | Path, partition: str) -> list[str]:
    """Return sorted trip IDs from a dataset partition."""
    partition_path = Path(data_root) / partition
    if not partition_path.is_dir():
        raise FileNotFoundError(f"Partition does not exist: {partition_path}")
    return sorted(path.name for path in partition_path.iterdir() if path.is_dir())


def _read_whitespace(path: Path, names: list[str], dtype: dict[str, str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required PathSnap file is missing: {path}")
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)


def load_trip(
    data_root: str | Path,
    partition: str,
    trip_id: str,
    *,
    include_route: bool = True,
    validate: bool = True,
) -> TripData:
    """Load a trip using the shared team contract.

    DataFrames include explicit zero-based IDs derived from their raw row order.
    ``route`` is optional so inference code cannot accidentally consume labels.
    """
    trip_dir = Path(data_root) / partition / str(trip_id)
    prefix = trip_dir / str(trip_id)

    gps = _read_whitespace(
        prefix.with_suffix(".track"),
        ["longitude", "latitude", "timestamp_s"],
        {"longitude": "float64", "latitude": "float64", "timestamp_s": "float64"},
    )
    gps.insert(0, "observation_id", range(len(gps)))

    nodes = _read_whitespace(
        prefix.with_suffix(".nodes"),
        ["longitude", "latitude"],
        {"longitude": "float64", "latitude": "float64"},
    )
    nodes.insert(0, "node_id", range(len(nodes)))

    arcs = _read_whitespace(
        prefix.with_suffix(".arcs"),
        ["from_node_id", "to_node_id"],
        {"from_node_id": "int64", "to_node_id": "int64"},
    )
    arcs.insert(0, "arc_id", range(len(arcs)))

    route: pd.DataFrame | None = None
    if include_route:
        route = _read_whitespace(
            prefix.with_suffix(".route"), ["arc_id"], {"arc_id": "int64"}
        )
        route.insert(0, "route_position", range(len(route)))

    trip = TripData(str(trip_id), gps, nodes, arcs, route)
    if validate:
        validate_trip(trip)
    return trip


def validate_trip(trip: TripData) -> None:
    """Raise a clear error when IDs or coordinate values are unusable."""
    for name, frame in (("gps", trip.gps), ("nodes", trip.nodes)):
        if frame.empty:
            raise ValueError(f"{trip.trip_id}: {name} is empty")
        if frame[["longitude", "latitude"]].isna().any().any():
            raise ValueError(f"{trip.trip_id}: {name} contains missing coordinates")
        if not frame["longitude"].between(-180, 180).all() or not frame["latitude"].between(-90, 90).all():
            raise ValueError(f"{trip.trip_id}: {name} has invalid longitude/latitude")

    node_count = len(trip.nodes)
    if not trip.arcs["from_node_id"].between(0, node_count - 1).all() or not trip.arcs[
        "to_node_id"
    ].between(0, node_count - 1).all():
        raise ValueError(f"{trip.trip_id}: arcs reference a node outside the node table")
    if not trip.gps["timestamp_s"].is_monotonic_increasing:
        raise ValueError(f"{trip.trip_id}: GPS timestamps are not ordered")

    if trip.route is not None and not trip.route["arc_id"].between(0, len(trip.arcs) - 1).all():
        raise ValueError(f"{trip.trip_id}: route references an arc outside the arc table")
