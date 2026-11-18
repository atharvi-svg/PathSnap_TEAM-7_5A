from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from backend.config import DATA_ROOT, MODEL_PATH, TRIP_PARTITION
from route_matching.route_evaluation import evaluate_trip
from route_matching.route_reconstruction import reconstruct_trip
from src.data import discover_trip_ids, load_trip


@lru_cache(maxsize=1)
def load_selected_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def available_trips() -> list[dict]:
    trip_ids = discover_trip_ids(DATA_ROOT, TRIP_PARTITION)
    options = []
    for trip_id in trip_ids:
        trip = load_trip(DATA_ROOT, TRIP_PARTITION, trip_id, include_route=False)
        options.append({"trip_id": trip_id, "observations": int(len(trip.gps))})
    return options


def predict_trip(trip_id: str) -> dict:
    if trip_id not in discover_trip_ids(DATA_ROOT, TRIP_PARTITION):
        raise ValueError(f"Trip {trip_id!r} is not available in partition {TRIP_PARTITION!r}")
    # Inference always begins without route labels. Evaluator partitions never
    # load labels; only the permitted student_test development split is scored.
    trip = load_trip(DATA_ROOT, TRIP_PARTITION, trip_id, include_route=False)
    result = reconstruct_trip(trip.gps, trip.nodes, trip.arcs, trip_id=trip_id,
                              partition=TRIP_PARTITION, model=load_selected_model())
    # Reference routes are permitted only for the bundled development partition.
    # Never open evaluator labels or send derived scores to the student-side UI.
    metrics = {}
    if TRIP_PARTITION == "student_test":
        labelled_trip = load_trip(DATA_ROOT, TRIP_PARTITION, trip_id, include_route=True)
        inference_candidates = result["candidate_rows"].drop(
            columns=["reference_arc_id", "reference_route_distance_m", "label"], errors="ignore")
        metrics, _ = evaluate_trip(labelled_trip.gps, labelled_trip.nodes, labelled_trip.arcs,
                                   labelled_trip.route, inference_candidates,
                                   result["observations"], result["graph"])
        metrics = {k: (float(v) if pd.notna(v) else None) for k, v in metrics.items()}
    gps_coords = [[float(lon), float(lat)] for lon, lat in zip(trip.gps.longitude, trip.gps.latitude)]
    return {"trip_id": trip_id, "partition": TRIP_PARTITION,
            "observation_count": int(len(trip.gps)),
            "predicted_arc_ids": [int(x) for x in result["route_arc_ids"]],
            "predicted_route_coordinates": result["graph"].arc_coordinates(result["route_arc_ids"]),
            "gps_coordinates": gps_coords, "metrics": metrics,
            "note": ("Development metrics use weak route-proximity references; no reference route is returned."
                     if TRIP_PARTITION == "student_test" else
                     "Evaluator ground-truth labels are not accessed or returned.")}
