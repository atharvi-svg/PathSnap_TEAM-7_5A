"""Public route-prediction interface for Person 4 integration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .hmm import decode_viterbi
from .road_graph import RoadGraph


@dataclass
class RoutePrediction:
    trip_id: str
    arc_ids: list[int]
    observation_arc_ids: list[int]
    metrics: dict[str, Any]
    candidates: pd.DataFrame


def predict_route(
    gps: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    *,
    trip_id: str = "trip",
    partition: str = "inference",
    model: Any = None,
    candidate_top_k: int = 30,
    transition_scale_m: float = 45.0,
) -> RoutePrediction:
    """Return ordered road arc IDs and route metrics for one GPS trip.

    Expects data frames compatible with ``src.data.candidate_generation`` and
    the Person 2 ``models.predict.predict_candidates`` contract. Reference
    route labels are never accessed.
    """
    if gps.empty:
        raise ValueError("gps must contain at least one observation")
    gps = gps.copy().reset_index(drop=True)
    if "observation_id" not in gps:
        gps.insert(0, "observation_id", np.arange(len(gps), dtype=np.int64))
    from src.data.candidate_generation import generate_candidates
    from src.data.feature_engineering import extract_features
    from models.predict import predict_candidates

    candidates = generate_candidates(gps, nodes, arcs, top_k=candidate_top_k)
    candidates = extract_features(candidates, gps)
    candidates.insert(0, "trip_id", str(trip_id))
    candidates.insert(0, "partition", str(partition))
    predictions = predict_candidates(candidates, model=model)
    scored = candidates.merge(predictions, on=["partition", "trip_id", "observation_id", "candidate_arc_id"],
                             how="left", validate="one_to_one")
    if scored.candidate_probability.isna().any():
        raise RuntimeError("candidate predictor did not return a probability for every candidate")
    graph = RoadGraph.from_tables(nodes, arcs)
    decoded = decode_viterbi(scored, gps, graph, transition_scale_m=transition_scale_m)
    top = (scored.sort_values(["observation_id", "candidate_probability", "candidate_arc_id"],
                              ascending=[True, False, True])
           .drop_duplicates("observation_id").sort_values("observation_id"))
    route = decoded.route_arc_ids
    total_length = float(sum(graph.lengths_m[a] for a in route))
    metrics = {
        "observation_count": int(len(gps)),
        "candidate_count": int(len(scored)),
        "classifier_top1_observations": int(len(top)),
        "decoded_observation_arc_count": int(len(decoded.states)),
        "route_arc_count": int(len(route)),
        "route_length_m": total_length,
        "connected_transition_ratio": float(decoded.connectivity),
        "viterbi_log_score": float(decoded.log_score),
        "candidate_top_k": int(candidate_top_k),
    }
    return RoutePrediction(str(trip_id), route, decoded.states, metrics, scored)
