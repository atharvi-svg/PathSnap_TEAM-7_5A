from __future__ import annotations

import pandas as pd

from models.predict import predict_candidates
from route_matching.hmm import decode_trip
from route_matching.road_graph import RoadGraph
from src.data.candidate_generation import generate_candidates
from src.data.feature_engineering import extract_features


def reconstruct_trip(gps: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame,
                     *, trip_id: str, partition: str, model=None):
    """Candidate generation -> model scoring -> network-aware Viterbi decoding."""
    candidates = generate_candidates(gps, nodes, arcs, top_k=30)
    candidates = extract_features(candidates, gps)
    candidates.insert(0, "trip_id", str(trip_id))
    candidates.insert(0, "partition", partition)
    scored = candidates.merge(predict_candidates(candidates, model),
        on=["partition", "trip_id", "observation_id", "candidate_arc_id"],
        validate="one_to_one")
    graph = RoadGraph(nodes, arcs)
    observations, route = decode_trip(scored, gps, graph)
    return {"candidate_rows": scored, "observations": observations,
            "route_arc_ids": route["expanded_route_arc_ids"],
            "classifier_arc_ids": route["classifier_arc_ids"],
            "hmm_arc_ids": route["hmm_arc_ids"], "decoder_stats": route,
            "graph": graph}
