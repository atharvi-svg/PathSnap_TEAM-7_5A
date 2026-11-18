from __future__ import annotations

import numpy as np
import pandas as pd

from route_matching.road_graph import RoadGraph, haversine_m
from route_matching.viterbi import viterbi_decode


def _transition_matrix(previous: pd.DataFrame, current: pd.DataFrame, graph: RoadGraph,
                       gps_distance_m: float, *, beta_m: float) -> np.ndarray:
    matrix = np.full((len(previous), len(current)), -np.inf, dtype=float)
    beta = max(float(beta_m), 15.0)
    for i, prev in enumerate(previous.itertuples(index=False)):
        for j, curr in enumerate(current.itertuples(index=False)):
            distance = graph.candidate_path_distance(
                prev.candidate_arc_id, prev.segment_fraction,
                curr.candidate_arc_id, curr.segment_fraction)
            if np.isfinite(distance):
                matrix[i, j] = -abs(distance - gps_distance_m) / beta
    return matrix


def decode_trip(candidates: pd.DataFrame, gps: pd.DataFrame, graph: RoadGraph,
                *, max_gap_s: float = 60.0, beta_m: float = 50.0, max_states: int = 12):
    """Return one HMM arc state per GPS observation plus classifier-only states.

    Hidden states are the generated candidate road arcs. Emissions are log
    candidate probabilities. Transitions score agreement between observed GPS
    displacement and directed shortest-path distance on the road graph.
    """
    required = {"observation_id", "candidate_arc_id", "candidate_probability", "segment_fraction", "arc_length_m"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing HMM candidate columns: {sorted(missing)}")
    gps_ordered = gps.sort_values("observation_id").reset_index(drop=True)
    by_obs = {int(k): g.sort_values("candidate_probability", ascending=False).head(max_states).reset_index(drop=True)
              for k, g in candidates.groupby("observation_id", sort=False)}
    obs_ids = [int(x) for x in gps_ordered.observation_id]
    if any(obs not in by_obs for obs in obs_ids):
        raise ValueError("Every GPS observation must have at least one candidate state")

    layers = [by_obs[obs] for obs in obs_ids]
    emissions = [np.log(np.clip(layer.candidate_probability.to_numpy(dtype=float), 1e-12, 1.0))
                 for layer in layers]
    transitions: list[np.ndarray] = []
    breaks_by_gap = set()
    for t in range(1, len(layers)):
        before = gps_ordered.iloc[t - 1]
        after = gps_ordered.iloc[t]
        dt = float(after.timestamp_s - before.timestamp_s)
        if dt <= 0 or dt > max_gap_s:
            transitions.append(np.full((len(layers[t - 1]), len(layers[t])), -np.inf))
            breaks_by_gap.add(t)
            continue
        displacement = haversine_m(before.longitude, before.latitude, after.longitude, after.latitude)
        transitions.append(_transition_matrix(layers[t - 1], layers[t], graph, displacement,
                                               beta_m=max(beta_m, displacement * .25)))

    state_indices, decode_stats = viterbi_decode(emissions, transitions)
    hmm_arcs = [int(layers[i].iloc[state].candidate_arc_id) for i, state in enumerate(state_indices)]
    classifier_arcs = [int(layer.iloc[0].candidate_arc_id) for layer in layers]
    records = []
    restarts = set(decode_stats["restart_indices"]) | breaks_by_gap
    for i, obs_id in enumerate(obs_ids):
        selected = layers[i].iloc[state_indices[i]]
        records.append({"observation_id": obs_id,
            "classifier_arc_id": classifier_arcs[i], "classifier_probability": float(layers[i].iloc[0].candidate_probability),
            "hmm_arc_id": hmm_arcs[i], "hmm_probability": float(selected.candidate_probability),
            "hmm_state_rank": int(selected.candidate_rank) if "candidate_rank" in selected.index else None,
            "segment_fraction": float(selected.segment_fraction),
            "decoder_restart": bool(i in restarts)})
    decoded = pd.DataFrame(records)
    expanded_route = graph.expand_arc_sequence(hmm_arcs)
    return decoded, {**decode_stats, "gps_gap_restart_count": len(breaks_by_gap),
                    "max_candidate_states": max_states,
                    "expanded_route_arc_ids": expanded_route,
                    "classifier_arc_ids": classifier_arcs, "hmm_arc_ids": hmm_arcs}
