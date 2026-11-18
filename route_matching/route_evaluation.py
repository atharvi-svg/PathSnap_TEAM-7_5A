from __future__ import annotations

import numpy as np
import pandas as pd

from route_matching.road_graph import RoadGraph
from src.data.feature_engineering import derive_reference_labels


def _edit_similarity(a: list[int], b: list[int]) -> float:
    if not a and not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b), 1)


def _collapse(values: list[int]) -> list[int]:
    out = []
    for value in values:
        if not out or int(value) != out[-1]:
            out.append(int(value))
    return out


def evaluate_trip(gps: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame,
                  route: pd.DataFrame, candidate_rows: pd.DataFrame,
                  decoded: pd.DataFrame, graph: RoadGraph) -> tuple[dict, pd.DataFrame]:
    """Evaluate predictions against weak nearest-route geometry, after inference."""
    labelled, label_stats = derive_reference_labels(candidate_rows, gps, nodes, arcs, route,
                                                     label_distance_threshold_m=50.0)
    truth = (labelled.groupby("observation_id", sort=True)
             .agg(reference_arc_id=("reference_arc_id", "first"),
                  reference_route_distance_m=("reference_route_distance_m", "first"))
             .reset_index())
    # Candidate-miss calculation is route-label analysis only and happens after scoring.
    candidate_match = labelled.assign(_match=labelled.candidate_arc_id.eq(labelled.reference_arc_id))
    has_candidate = candidate_match.groupby("observation_id", sort=True)._match.any().rename("has_reference_candidate")
    truth = truth.merge(has_candidate, on="observation_id", how="left")
    per_obs = decoded.merge(truth, on="observation_id", how="left", validate="one_to_one")
    per_obs["valid_for_weak_evaluation"] = per_obs.reference_route_distance_m.le(50.0)
    valid = per_obs.loc[per_obs.valid_for_weak_evaluation].copy()
    valid["classifier_correct"] = valid.classifier_arc_id.eq(valid.reference_arc_id)
    valid["hmm_correct"] = valid.hmm_arc_id.eq(valid.reference_arc_id)

    # Route connectivity uses directed shortest-path reachability between the
    # selected arc positions; gaps over 60 seconds are treated as new sequences.
    cand_lookup = candidate_rows.set_index(["observation_id", "candidate_arc_id"])
    def continuity(arc_column: str) -> float:
        values, obs = per_obs[arc_column].to_list(), per_obs.observation_id.to_list()
        times = gps.set_index("observation_id").timestamp_s
        connected, eligible = 0, 0
        for i in range(1, len(values)):
            dt = float(times.loc[obs[i]] - times.loc[obs[i - 1]])
            if dt <= 0 or dt > 60:
                continue
            eligible += 1
            f0 = float(cand_lookup.loc[(obs[i - 1], values[i - 1]), "segment_fraction"])
            f1 = float(cand_lookup.loc[(obs[i], values[i]), "segment_fraction"])
            if np.isfinite(graph.candidate_path_distance(values[i - 1], f0, values[i], f1)):
                connected += 1
        return connected / eligible if eligible else float("nan")

    seq_sims, exact = [], []
    for _, trip_group in [("trip", valid)]:
        pred_seq = _collapse(trip_group.hmm_arc_id.astype(int).to_list())
        ref_seq = _collapse(trip_group.reference_arc_id.astype(int).to_list())
        seq_sims.append(_edit_similarity(pred_seq, ref_seq))
        exact.append(float(trip_group.hmm_correct.all()) if len(trip_group) else float("nan"))

    miss_denom = int(valid.reference_arc_id.notna().sum())
    misses = int((~valid.has_reference_candidate.fillna(False)).sum())
    metrics = {"observations_total": int(len(per_obs)),
        "observations_within_50m": int(len(valid)),
        "unlabelled_over_50m": int((~per_obs.valid_for_weak_evaluation).sum()),
        "candidate_miss_rate": misses / miss_denom if miss_denom else float("nan"),
        "classifier_top1_accuracy": float(valid.classifier_correct.mean()) if len(valid) else float("nan"),
        "hmm_top1_accuracy": float(valid.hmm_correct.mean()) if len(valid) else float("nan"),
        "classifier_route_continuity": continuity("classifier_arc_id"),
        "hmm_route_continuity": continuity("hmm_arc_id"),
        "hmm_weak_sequence_edit_similarity": float(np.mean(seq_sims)) if seq_sims else float("nan"),
        "hmm_exact_trip_sequence_accuracy": float(np.mean(exact)) if exact else float("nan"),
        "decoder_restarts": int(decoded.decoder_restart.sum()),
        "mean_reference_route_distance_m": float(label_stats["mean_reference_route_distance_m"])}
    return metrics, per_obs
