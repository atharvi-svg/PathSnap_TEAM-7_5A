"""Development evaluation against supplied reference routes (weak labels)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _edit_similarity(predicted: list[int], reference: list[int]) -> float:
    """Normalized Levenshtein similarity for ordered, consecutive-deduped arcs."""
    a = [x for i, x in enumerate(predicted) if i == 0 or x != predicted[i - 1]]
    b = [x for i, x in enumerate(reference) if i == 0 or x != reference[i - 1]]
    if not a and not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        curr = [i]
        for j, y in enumerate(b, 1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = curr
    return 1.0 - prev[-1] / max(len(a), len(b), 1)


def evaluate_prediction(prediction, trip, graph) -> dict[str, float | int | str]:
    """Compare classifier and HMM per-fix choices to weak route proximity labels.

    This function must be called only in development evaluation, after inference.
    The supplied route is used to define the nearest route arc within 50 m.
    """
    if trip.route is None or trip.route.empty:
        raise ValueError("evaluation requires a non-empty reference route")
    from src.data.candidate_generation import _project, _point_segment_distance
    route_ids = trip.route.arc_id.to_numpy(dtype=int)
    route_ids = np.unique(route_ids)
    lat0 = float(trip.gps.latitude.mean())
    lon0 = float(trip.gps.longitude.mean())
    node_xy = _project(trip.nodes.longitude.to_numpy(), trip.nodes.latitude.to_numpy(), lon0, lat0)
    arc_u = trip.arcs.from_node_id.to_numpy(dtype=int)
    arc_v = trip.arcs.to_node_id.to_numpy(dtype=int)
    starts, ends = node_xy[arc_u[route_ids]], node_xy[arc_v[route_ids]]
    gps_xy = _project(trip.gps.longitude.to_numpy(), trip.gps.latitude.to_numpy(), lon0, lat0)
    references, valid = [], []
    for point in gps_xy:
        d, _ = _point_segment_distance(np.repeat(point[None, :], len(route_ids), axis=0), starts, ends)
        idx = int(np.argmin(d))
        references.append(int(route_ids[idx]))
        valid.append(float(d[idx]) <= 50.0)
    references = np.asarray(references, dtype=int)
    valid = np.asarray(valid, dtype=bool)
    scored = prediction.candidates
    best = (scored.sort_values(["observation_id", "candidate_probability", "candidate_arc_id"],
                               ascending=[True, False, True])
            .drop_duplicates("observation_id").set_index("observation_id"))
    pred_cls = best.loc[trip.gps.observation_id, "candidate_arc_id"].to_numpy(dtype=int)
    pred_hmm = np.asarray(prediction.observation_arc_ids, dtype=int)
    if len(pred_hmm) != len(trip.gps):
        raise ValueError("decoded observations do not align with GPS fixes")
    n_valid = int(valid.sum())
    route_reference = trip.route.arc_id.to_numpy(dtype=int).tolist()
    return {
        "trip_id": str(trip.trip_id),
        "observations": int(len(trip.gps)),
        "valid_observations_50m": n_valid,
        "weak_label_coverage": float(valid.mean()) if len(valid) else float("nan"),
        "classifier_top1_accuracy": float(np.mean(pred_cls[valid] == references[valid])) if n_valid else float("nan"),
        "hmm_top1_accuracy": float(np.mean(pred_hmm[valid] == references[valid])) if n_valid else float("nan"),
        "classifier_route_edit_similarity": _edit_similarity(pred_cls.tolist(), route_reference),
        "hmm_route_edit_similarity": _edit_similarity(prediction.arc_ids, route_reference),
        "route_arc_count": int(len(prediction.arc_ids)),
        "route_length_m": float(sum(graph.lengths_m[a] for a in prediction.arc_ids)),
        "connected_transition_ratio": float(prediction.metrics["connected_transition_ratio"]),
    }


def evaluate_partition(data_root: str | Path, partition: str, output_csv: str | Path, *, max_trips: int | None = None) -> pd.DataFrame:
    """Evaluate all trips in a partition and save per-trip comparison metrics."""
    from src.data.loader import discover_trip_ids, load_trip
    from .route_reconstruction import predict_route
    rows = []
    trip_ids = discover_trip_ids(data_root, partition)
    if max_trips is not None:
        trip_ids = trip_ids[:max_trips]
    for trip_id in trip_ids:
        trip = load_trip(data_root, partition, trip_id, include_route=True)
        prediction = predict_route(trip.gps, trip.nodes, trip.arcs, trip_id=trip.trip_id, partition=partition)
        from .road_graph import RoadGraph
        graph = RoadGraph.from_tables(trip.nodes, trip.arcs)
        rows.append(evaluate_prediction(prediction, trip, graph))
        print(f"Evaluated {partition}/{trip_id}: {rows[-1]['classifier_top1_accuracy']:.3f} -> {rows[-1]['hmm_top1_accuracy']:.3f}")
    result = pd.DataFrame(rows)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
