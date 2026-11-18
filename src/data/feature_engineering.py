"""Feature extraction and reference-route label derivation for PathSnap."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .candidate_generation import _project, _point_segment_distance

FEATURE_COLUMNS = [
    "distance_to_road_m",
    "midpoint_distance_m",
    "bearing_difference_deg",
    "gps_speed_mps",
    "arc_length_m",
    "segment_fraction",
    "candidate_rank",
    "time_gap_s",
]


def add_temporal_features(candidates: pd.DataFrame, gps: pd.DataFrame) -> pd.DataFrame:
    """Attach time since previous GPS fix and clean infinite numeric values."""
    out = candidates.copy()
    times = gps.set_index("observation_id")["timestamp_s"]
    gaps = times.diff().rename("time_gap_s")
    out = out.join(gaps, on="observation_id")
    out["time_gap_s"] = out["time_gap_s"].fillna(0.0)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def derive_reference_labels(
    candidates: pd.DataFrame,
    gps: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    route: pd.DataFrame,
    *,
    label_distance_threshold_m: float = 50.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Label candidate equal to nearest supplied route arc for each GPS fix.

    This is a transparent weak-label alignment heuristic: each GPS observation
    is assigned to the geometrically nearest arc among the trip's ordered
    reference-route arc IDs. If that nearest route arc is farther than the
    threshold, the observation is marked unlabelled (-1). Candidate generation
    is independent of route labels, so candidate miss rate remains measurable.
    """
    if route is None or route.empty:
        raise ValueError("Reference route is required only for label generation")
    route_ids = route.arc_id.to_numpy(dtype=np.int64)
    route_ids = np.unique(route_ids)
    if len(route_ids) == 0:
        raise ValueError("Reference route contains no arc IDs")

    lon0 = float(gps.longitude.mean())
    lat0 = float(gps.latitude.mean())
    node_xy = _project(nodes.longitude.to_numpy(), nodes.latitude.to_numpy(), lon0, lat0)
    arc_starts = node_xy[arcs.from_node_id.to_numpy(dtype=np.int64)]
    arc_ends = node_xy[arcs.to_node_id.to_numpy(dtype=np.int64)]
    route_starts, route_ends = arc_starts[route_ids], arc_ends[route_ids]
    route_midpoints = (route_starts + route_ends) / 2.0
    tree = cKDTree(route_midpoints)
    gps_xy = _project(gps.longitude.to_numpy(), gps.latitude.to_numpy(), lon0, lat0)

    nearest_route_ids = np.empty(len(gps), dtype=np.int64)
    nearest_route_dist = np.empty(len(gps), dtype=float)
    # Route-only candidate pool is generally much smaller than the full graph.
    k = min(20, len(route_ids))
    _, near_idx = tree.query(gps_xy, k=k, workers=-1)
    if k == 1:
        near_idx = near_idx[:, None]
    for i in range(len(gps)):
        local_route_pos = np.asarray(near_idx[i], dtype=np.int64)
        local_ids = route_ids[local_route_pos]
        pts = np.repeat(gps_xy[i][None, :], len(local_ids), axis=0)
        d, _ = _point_segment_distance(pts, arc_starts[local_ids], arc_ends[local_ids])
        best = int(np.argmin(d))
        nearest_route_ids[i] = int(local_ids[best])
        nearest_route_dist[i] = float(d[best])

    labels = pd.DataFrame({
        "observation_id": gps.observation_id.to_numpy(dtype=np.int64),
        "reference_arc_id": nearest_route_ids,
        "reference_route_distance_m": nearest_route_dist,
    })
    out = candidates.merge(labels, on="observation_id", how="left", validate="many_to_one")
    out["label"] = (out.candidate_arc_id == out.reference_arc_id).astype(np.int8)
    out.loc[out.reference_route_distance_m > label_distance_threshold_m, "label"] = -1

    valid = labels.reference_route_distance_m <= label_distance_threshold_m
    candidate_sets = out.groupby("observation_id").candidate_arc_id.apply(set)
    misses = []
    for row in labels.itertuples(index=False):
        if row.reference_route_distance_m <= label_distance_threshold_m:
            misses.append(int(row.reference_arc_id not in candidate_sets.get(row.observation_id, set())))
    stats = {
        "observations": int(len(gps)),
        "labelled_observations": int(valid.sum()),
        "unlabelled_observations": int((~valid).sum()),
        "candidate_miss_rate": float(np.mean(misses)) if misses else float("nan"),
        "mean_reference_route_distance_m": float(nearest_route_dist.mean()),
        "label_distance_threshold_m": float(label_distance_threshold_m),
    }
    return out, stats


def extract_features(candidates: pd.DataFrame, gps: pd.DataFrame) -> pd.DataFrame:
    """Return candidate rows with temporal features and stable identifiers."""
    from .feature_engineering import add_temporal_features
    return add_temporal_features(candidates, gps)
