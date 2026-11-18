"""Generate nearby road-arc candidates for each GPS observation.

Distances are computed in a local equirectangular metric projection centred on
this trip. Candidate search uses arc midpoints in a cKDTree, followed by exact
point-to-segment distance calculations in that local projection.

Reference routes are used ONLY by build_candidate_dataset.py to create labels
for permitted development partitions. This module's inference function does
not read route labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_RADIUS_M = 6_371_008.8


def _project(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float):
    lat0_rad = np.deg2rad(lat0)
    x = EARTH_RADIUS_M * np.deg2rad(lon - lon0) * np.cos(lat0_rad)
    y = EARTH_RADIUS_M * np.deg2rad(lat - lat0)
    return np.column_stack((x, y))


def _bearing_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Initial bearing from projected point a to b, clockwise from north."""
    dx = b[..., 0] - a[..., 0]
    dy = b[..., 1] - a[..., 1]
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def _angle_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def _point_segment_distance(points: np.ndarray, starts: np.ndarray, ends: np.ndarray):
    """Vectorised distance and projection fraction for paired points/segments."""
    vectors = ends - starts
    denom = np.einsum("ij,ij->i", vectors, vectors)
    denom = np.maximum(denom, 1e-12)
    t = np.einsum("ij,ij->i", points - starts, vectors) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = starts + vectors * t[:, None]
    distance = np.linalg.norm(points - closest, axis=1)
    return distance, t


def prepare_road_network(nodes: pd.DataFrame, arcs: pd.DataFrame, lon0: float, lat0: float):
    """Prepare projected arc geometry and midpoint index once per trip."""
    node_xy = _project(nodes.longitude.to_numpy(), nodes.latitude.to_numpy(), lon0, lat0)
    from_idx = arcs.from_node_id.to_numpy(dtype=np.int64)
    to_idx = arcs.to_node_id.to_numpy(dtype=np.int64)
    starts, ends = node_xy[from_idx], node_xy[to_idx]
    midpoints = (starts + ends) / 2.0
    lengths = np.linalg.norm(ends - starts, axis=1)
    bearings = _bearing_deg(starts, ends)
    tree = cKDTree(midpoints)
    return node_xy, starts, ends, midpoints, lengths, bearings, tree


def generate_candidates(
    gps: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    *,
    top_k: int = 30,
) -> pd.DataFrame:
    """Return up to top_k nearby arc candidates per GPS observation.

    Output preserves observation_id and candidate_arc_id. No route/label data
    is accepted by this function, making it suitable for inference as well.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if gps.empty or nodes.empty or arcs.empty:
        raise ValueError("gps, nodes and arcs must all be non-empty")

    lon0 = float(gps.longitude.mean())
    lat0 = float(gps.latitude.mean())
    _, starts, ends, _, lengths, arc_bearings, tree = prepare_road_network(
        nodes, arcs, lon0, lat0
    )
    gps_xy = _project(gps.longitude.to_numpy(), gps.latitude.to_numpy(), lon0, lat0)
    k = min(int(top_k), len(arcs))
    midpoint_dist, candidate_ids = tree.query(gps_xy, k=k, workers=-1)
    if k == 1:
        candidate_ids = candidate_ids[:, None]
        midpoint_dist = midpoint_dist[:, None]

    # GPS heading/speed from adjacent observations. Endpoints use their only
    # available neighbour; zero-time intervals are handled safely.
    n_obs = len(gps)
    gps_bearings = np.full(n_obs, np.nan, dtype=float)
    speeds = np.full(n_obs, np.nan, dtype=float)
    times = gps.timestamp_s.to_numpy(dtype=float)
    if n_obs > 1:
        delta_xy = np.diff(gps_xy, axis=0)
        delta_t = np.diff(times)
        step_bearings = _bearing_deg(gps_xy[:-1], gps_xy[1:])
        step_distances = np.linalg.norm(delta_xy, axis=1)
        step_speeds = np.divide(
            step_distances, delta_t, out=np.full_like(step_distances, np.nan), where=delta_t > 0
        )
        gps_bearings[:-1] = step_bearings
        gps_bearings[-1] = step_bearings[-1]
        speeds[:-1] = step_speeds
        speeds[-1] = step_speeds[-1]

    rows = []
    for obs_idx in range(n_obs):
        ids = np.asarray(candidate_ids[obs_idx], dtype=np.int64)
        point = np.repeat(gps_xy[obs_idx][None, :], len(ids), axis=0)
        dists, fractions = _point_segment_distance(point, starts[ids], ends[ids])
        # Sort by actual point-to-segment distance, not midpoint distance.
        order = np.argsort(dists)
        for rank in order:
            arc_id = int(ids[rank])
            bearing_diff = (
                float(_angle_difference(np.array([gps_bearings[obs_idx]]), np.array([arc_bearings[arc_id]]))[0])
                if np.isfinite(gps_bearings[obs_idx]) else np.nan
            )
            rows.append({
                "observation_id": int(gps.iloc[obs_idx].observation_id),
                "candidate_arc_id": arc_id,
                "candidate_rank": int(np.where(order == rank)[0][0]) + 1,
                "distance_to_road_m": float(dists[rank]),
                "midpoint_distance_m": float(midpoint_dist[obs_idx, rank]),
                "bearing_difference_deg": bearing_diff,
                "gps_bearing_deg": float(gps_bearings[obs_idx]) if np.isfinite(gps_bearings[obs_idx]) else np.nan,
                "arc_bearing_deg": float(arc_bearings[arc_id]),
                "gps_speed_mps": float(speeds[obs_idx]) if np.isfinite(speeds[obs_idx]) else np.nan,
                "arc_length_m": float(lengths[arc_id]),
                "segment_fraction": float(fractions[rank]),
                "timestamp_s": float(times[obs_idx]),
            })
    return pd.DataFrame(rows)
