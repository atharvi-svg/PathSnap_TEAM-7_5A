"""Viterbi decoding over classifier-scored road-arc candidates."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .road_graph import RoadGraph


@dataclass
class DecodeResult:
    states: list[int]
    log_score: float
    connected_transitions: int
    transition_count: int
    route_arc_ids: list[int]

    @property
    def connectivity(self) -> float:
        return self.connected_transitions / self.transition_count if self.transition_count else 1.0


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6_371_008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _ordered_observations(candidates: pd.DataFrame) -> list[pd.DataFrame]:
    required = {"observation_id", "candidate_arc_id", "candidate_probability"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate predictions missing columns: {sorted(missing)}")
    if candidates.empty:
        raise ValueError("candidate predictions are empty")
    return [g.sort_values(["candidate_probability", "candidate_arc_id"], ascending=[False, True])
            for _, g in candidates.groupby("observation_id", sort=True)]


def decode_viterbi(
    candidates: pd.DataFrame,
    gps: pd.DataFrame,
    graph: RoadGraph,
    *,
    transition_scale_m: float = 45.0,
    disconnected_penalty: float = 9.0,
) -> DecodeResult:
    """Decode one most likely arc per observation with continuity-aware transitions.

    Emissions are Person 2's normalized candidate probabilities. A transition
    favors a directed network path whose length (including within-arc fractions)
    agrees with the GPS displacement. Unreachable candidate pairs receive a
    finite penalty so noisy fixes/candidate misses do not make decoding fail.
    """
    if transition_scale_m <= 0:
        raise ValueError("transition_scale_m must be positive")
    groups = _ordered_observations(candidates)
    gps_by_id = gps.set_index("observation_id")
    missing_obs = [int(g.observation_id.iloc[0]) for g in groups if int(g.observation_id.iloc[0]) not in gps_by_id.index]
    if missing_obs:
        raise ValueError(f"GPS rows are missing candidate observations: {missing_obs[:5]}")
    arc_u = graph.arcs.from_node_id.to_numpy(dtype=int)
    arc_v = graph.arcs.to_node_id.to_numpy(dtype=int)
    states_by_t = [g.candidate_arc_id.to_numpy(dtype=int) for g in groups]
    frac_by_t = [g.segment_fraction.to_numpy(dtype=float) if "segment_fraction" in g else np.full(len(g), 0.5) for g in groups]
    emit_by_t = [np.log(np.maximum(g.candidate_probability.to_numpy(dtype=float), 1e-12)) for g in groups]
    backpointers: list[np.ndarray] = [np.full(len(states_by_t[0]), -1, dtype=np.int32)]
    scores = emit_by_t[0].copy()
    for t in range(1, len(groups)):
        prev_states, curr_states = states_by_t[t - 1], states_by_t[t]
        prev_frac, curr_frac = frac_by_t[t - 1], frac_by_t[t]
        prev_obs_id = int(groups[t - 1].observation_id.iloc[0])
        curr_obs_id = int(groups[t].observation_id.iloc[0])
        p = gps_by_id.loc[prev_obs_id]
        q = gps_by_id.loc[curr_obs_id]
        observed = _haversine_m(float(p.longitude), float(p.latitude), float(q.longitude), float(q.latitude))
        dt = max(float(q.timestamp_s) - float(p.timestamp_s), 0.0) if "timestamp_s" in gps_by_id.columns else 1.0
        beta = max(float(transition_scale_m), observed * 0.35 + 20.0, min(dt, 600.0) * 12.0)
        prev_ends = arc_v[prev_states]
        curr_starts = arc_u[curr_states]
        # Bound graph search by plausible movement during this observation gap.
        # This prevents an unreachable pair from triggering a full-network walk.
        search_limit = min(5_000.0, max(500.0, observed * 2.0 + min(dt, 120.0) * 25.0))
        distances_by_source: dict[int, dict[int, float]] = {}
        for source in np.unique(prev_ends):
            distances_by_source[int(source)] = graph.shortest_paths(
                int(source), np.unique(curr_starts), max_distance_m=search_limit
            )[0]
        transition = np.empty((len(prev_states), len(curr_states)), dtype=float)
        connected = np.zeros_like(transition, dtype=bool)
        for i, (prev_arc, prev_end) in enumerate(zip(prev_states, prev_ends)):
            remaining = graph.lengths_m[prev_arc] * max(0.0, 1.0 - prev_frac[i])
            node_distances = distances_by_source[int(prev_end)]
            for j, (curr_arc, curr_start) in enumerate(zip(curr_states, curr_starts)):
                if prev_arc == curr_arc:
                    net_dist = graph.lengths_m[prev_arc] * max(0.0, curr_frac[j] - prev_frac[i])
                    reachable = True
                elif int(curr_start) in node_distances:
                    net_dist = remaining + node_distances[int(curr_start)] + graph.lengths_m[curr_arc] * min(max(curr_frac[j], 0.0), 1.0)
                    reachable = True
                else:
                    # Fallback retains a usable score while penalizing a broken
                    # directed path. Straight-line displacement is a lower bound.
                    net_dist = observed + transition_scale_m
                    reachable = False
                transition[i, j] = -abs(net_dist - observed) / beta
                if not reachable:
                    transition[i, j] -= disconnected_penalty
                connected[i, j] = reachable
        joint = scores[:, None] + transition
        back = np.argmax(joint, axis=0)
        scores = joint[back, np.arange(len(curr_states))] + emit_by_t[t]
        backpointers.append(back.astype(np.int32))
    state_idx = int(np.argmax(scores))
    path_indices = [state_idx]
    for t in range(len(groups) - 1, 0, -1):
        state_idx = int(backpointers[t][state_idx])
        path_indices.append(state_idx)
    path_indices.reverse()
    states = [int(states_by_t[t][idx]) for t, idx in enumerate(path_indices)]
    connected_total = 0
    transition_total = max(0, len(states) - 1)
    arc_u_lookup, arc_v_lookup = graph.arcs.from_node_id.to_numpy(dtype=int), graph.arcs.to_node_id.to_numpy(dtype=int)
    for prev_arc, curr_arc in zip(states[:-1], states[1:]):
        if prev_arc == curr_arc or graph.shortest_paths(int(arc_v_lookup[prev_arc]), [int(arc_u_lookup[curr_arc])], max_distance_m=5_000.0)[0].get(int(arc_u_lookup[curr_arc])) is not None:
            connected_total += 1
    route: list[int] = []
    for i, arc_id in enumerate(states):
        if i == 0:
            route.append(arc_id)
            continue
        prev_arc = states[i - 1]
        if prev_arc == arc_id:
            continue
        middle = graph.shortest_arc_path(int(arc_v[prev_arc]), int(arc_u[arc_id]), max_distance_m=5_000.0)
        if middle is not None:
            route.extend(middle)
        if not route or route[-1] != arc_id:
            route.append(arc_id)
    return DecodeResult(states, float(np.max(scores)), connected_total, transition_total, route)
