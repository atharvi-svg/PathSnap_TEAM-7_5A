from __future__ import annotations

from functools import lru_cache
import heapq
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

EARTH_M = 6_371_008.8


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = np.radians([lat1, lat2])
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * EARTH_M * np.arcsin(np.sqrt(a)))


class RoadGraph:
    """Directed road graph with bounded cached shortest-path queries."""

    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame, *, search_limit_m: float = 5000.0):
        self.nodes = nodes.reset_index(drop=True)
        self.arcs = arcs.reset_index(drop=True)
        self.search_limit_m = float(search_limit_m)
        self.arc_from = self.arcs.from_node_id.to_numpy(dtype=np.int64)
        self.arc_to = self.arcs.to_node_id.to_numpy(dtype=np.int64)
        lon = self.nodes.longitude.to_numpy()
        lat = self.nodes.latitude.to_numpy()
        self.arc_lengths = np.array([
            haversine_m(lon[u], lat[u], lon[v], lat[v]) for u, v in zip(self.arc_from, self.arc_to)
        ], dtype=float)
        self.edge_arc_id: dict[tuple[int, int], int] = {}
        self.adjacency: dict[int, list[tuple[int, float, int]]] = {}
        weights: dict[tuple[int, int], float] = {}
        for arc_id, (u, v, length) in enumerate(zip(self.arc_from, self.arc_to, self.arc_lengths)):
            key = (int(u), int(v))
            if key not in weights or length < weights[key]:
                weights[key] = float(max(length, 1e-3))
                self.edge_arc_id[key] = int(arc_id)
        for (u, v), weight in weights.items():
            self.adjacency.setdefault(u, []).append((v, weight, self.edge_arc_id[(u, v)]))
        edges = list(weights)
        matrix = csr_matrix((np.array([weights[e] for e in edges]),
                             (np.array([e[0] for e in edges]), np.array([e[1] for e in edges]))),
                            shape=(len(self.nodes), len(self.nodes)))
        self.matrix = matrix

    @lru_cache(maxsize=128)
    def _source_tree(self, source: int):
        return dijkstra(self.matrix, directed=True, indices=int(source),
                        return_predecessors=True, limit=self.search_limit_m)

    def distance_from(self, source_node: int, target_node: int) -> float:
        distance, _ = self._source_tree(int(source_node))
        value = float(distance[int(target_node)])
        return value if np.isfinite(value) and value <= self.search_limit_m else float("inf")

    @lru_cache(maxsize=512)
    def _local_distances(self, source: int):
        """Small bounded Dijkstra neighborhood for repeated HMM transitions."""
        source = int(source)
        best = {source: 0.0}
        hops = {source: 0}
        parent: dict[int, tuple[int, int]] = {}
        queue = [(0.0, 0, source)]
        max_hops, max_distance = 8, 2000.0
        while queue:
            dist, depth, node = heapq.heappop(queue)
            if dist != best.get(node) or depth != hops.get(node) or depth >= max_hops:
                continue
            for nxt, weight, _ in self.adjacency.get(node, ()):
                candidate = dist + weight
                if candidate <= max_distance and candidate < best.get(nxt, float("inf")):
                    best[nxt] = candidate
                    hops[nxt] = depth + 1
                    parent[nxt] = (node, int(self.edge_arc_id[(node, nxt)]))
                    heapq.heappush(queue, (candidate, depth + 1, nxt))
        return best, parent

    def candidate_path_distance(self, prev_arc: int, prev_fraction: float,
                                next_arc: int, next_fraction: float) -> float:
        prev_arc, next_arc = int(prev_arc), int(next_arc)
        f0 = float(np.clip(prev_fraction, 0.0, 1.0))
        f1 = float(np.clip(next_fraction, 0.0, 1.0))
        if prev_arc == next_arc and f1 >= f0:
            return (f1 - f0) * self.arc_lengths[prev_arc]
        middle = self._local_distances(int(self.arc_to[prev_arc]))[0].get(int(self.arc_from[next_arc]), float("inf"))
        if not np.isfinite(middle):
            return float("inf")
        return ((1.0 - f0) * self.arc_lengths[prev_arc] + middle +
                f1 * self.arc_lengths[next_arc])

    def connector_arc_ids(self, source_node: int, target_node: int) -> list[int] | None:
        source_node, target_node = int(source_node), int(target_node)
        if source_node == target_node:
            return []
        distances, parent = self._local_distances(source_node)
        if target_node not in distances:
            return None
        path_arcs = []
        current = target_node
        while current != source_node:
            if current not in parent or len(path_arcs) > 8:
                return None
            prior, arc_id = parent[current]
            path_arcs.append(arc_id)
            current = prior
        path_arcs.reverse()
        return path_arcs

    def expand_arc_sequence(self, arc_ids: Iterable[int]) -> list[int]:
        """Insert road arcs along shortest connectors between observed states."""
        sequence = [int(x) for x in arc_ids]
        if not sequence:
            return []
        out = [sequence[0]]
        for nxt in sequence[1:]:
            prev = out[-1]
            if nxt == prev:
                continue
            connectors = self.connector_arc_ids(self.arc_to[prev], self.arc_from[nxt])
            if connectors is None:
                # Keep the observation-level state and expose the discontinuity
                # to route-continuity metrics instead of inventing a connector.
                if out[-1] != nxt:
                    out.append(nxt)
                continue
            for arc in connectors + [nxt]:
                if out[-1] != arc:
                    out.append(arc)
        return out

    def arc_coordinates(self, arc_ids: Iterable[int]) -> list[list[float]]:
        coords: list[list[float]] = []
        for arc_id in arc_ids:
            u, v = self.arc_from[int(arc_id)], self.arc_to[int(arc_id)]
            a = [float(self.nodes.longitude.iloc[u]), float(self.nodes.latitude.iloc[u])]
            b = [float(self.nodes.longitude.iloc[v]), float(self.nodes.latitude.iloc[v])]
            if not coords or coords[-1] != a:
                coords.append(a)
            coords.append(b)
        return coords
