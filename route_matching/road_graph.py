"""Directed road graph utilities for PathSnap route decoding."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class RoadGraph:
    """Directed graph whose arc IDs are the row indices in the supplied table."""
    arcs: pd.DataFrame
    lengths_m: np.ndarray
    outgoing: list[list[tuple[int, int, float]]]

    @classmethod
    def from_tables(cls, nodes: pd.DataFrame, arcs: pd.DataFrame) -> "RoadGraph":
        if nodes.empty or arcs.empty:
            raise ValueError("nodes and arcs must be non-empty")
        if not {"from_node_id", "to_node_id"}.issubset(arcs.columns):
            raise ValueError("arcs must contain from_node_id and to_node_id")
        lon = nodes.longitude.to_numpy(dtype=float)
        lat = nodes.latitude.to_numpy(dtype=float)
        u = arcs.from_node_id.to_numpy(dtype=np.int64)
        v = arcs.to_node_id.to_numpy(dtype=np.int64)
        if (u.min() < 0 or v.min() < 0 or u.max() >= len(nodes) or v.max() >= len(nodes)):
            raise ValueError("an arc references a node outside the nodes table")
        mean_lat = np.deg2rad(float(np.mean(lat)))
        x = 6_371_008.8 * np.deg2rad(lon) * np.cos(mean_lat)
        y = 6_371_008.8 * np.deg2rad(lat)
        lengths = np.hypot(x[v] - x[u], y[v] - y[u])
        outgoing: list[list[tuple[int, int, float]]] = [[] for _ in range(len(nodes))]
        for arc_id, (start, end, length) in enumerate(zip(u, v, lengths)):
            outgoing[int(start)].append((int(end), arc_id, float(length)))
        return cls(arcs.reset_index(drop=True), lengths, outgoing)

    def shortest_paths(self, source: int, targets: Iterable[int], max_distance_m: float | None = None) -> tuple[dict[int, float], dict[int, tuple[int, int]]]:
        """Dijkstra from source, stopping once all requested nodes are settled.

        Returns distances and predecessor (node, arc_id) records for settled nodes.
        """
        target_set = set(map(int, targets))
        if not target_set:
            return {}, {}
        n = len(self.outgoing)
        if source < 0 or source >= n:
            return {}, {}
        dist = {int(source): 0.0}
        prev: dict[int, tuple[int, int]] = {}
        queue = [(0.0, int(source))]
        remaining = target_set - {int(source)}
        while queue and remaining:
            cost, node = heapq.heappop(queue)
            if cost != dist.get(node):
                continue
            if max_distance_m is not None and cost > max_distance_m:
                break
            remaining.discard(node)
            for nxt, arc_id, weight in self.outgoing[node]:
                new_cost = cost + weight
                if new_cost < dist.get(nxt, float("inf")):
                    dist[nxt] = new_cost
                    prev[nxt] = (node, arc_id)
                    heapq.heappush(queue, (new_cost, nxt))
        return {t: dist[t] for t in target_set if t in dist}, prev

    def shortest_arc_path(self, source: int, target: int, max_distance_m: float | None = None) -> list[int] | None:
        if source == target:
            return []
        distances, prev = self.shortest_paths(source, [target], max_distance_m=max_distance_m)
        if target not in distances:
            return None
        path: list[int] = []
        node = target
        while node != source:
            parent, arc_id = prev[node]
            path.append(arc_id)
            node = parent
        path.reverse()
        return path
