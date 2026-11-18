"""Create reproducible Person-1 EDA outputs for PathSnap.

Example:
python3 scripts/explore_dataset.py --data-root '/path/to/Student folder' \
  --output-dir results/eda --map-trip 00000001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import discover_trip_ids, load_trip  # noqa: E402


def trip_summary(trip, partition: str) -> dict[str, object]:
    gaps = trip.gps["timestamp_s"].diff().dropna()
    graph = nx.from_pandas_edgelist(
        trip.arcs, source="from_node_id", target="to_node_id", create_using=nx.DiGraph
    )
    weak_components = list(nx.weakly_connected_components(graph))
    return {
        "partition": partition,
        "trip_id": trip.trip_id,
        "gps_observations": len(trip.gps),
        "nodes": len(trip.nodes),
        "arcs": len(trip.arcs),
        "route_arcs": len(trip.route) if trip.route is not None else None,
        "missing_gps_values": int(trip.gps.isna().sum().sum()),
        "duplicate_gps_rows": int(trip.gps.duplicated().sum()),
        "non_one_second_gaps": int((gaps != 1).sum()),
        "largest_timestamp_gap_s": float(gaps.max()) if not gaps.empty else 0.0,
        "weak_components": len(weak_components),
        "isolated_nodes": int(sum(1 for node in trip.nodes.node_id if node not in graph)),
        "gps_lon_min": trip.gps.longitude.min(),
        "gps_lon_max": trip.gps.longitude.max(),
        "gps_lat_min": trip.gps.latitude.min(),
        "gps_lat_max": trip.gps.latitude.max(),
    }


def make_map(trip, destination: Path) -> None:
    center = [trip.gps.latitude.mean(), trip.gps.longitude.mean()]
    fmap = folium.Map(location=center, zoom_start=14, control_scale=True)
    gps_layer = folium.FeatureGroup(name="Raw GPS observations")
    folium.PolyLine(
        trip.gps[["latitude", "longitude"]].values.tolist(), color="#2563eb", weight=3, opacity=0.8
    ).add_to(gps_layer)
    gps_layer.add_to(fmap)

    node_lookup = trip.nodes.set_index("node_id")[["latitude", "longitude"]]
    # The supplied road graphs can contain hundreds of thousands of arcs.  The
    # report needs the road geometry relevant to this trajectory, not an
    # unresponsive full-city HTML file.  Keep arcs whose endpoint falls close
    # to the GPS bounding box; the reference route remains drawn in full.
    padding_degrees = 0.003
    nearby_nodes = trip.nodes.loc[
        trip.nodes.longitude.between(
            trip.gps.longitude.min() - padding_degrees,
            trip.gps.longitude.max() + padding_degrees,
        )
        & trip.nodes.latitude.between(
            trip.gps.latitude.min() - padding_degrees,
            trip.gps.latitude.max() + padding_degrees,
        ),
        "node_id",
    ]
    local_arcs = trip.arcs.loc[
        trip.arcs.from_node_id.isin(nearby_nodes) | trip.arcs.to_node_id.isin(nearby_nodes)
    ]
    # Keep the interactive artifact small enough to open during a viva.  The
    # exact reference-route geometry is always retained in its own layer.
    max_display_arcs = 12_000
    if len(local_arcs) > max_display_arcs:
        stride = max(1, len(local_arcs) // max_display_arcs)
        local_arcs = local_arcs.iloc[::stride]
    network_layer = folium.FeatureGroup(name="Road network")
    for arc in local_arcs.itertuples(index=False):
        endpoints = node_lookup.loc[[arc.from_node_id, arc.to_node_id]].values.tolist()
        folium.PolyLine(endpoints, color="#94a3b8", weight=1, opacity=0.35).add_to(network_layer)
    network_layer.add_to(fmap)

    if trip.route is not None:
        route_layer = folium.FeatureGroup(name="Reference route")
        for arc_id in trip.route.arc_id:
            arc = trip.arcs.iloc[int(arc_id)]
            endpoints = node_lookup.loc[[arc.from_node_id, arc.to_node_id]].values.tolist()
            folium.PolyLine(endpoints, color="#16a34a", weight=4, opacity=0.9).add_to(route_layer)
        route_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(destination)


def make_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    for column, title in [
        ("gps_observations", "GPS observations per trip"),
        ("nodes", "Road-network nodes per trip"),
        ("arcs", "Road-network arcs per trip"),
        ("largest_timestamp_gap_s", "Largest timestamp gap per trip"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4))
        summary.boxplot(column=column, by="partition", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Partition")
        ax.set_ylabel(column.replace("_", " "))
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(output_dir / f"{column}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/eda"))
    parser.add_argument("--map-trip", default="00000001")
    parser.add_argument("--map-partition", default="train", choices=["train", "student_test"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for partition in ("train", "student_test"):
        for trip_id in discover_trip_ids(args.data_root, partition):
            summaries.append(trip_summary(load_trip(args.data_root, partition, trip_id), partition))
    summary = pd.DataFrame(summaries).sort_values(["partition", "trip_id"])
    summary.to_csv(args.output_dir / "trip_inventory.csv", index=False)
    numeric_columns = summary.select_dtypes(include="number").columns
    summary.groupby("partition")[numeric_columns].agg(["count", "min", "median", "max"]).to_csv(
        args.output_dir / "partition_summary.csv"
    )
    make_plots(summary, args.output_dir)
    map_trip = load_trip(args.data_root, args.map_partition, args.map_trip)
    make_map(map_trip, args.output_dir / f"map_{args.map_partition}_{args.map_trip}.html")
    print(f"Wrote EDA outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
