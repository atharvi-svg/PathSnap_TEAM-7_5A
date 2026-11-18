"""Compare classifier-only and HMM routes on permitted student-test trips."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from route_matching.hmm import decode_trip
from route_matching.road_graph import RoadGraph
from route_matching.route_evaluation import evaluate_trip
from route_matching.route_visualization import create_route_map
from src.data import discover_trip_ids, load_trip


def main() -> None:
    data_root = ROOT / "Student folder"
    out_dir = ROOT / "results" / "route_matching"
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ["partition", "trip_id", "observation_id", "candidate_arc_id", "candidate_rank",
            "distance_to_road_m", "midpoint_distance_m", "bearing_difference_deg", "gps_speed_mps",
            "arc_length_m", "segment_fraction", "timestamp_s", "time_gap_s"]
    features = pd.read_csv(ROOT / "results/candidates/candidate_features.csv", usecols=keep,
        dtype={"trip_id": str}, chunksize=200_000)
    candidate_frames = []
    for chunk in features:
        part = chunk.loc[chunk.partition.eq("student_test"), keep]
        if not part.empty:
            candidate_frames.append(part)
    all_candidates = pd.concat(candidate_frames, ignore_index=True)
    predictions = pd.read_csv(ROOT / "results/candidates/student_test_predictions.csv", dtype={"trip_id": str})
    records, all_obs = [], []
    first_map = None

    for trip_id in discover_trip_ids(data_root, "student_test"):
        trip = load_trip(data_root, "student_test", trip_id, include_route=True)
        subset = all_candidates.loc[all_candidates.trip_id.eq(trip_id)].drop(columns="partition")
        pred = predictions.loc[predictions.trip_id.eq(trip_id),
            ["observation_id", "candidate_arc_id", "candidate_probability"]]
        scored = subset.merge(pred, on=["observation_id", "candidate_arc_id"], how="inner", validate="one_to_one")
        graph = RoadGraph(trip.nodes, trip.arcs)
        print(f"{trip_id}: decoding {len(trip.gps):,} observations", flush=True)
        decoded, stats = decode_trip(scored, trip.gps, graph)
        metrics, detail = evaluate_trip(trip.gps, trip.nodes, trip.arcs, trip.route, subset, decoded, graph)
        records.append({"partition": "student_test", "trip_id": trip_id, **metrics,
                        "expanded_route_arcs": len(stats["expanded_route_arc_ids"])})
        detail.insert(0, "trip_id", trip_id)
        detail.insert(0, "partition", "student_test")
        all_obs.append(detail)
        if trip_id == "00000000":
            first_map = create_route_map(trip.gps, graph, trip.route.arc_id.to_list(),
                stats["expanded_route_arc_ids"], out_dir / "student_test_00000000_map.html",
                title="PathSnap student-test trip 00000000")
            (out_dir / "student_test_00000000_route.json").write_text(json.dumps({
                "trip_id": trip_id, "reference_arc_ids": trip.route.arc_id.astype(int).to_list(),
                "classifier_arc_ids": stats["classifier_arc_ids"],
                "hmm_observation_arc_ids": stats["hmm_arc_ids"],
                "expanded_hmm_route_arc_ids": stats["expanded_route_arc_ids"]}, indent=2), encoding="utf-8")
        print(f"{trip_id}: top1 {metrics['classifier_top1_accuracy']:.4f} -> "
              f"{metrics['hmm_top1_accuracy']:.4f}; continuity "
              f"{metrics['classifier_route_continuity']:.4f} -> {metrics['hmm_route_continuity']:.4f}", flush=True)

    per_trip = pd.DataFrame(records)
    per_trip.to_csv(out_dir / "hmm_comparison_per_trip.csv", index=False)
    summary = {"partition": "student_test", "trip_id": "MEAN", "trips": len(per_trip)}
    for col in per_trip.select_dtypes(include="number").columns:
        if col != "expanded_route_arcs":
            summary[col] = float(per_trip[col].mean())
    summary["expanded_route_arcs"] = int(per_trip.expanded_route_arcs.sum())
    pd.concat([per_trip, pd.DataFrame([summary])], ignore_index=True).to_csv(out_dir / "hmm_comparison.csv", index=False)
    pd.concat(all_obs, ignore_index=True).to_csv(out_dir / "student_test_route_predictions.csv", index=False)
    print(f"Wrote route comparison and maps to {out_dir}; map={first_map}")


if __name__ == "__main__":
    main()
