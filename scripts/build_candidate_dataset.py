"""Build the permitted train/student-test candidate table incrementally."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import discover_trip_ids, load_trip  # noqa: E402
from src.data.candidate_generation import generate_candidates  # noqa: E402
from src.data.feature_engineering import derive_reference_labels, extract_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/candidates"))
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--label-distance-threshold-m", type=float, default=50.0)
    parser.add_argument("--partitions", nargs="+", choices=["train", "student_test"], default=["train", "student_test"])
    parser.add_argument("--limit-trips", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_path = args.output_dir / "candidate_features.csv"
    stats_path = args.output_dir / "candidate_generation_summary.csv"
    features_path.unlink(missing_ok=True)
    stats: list[dict] = []
    row_count = 0

    for partition in args.partitions:
        trip_ids = discover_trip_ids(args.data_root, partition)
        if args.limit_trips is not None:
            trip_ids = trip_ids[:args.limit_trips]
        for idx, trip_id in enumerate(trip_ids, start=1):
            trip = load_trip(args.data_root, partition, trip_id, include_route=True)
            candidates = generate_candidates(trip.gps, trip.nodes, trip.arcs, top_k=args.top_k)
            features = extract_features(candidates, trip.gps)
            labelled, trip_stats = derive_reference_labels(
                features, trip.gps, trip.nodes, trip.arcs, trip.route,
                label_distance_threshold_m=args.label_distance_threshold_m,
            )
            labelled.insert(0, "trip_id", trip_id)
            labelled.insert(0, "partition", partition)
            labelled.to_csv(features_path, mode="a", header=not features_path.exists(), index=False)
            row_count += len(labelled)
            stats.append({"partition": partition, "trip_id": trip_id,
                          "candidate_rows": int(len(labelled)),
                          "positive_candidate_rows": int(labelled.label.eq(1).sum()),
                          **trip_stats})
            pd.DataFrame(stats).to_csv(stats_path, index=False)
            print(f"[{partition} {idx}/{len(trip_ids)}] trip={trip_id} "
                  f"GPS={len(trip.gps)} candidates={len(labelled)} "
                  f"miss_rate={trip_stats['candidate_miss_rate']:.3f} "
                  f"total_rows={row_count}", flush=True)

    if not row_count:
        raise RuntimeError("No trips processed")
    (args.output_dir / "feature_columns.json").write_text(json.dumps({
        "identifiers": ["partition", "trip_id", "observation_id", "candidate_arc_id"],
        "features": ["distance_to_road_m", "midpoint_distance_m", "bearing_difference_deg",
                     "gps_speed_mps", "arc_length_m", "segment_fraction", "candidate_rank", "time_gap_s"],
        "target": "label",
        "label_definition": "1 if candidate arc equals nearest reference-route arc within threshold; "
                            "0 otherwise; -1 means observation outside threshold and should be excluded from supervised fitting.",
        "label_distance_threshold_m": args.label_distance_threshold_m,
        "top_k": args.top_k,
        "partitions": args.partitions,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {row_count:,} candidate rows to {features_path}")


if __name__ == "__main__":
    main()
