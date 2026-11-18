"""Recompute macro and observation-weighted HMM summaries from saved predictions."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_trip


def main() -> None:
    folder = ROOT / "results" / "route_matching"
    rows = pd.read_csv(folder / "student_test_route_predictions.csv", dtype={"trip_id": str})
    prior = pd.read_csv(folder / "hmm_comparison_per_trip.csv", dtype={"trip_id": str})
    per_trip = []
    for trip_id, group in rows.groupby("trip_id", sort=True):
        group = group.sort_values("observation_id")
        valid = group.valid_for_weak_evaluation.astype(bool)
        valid_rows = group.loc[valid]
        trip = load_trip(ROOT / "Student folder", "student_test", trip_id, include_route=False)
        dt = trip.gps.sort_values("observation_id").timestamp_s.diff().to_numpy()[1:]
        eligible = int(((dt > 0) & (dt <= 60)).sum())
        p = prior.loc[prior.trip_id.eq(trip_id)].iloc[0]
        c_correct = int(valid_rows.classifier_arc_id.eq(valid_rows.reference_arc_id).sum())
        h_correct = int(valid_rows.hmm_arc_id.eq(valid_rows.reference_arc_id).sum())
        misses = int((~valid_rows.has_reference_candidate.astype(bool)).sum())
        c_connected = int(round(float(p.classifier_route_continuity) * eligible))
        h_connected = int(round(float(p.hmm_route_continuity) * eligible))
        per_trip.append({**p.to_dict(), "classifier_correct_observations": c_correct,
            "hmm_correct_observations": h_correct, "candidate_miss_observations": misses,
            "eligible_adjacent_transitions": eligible,
            "classifier_connected_transitions": c_connected,
            "hmm_connected_transitions": h_connected})
    detail = pd.DataFrame(per_trip)
    detail.to_csv(folder / "hmm_comparison_per_trip.csv", index=False)

    total_valid = int(detail.observations_within_50m.sum())
    total_obs = int(detail.observations_total.sum())
    total_adjacent = int(detail.eligible_adjacent_transitions.sum())
    pooled = {"partition": "student_test", "trip_id": "POOLED",
        "observations_total": total_obs, "observations_within_50m": total_valid,
        "unlabelled_over_50m": int(detail.unlabelled_over_50m.sum()),
        "candidate_miss_rate": float(detail.candidate_miss_observations.sum() / max(1, total_valid)),
        "classifier_top1_accuracy": float(detail.classifier_correct_observations.sum() / max(1, total_valid)),
        "hmm_top1_accuracy": float(detail.hmm_correct_observations.sum() / max(1, total_valid)),
        "classifier_route_continuity": float(detail.classifier_connected_transitions.sum() / max(1, total_adjacent)),
        "hmm_route_continuity": float(detail.hmm_connected_transitions.sum() / max(1, total_adjacent)),
        "hmm_weak_sequence_edit_similarity": float(detail.hmm_weak_sequence_edit_similarity.mean()),
        "hmm_exact_trip_sequence_accuracy": float(detail.hmm_exact_trip_sequence_accuracy.mean()),
        "decoder_restarts": int(detail.decoder_restarts.sum()),
        "mean_reference_route_distance_m": float(np.average(detail.mean_reference_route_distance_m,
            weights=np.maximum(detail.observations_total, 1))),
        "expanded_route_arcs": int(detail.expanded_route_arcs.sum()),
        "trips": int(len(detail)), "eligible_adjacent_transitions": total_adjacent}
    mean = {"partition": "student_test", "trip_id": "MEAN", "trips": int(len(detail))}
    numeric = [c for c in detail.select_dtypes(include="number").columns
               if c not in {"eligible_adjacent_transitions", "classifier_connected_transitions",
                            "hmm_connected_transitions", "classifier_correct_observations",
                            "hmm_correct_observations", "candidate_miss_observations"}]
    mean.update({c: float(detail[c].mean()) for c in numeric})
    pd.concat([detail, pd.DataFrame([mean, pooled])], ignore_index=True).to_csv(folder / "hmm_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    x = np.arange(len(detail))
    width = .36
    ids = detail.trip_id.astype(str)
    axes[0].bar(x - width/2, detail.classifier_top1_accuracy, width, label="Classifier only", color="#2563eb")
    axes[0].bar(x + width/2, detail.hmm_top1_accuracy, width, label="Classifier + HMM", color="#16a34a")
    axes[0].set_title("Weak-label observation top-1 accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[1].bar(x - width/2, detail.classifier_route_continuity, width, label="Classifier only", color="#2563eb")
    axes[1].bar(x + width/2, detail.hmm_route_continuity, width, label="Classifier + HMM", color="#16a34a")
    axes[1].set_title("Directed network route continuity")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Connected adjacent predictions")
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(x, ids, rotation=30, ha="right")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Classifier-only and HMM/Viterbi route evaluation")
    fig.savefig(folder / "hmm_comparison.png", dpi=180)
    plt.close(fig)
    print(pd.DataFrame([pooled]).to_string(index=False))


if __name__ == "__main__":
    main()
