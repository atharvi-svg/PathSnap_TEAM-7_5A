"""Compare five classifiers with trip-grouped cross-validation."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold

from models.common import (FEATURE_COLUMNS, MODEL_DIR, MODEL_NAMES, SEED, fit_model,
                           keep_trainable_observations, load_rows, make_model,
                           observation_candidate_miss_rate, probabilities, top1_accuracy)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    cap = 100
    all_rows = load_rows("train", max_observations_per_trip=cap)
    rows = keep_trainable_observations(all_rows)
    trip_ids = np.asarray(sorted(all_rows.trip_id.astype(str).unique()))
    splitter = GroupKFold(n_splits=4)
    fold_records, summary = [], []
    print(f"CV rows={len(rows):,}; train trips={len(trip_ids)}; "
          f"sample cap={cap} GPS observations/trip; seed={SEED}", flush=True)

    for name in MODEL_NAMES:
        start = time.perf_counter()
        oof_scores = np.full(len(all_rows), np.nan)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(trip_ids, groups=trip_ids), start=1):
            train_trip_ids, valid_trip_ids = set(trip_ids[train_idx]), set(trip_ids[valid_idx])
            fold_train = rows.loc[rows.trip_id.astype(str).isin(train_trip_ids)].reset_index(drop=True)
            fold_valid_all = all_rows.loc[all_rows.trip_id.astype(str).isin(valid_trip_ids)].reset_index(drop=True)
            fold_valid = keep_trainable_observations(fold_valid_all)
            model = make_model(name)
            fit_model(model, fold_train[FEATURE_COLUMNS], fold_train.label.astype(int))
            scores_all = probabilities(model, fold_valid_all[FEATURE_COLUMNS])
            oof_scores[all_rows.trip_id.astype(str).isin(valid_trip_ids).to_numpy()] = scores_all
            valid_mask = (fold_valid_all.label.isin([0, 1]) & fold_valid_all.groupby(
                ["partition", "trip_id", "observation_id"], sort=False).label.transform("max").eq(1)).to_numpy()
            scores = scores_all[valid_mask]
            valid_rows = fold_valid
            y_valid = valid_rows.label.astype(int)
            fold_records.append({"model": name, "fold": fold,
                "train_trips": int(fold_train.trip_id.nunique()),
                "validation_trips": int(fold_valid_all.trip_id.nunique()),
                "validation_rows": int(len(valid_rows)),
                "candidate_accuracy": accuracy_score(y_valid, scores >= 0.5),
                "precision": precision_score(y_valid, scores >= 0.5, zero_division=0),
                "recall": recall_score(y_valid, scores >= 0.5, zero_division=0),
                "f1": f1_score(y_valid, scores >= 0.5, zero_division=0),
                "macro_f1": f1_score(y_valid, scores >= 0.5, average="macro", zero_division=0),
                "log_loss": log_loss(y_valid, np.clip(scores, 1e-7, 1 - 1e-7), labels=[0, 1]),
                "observation_top1_accuracy": top1_accuracy(fold_valid_all, scores_all),
                "candidate_miss_rate": observation_candidate_miss_rate(fold_valid_all)})
            print(f"{name}: fold {fold}/4 complete", flush=True)

        elapsed = time.perf_counter() - start
        usable = np.isfinite(oof_scores)
        trainable_mask = (all_rows.label.isin([0, 1]) & all_rows.groupby(
            ["partition", "trip_id", "observation_id"], sort=False).label.transform("max").eq(1)).to_numpy()
        class_mask = usable & trainable_mask
        yy = all_rows.label.to_numpy()[class_mask].astype(int)
        pp = oof_scores[class_mask]
        summary.append({"model": name, "algorithm_family": {
            "LogisticRegression": "linear probabilistic", "RandomForest": "bagged trees",
            "ExtraTrees": "randomized trees", "HistGradientBoosting": "gradient boosting",
            "GaussianNB": "probabilistic generative"}[name],
            "cv_folds": 4, "grouping": "trip_id", "cv_candidate_accuracy": accuracy_score(yy, pp >= .5),
            "cv_precision": precision_score(yy, pp >= .5, zero_division=0),
            "cv_recall": recall_score(yy, pp >= .5, zero_division=0),
            "cv_f1": f1_score(yy, pp >= .5, zero_division=0),
            "cv_macro_f1": f1_score(yy, pp >= .5, average="macro", zero_division=0),
            "cv_log_loss": log_loss(yy, np.clip(pp, 1e-7, 1 - 1e-7), labels=[0, 1]),
            "cv_observation_top1_accuracy": top1_accuracy(all_rows.loc[usable].reset_index(drop=True), oof_scores[usable]),
            "cv_candidate_miss_rate": observation_candidate_miss_rate(all_rows),
            "training_seconds": elapsed, "sampled_train_observations": int(rows[["trip_id", "observation_id"]].drop_duplicates().shape[0]),
            "sampled_candidate_rows": int(len(rows)), "sampling_cap_per_trip": cap})
        print(f"{name}: OOF top-1={summary[-1]['cv_observation_top1_accuracy']:.4f} "
              f"time={elapsed:.1f}s", flush=True)

    pd.DataFrame(summary).to_csv(MODEL_DIR / "experiment_results.csv", index=False)
    pd.DataFrame(fold_records).to_csv(MODEL_DIR / "cv_fold_results.csv", index=False)
    selected = max(summary, key=lambda r: (r["cv_observation_top1_accuracy"], -r["cv_log_loss"]))
    (MODEL_DIR / "selected_model.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"Selected for Bayesian optimization by grouped OOF top-1: {selected['model']}")


if __name__ == "__main__":
    main()
