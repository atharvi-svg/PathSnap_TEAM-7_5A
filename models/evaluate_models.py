"""Fit the five baselines and report their held-out student-test metrics."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

from models.common import (FEATURE_COLUMNS, MODEL_DIR, MODEL_NAMES, fit_model,
                           keep_trainable_observations, load_rows, make_model,
                           observation_candidate_miss_rate, probabilities, top1_accuracy)


def metrics(name: str, rows: pd.DataFrame, score: np.ndarray) -> dict:
    trainable_mask = (rows.label.isin([0, 1]) & rows.groupby(
        ["partition", "trip_id", "observation_id"], sort=False).label.transform("max").eq(1)).to_numpy()
    trainable = rows.loc[trainable_mask]
    y = trainable.label.astype(int).to_numpy()
    p = score[trainable_mask]
    pred = p >= 0.5
    return {"model": name,
        "student_test_candidate_accuracy": accuracy_score(y, pred),
        "student_test_precision": precision_score(y, pred, zero_division=0),
        "student_test_recall": recall_score(y, pred, zero_division=0),
        "student_test_f1": f1_score(y, pred, zero_division=0),
        "student_test_log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "student_test_observation_top1_accuracy": top1_accuracy(rows, score),
        "student_test_candidate_miss_rate": observation_candidate_miss_rate(rows),
        "student_test_scored_observations": int(rows[["trip_id", "observation_id"]].drop_duplicates().shape[0]),
        "student_test_trainable_candidate_rows": int(len(trainable))}


def main() -> None:
    train_all = load_rows("train", max_observations_per_trip=100)
    train = keep_trainable_observations(train_all)
    test = load_rows("student_test")
    test = test.loc[test.label.isin([0, 1])].reset_index(drop=True)
    print(f"fit rows={len(train):,}; train trips={train.trip_id.nunique()}; "
          f"student-test rows={len(test):,}; test trips={test.trip_id.nunique()}", flush=True)
    results = []
    for name in MODEL_NAMES:
        start = time.perf_counter()
        model = make_model(name)
        fit_model(model, train[FEATURE_COLUMNS], train.label.astype(int))
        score = probabilities(model, test[FEATURE_COLUMNS])
        results.append({**metrics(name, test, score), "refit_seconds": time.perf_counter() - start})
        print(f"{name}: student-test top1={results[-1]['student_test_observation_top1_accuracy']:.4f}", flush=True)
    tuned_model = __import__("joblib").load(MODEL_DIR / "best_model.pkl")
    tuned_name = json.loads((MODEL_DIR / "best_config.json").read_text(encoding="utf-8"))["selected_model"] + " (Bayesian tuned)"
    start = time.perf_counter()
    score = probabilities(tuned_model, test[FEATURE_COLUMNS])
    results.append({**metrics(tuned_name, test, score), "refit_seconds": time.perf_counter() - start})
    pd.DataFrame(results).to_csv(MODEL_DIR / "student_test_metrics.csv", index=False)
    cv = pd.read_csv(MODEL_DIR / "experiment_results.csv")
    combined = cv.merge(pd.DataFrame(results), left_on="model", right_on="model", how="left")
    tuned_row = pd.DataFrame(results).tail(1).copy()
    config = json.loads((MODEL_DIR / "best_config.json").read_text(encoding="utf-8"))
    tuned_row["cv_observation_top1_accuracy"] = config["best_cv_top1"]
    tuned_row["training_seconds"] = np.nan
    tuned_row["cv_candidate_accuracy"] = config["tuned_cv_metrics"]["accuracy"]
    tuned_row["cv_precision"] = config["tuned_cv_metrics"]["precision"]
    tuned_row["cv_recall"] = config["tuned_cv_metrics"]["recall"]
    tuned_row["cv_f1"] = config["tuned_cv_metrics"]["f1"]
    tuned_row["cv_log_loss"] = config["tuned_cv_metrics"]["log_loss"]
    for col in cv.columns:
        if col not in tuned_row:
            tuned_row[col] = np.nan
    combined = pd.concat([combined, tuned_row[combined.columns]], ignore_index=True)
    combined.to_csv(MODEL_DIR / "model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
