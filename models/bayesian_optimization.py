"""Small Gaussian-process Bayesian search over the CV-selected classifier."""
from __future__ import annotations

import json
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import GroupKFold

from models.common import (FEATURE_COLUMNS, MODEL_DIR, SEED, fit_model, keep_trainable_observations,
                           load_rows, make_model, probabilities, top1_accuracy)


def decode(name: str, x: np.ndarray) -> dict:
    if name == "HistGradientBoosting":
        return {"max_iter": int(round(70 + x[0] * 110)),
                "learning_rate": float(0.03 + x[1] * 0.17),
                "max_leaf_nodes": int(round(7 + x[2] * 24)),
                "l2_regularization": float(10 ** (-3 + x[3] * 2.7))}
    if name in {"RandomForest", "ExtraTrees"}:
        return {"n_estimators": 100,
                "max_depth": int(round(8 + x[0] * 20)),
                "min_samples_leaf": int(round(1 + x[1] * 7)),
                "max_features": "sqrt" if x[2] < .5 else .8}
    if name == "LogisticRegression":
        return {"C": float(10 ** (-2 + x[0] * 3))}
    return {"var_smoothing": float(10 ** (-12 + x[0] * 8))}


def objective(name: str, params: dict, rows: pd.DataFrame) -> tuple[float, float]:
    trainable = keep_trainable_observations(rows)
    trip_ids = np.asarray(sorted(rows.trip_id.astype(str).unique()))
    splitter = GroupKFold(n_splits=4)
    fold_tops, valid_frames, score_arrays = [], [], []
    for train_i, valid_i in splitter.split(trip_ids, groups=trip_ids):
        train_ids, valid_ids = set(trip_ids[train_i]), set(trip_ids[valid_i])
        fold_train = trainable.loc[trainable.trip_id.astype(str).isin(train_ids)]
        fold_valid = rows.loc[rows.trip_id.astype(str).isin(valid_ids)].reset_index(drop=True)
        model = make_model(name, params)
        fit_model(model, fold_train[FEATURE_COLUMNS], fold_train.label.astype(int))
        pred = probabilities(model, fold_valid[FEATURE_COLUMNS])
        fold_tops.append(top1_accuracy(fold_valid, pred))
        valid_frames.append(fold_valid)
        score_arrays.append(pred)
    pooled = top1_accuracy(pd.concat(valid_frames, ignore_index=True), np.concatenate(score_arrays))
    return float(pooled), float(np.nanstd(fold_tops))


def cross_validated_metrics(name: str, params: dict, rows: pd.DataFrame) -> dict:
    trainable = keep_trainable_observations(rows)
    trip_ids = np.asarray(sorted(rows.trip_id.astype(str).unique()))
    fold_rows, fold_scores = [], []
    for train_i, valid_i in GroupKFold(n_splits=4).split(trip_ids, groups=trip_ids):
        train_ids, valid_ids = set(trip_ids[train_i]), set(trip_ids[valid_i])
        fold_train = trainable.loc[trainable.trip_id.astype(str).isin(train_ids)]
        fold_valid = rows.loc[rows.trip_id.astype(str).isin(valid_ids)].reset_index(drop=True)
        model = make_model(name, params)
        fit_model(model, fold_train[FEATURE_COLUMNS], fold_train.label.astype(int))
        fold_rows.append(fold_valid)
        fold_scores.append(probabilities(model, fold_valid[FEATURE_COLUMNS]))
    scored = pd.concat(fold_rows, ignore_index=True)
    scores = np.concatenate(fold_scores)
    mask = (scored.label.isin([0, 1]) & scored.groupby(
        ["partition", "trip_id", "observation_id"], sort=False).label.transform("max").eq(1)).to_numpy()
    y, p = scored.loc[mask, "label"].astype(int), scores[mask]
    return {"accuracy": float(accuracy_score(y, p >= .5)),
        "precision": float(precision_score(y, p >= .5, zero_division=0)),
        "recall": float(recall_score(y, p >= .5, zero_division=0)),
        "f1": float(f1_score(y, p >= .5, zero_division=0)),
        "macro_f1": float(f1_score(y, p >= .5, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "observation_top1_accuracy": float(top1_accuracy(scored, scores))}


def main() -> None:
    selection = json.loads((MODEL_DIR / "selected_model.json").read_text(encoding="utf-8"))
    name = selection["model"]
    rows = load_rows("train", max_observations_per_trip=100)
    dimensions = 4 if name == "HistGradientBoosting" else (3 if name in {"RandomForest", "ExtraTrees"} else 1)
    rng = np.random.default_rng(SEED)
    trials: list[dict] = []
    X_init = rng.random((5, dimensions))
    started = time.perf_counter()
    baseline_score, baseline_std = objective(name, {}, rows)
    print(f"default config: grouped OOF top1={baseline_score:.4f}", flush=True)

    for i in range(10):
        if i < len(X_init):
            point = X_init[i]
            acquisition = float("nan")
        else:
            gp = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * Matern(length_scale=np.ones(dimensions), nu=2.5) + WhiteKernel(1e-5),
                normalize_y=True, random_state=SEED + i, n_restarts_optimizer=1)
            gp.fit(np.asarray([t["x"] for t in trials]), np.asarray([t["objective_top1"] for t in trials]))
            candidates = rng.random((256, dimensions))
            mean, std = gp.predict(candidates, return_std=True)
            best = max(t["objective_top1"] for t in trials)
            improvement = mean - best - 0.005
            z = improvement / np.maximum(std, 1e-9)
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            point = candidates[int(np.argmax(ei))]
            acquisition = float(np.max(ei))
        params = decode(name, point)
        t0 = time.perf_counter()
        score, fold_sd = objective(name, params, rows)
        trials.append({"trial": i + 1, "model": name, "objective_top1": score,
                       "fold_std": fold_sd, "acquisition_expected_improvement": acquisition,
                       "parameters_json": json.dumps(params), "x": point.tolist(),
                       "trial_seconds": time.perf_counter() - t0})
        print(f"trial {i+1}/10: top1={score:.4f} params={params}", flush=True)

    history = pd.DataFrame([{k: v for k, v in trial.items() if k != "x"} for trial in trials])
    baseline_record = {"trial": 0, "model": name, "objective_top1": baseline_score,
        "fold_std": baseline_std, "acquisition_expected_improvement": np.nan,
        "parameters_json": json.dumps({"configuration": "untuned default"}),
        "trial_seconds": np.nan, "record_type": "default_baseline"}
    history["record_type"] = "bayesian_search"
    history = pd.concat([pd.DataFrame([baseline_record]), history], ignore_index=True)
    history.to_csv(MODEL_DIR / "optimization_results.csv", index=False)
    best_trial = max(trials, key=lambda t: t["objective_top1"])
    best_params = json.loads(best_trial["parameters_json"])
    tuned_cv = cross_validated_metrics(name, best_params, rows)

    # Refit the selected configuration on the complete deterministic training
    # sample used for CV (up to 100 observations from each of the 80 trips).
    train_rows = keep_trainable_observations(load_rows("train", max_observations_per_trip=100))
    model = make_model(name, best_params)
    fit_model(model, train_rows[FEATURE_COLUMNS], train_rows.label.astype(int))
    import joblib
    joblib.dump(model, MODEL_DIR / "best_model.pkl", compress=3)

    config = {"selected_model": name, "best_parameters": best_params,
        "selection_basis": "highest mean observation-level top-1 accuracy in 4-fold GroupKFold by trip_id",
        "bayesian_objective": "pooled observation-level top-1 accuracy in 4-fold GroupKFold by trip_id",
        "search_method": "Gaussian process with Matérn 5/2 kernel and expected improvement",
        "trials": len(trials), "best_trial": best_trial["trial"],
        "best_cv_top1": best_trial["objective_top1"], "cv_fold_std": best_trial["fold_std"],
        "default_cv_top1": baseline_score, "default_cv_fold_std": baseline_std,
        "cv_top1_improvement": best_trial["objective_top1"] - baseline_score,
        "tuned_cv_metrics": tuned_cv,
        "sampled_observations_per_trip_during_optimization": 100,
        "sampled_observations_per_trip_for_final_fit": 100,
        "python_version": platform.python_version(), "scikit_learn_version": sklearn.__version__,
        "feature_columns": FEATURE_COLUMNS, "random_seed": SEED,
        "candidate_generation_top_k": 30, "weak_label_threshold_m": 50,
        "optimization_wall_seconds": time.perf_counter() - started,
        "ai_assistance_disclosure": "Code and experimental runs were prepared with Codex assistance; review and understand before academic use."}
    (MODEL_DIR / "best_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved tuned {name} model and config; best trial={best_trial['trial']}")


if __name__ == "__main__":
    main()
