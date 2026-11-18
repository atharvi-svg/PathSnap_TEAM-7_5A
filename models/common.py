from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "results" / "candidates" / "candidate_features.csv"
SUMMARY_CSV = ROOT / "results" / "candidates" / "candidate_generation_summary.csv"
MODEL_DIR = ROOT / "models"
FEATURE_COLUMNS = ["distance_to_road_m", "midpoint_distance_m", "bearing_difference_deg",
                   "gps_speed_mps", "arc_length_m", "segment_fraction", "candidate_rank", "time_gap_s"]
ID_COLUMNS = ["partition", "trip_id", "observation_id", "candidate_arc_id"]
USE_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + ["reference_arc_id", "reference_route_distance_m", "label"]
MODEL_NAMES = ["LogisticRegression", "RandomForest", "ExtraTrees", "HistGradientBoosting", "GaussianNB"]
SEED = 20261118


def load_rows(partition: str, max_observations_per_trip: int | None = None) -> pd.DataFrame:
    """Load a partition in chunks, optionally taking evenly spaced observations per trip."""
    summary = pd.read_csv(SUMMARY_CSV, dtype={"trip_id": str})
    counts = summary.loc[summary.partition.eq(partition)].set_index("trip_id")["observations"].to_dict()
    steps = {trip: max(1, int(np.ceil(int(count) / max_observations_per_trip)))
             for trip, count in counts.items()} if max_observations_per_trip else {}
    frames = []
    for chunk in pd.read_csv(FEATURES_CSV, usecols=USE_COLUMNS, dtype={"trip_id": str}, chunksize=200_000):
        chunk = chunk.loc[chunk.partition.eq(partition)]
        if chunk.empty:
            continue
        if max_observations_per_trip:
            keep = np.zeros(len(chunk), dtype=bool)
            for trip, indices in chunk.groupby("trip_id", sort=False).groups.items():
                step = steps.get(trip, 1)
                loc = chunk.index.get_indexer(indices)
                keep[loc] = (chunk.loc[indices, "observation_id"].to_numpy() % step) == 0
            chunk = chunk.loc[keep]
        frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=USE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def keep_trainable_observations(rows: pd.DataFrame) -> pd.DataFrame:
    """Exclude weak-label misses and observations whose positive arc was not a candidate."""
    rows = rows.loc[rows.label.isin([0, 1])].copy()
    has_positive = rows.groupby(["partition", "trip_id", "observation_id"], sort=False).label.transform("max").eq(1)
    return rows.loc[has_positive].reset_index(drop=True)


def make_model(name: str, params: dict | None = None):
    p = params or {}
    if name == "LogisticRegression":
        classifier = LogisticRegression(C=float(p.get("C", 1.0)), max_iter=250, class_weight="balanced",
                                        solver="lbfgs", random_state=SEED)
        scale = True
    elif name == "RandomForest":
        classifier = RandomForestClassifier(n_estimators=int(p.get("n_estimators", 100)),
            max_depth=p.get("max_depth", 16), min_samples_leaf=int(p.get("min_samples_leaf", 2)),
            max_features=p.get("max_features", "sqrt"), class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED)
        scale = False
    elif name == "ExtraTrees":
        classifier = ExtraTreesClassifier(n_estimators=int(p.get("n_estimators", 120)),
            max_depth=p.get("max_depth", 16), min_samples_leaf=int(p.get("min_samples_leaf", 2)),
            max_features=p.get("max_features", "sqrt"), class_weight="balanced",
            n_jobs=-1, random_state=SEED)
        scale = False
    elif name == "HistGradientBoosting":
        classifier = HistGradientBoostingClassifier(max_iter=int(p.get("max_iter", 120)),
            learning_rate=float(p.get("learning_rate", 0.08)), max_leaf_nodes=int(p.get("max_leaf_nodes", 15)),
            l2_regularization=float(p.get("l2_regularization", 1.0)), early_stopping=True,
            random_state=SEED)
        scale = False
    elif name == "GaussianNB":
        classifier = GaussianNB(var_smoothing=float(p.get("var_smoothing", 1e-9)), priors=[0.5, 0.5])
        scale = True
    else:
        raise ValueError(f"Unknown model: {name}")
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def fit_model(model, X: pd.DataFrame, y: pd.Series):
    """Fit HGB with balanced sample weights; the other estimators balance internally."""
    if isinstance(model.named_steps["classifier"], HistGradientBoostingClassifier):
        n_pos = max(1, int((y == 1).sum()))
        n_neg = max(1, int((y == 0).sum()))
        weights = np.where(y.to_numpy() == 1, len(y) / (2 * n_pos), len(y) / (2 * n_neg))
        model.fit(X, y, classifier__sample_weight=weights)
    else:
        model.fit(X, y)
    return model


def probabilities(model, X: pd.DataFrame) -> np.ndarray:
    cls = model.named_steps["classifier"].classes_
    values = model.predict_proba(X)
    pos_index = int(np.where(cls == 1)[0][0])
    return values[:, pos_index]


def top1_accuracy(rows: pd.DataFrame, scores: np.ndarray) -> float:
    work = rows[["partition", "trip_id", "observation_id", "candidate_arc_id", "reference_arc_id", "reference_route_distance_m"]].copy()
    work["score"] = scores
    valid = work.groupby(["partition", "trip_id", "observation_id"], sort=False).reference_route_distance_m.transform("min").le(50.0)
    work = work.loc[valid]
    if work.empty:
        return float("nan")
    idx = work.groupby(["partition", "trip_id", "observation_id"], sort=False).score.idxmax()
    winners = work.loc[idx]
    return float(winners.candidate_arc_id.eq(winners.reference_arc_id).mean())


def observation_candidate_miss_rate(rows: pd.DataFrame) -> float:
    """Fraction of label-valid observations whose reference arc is absent from candidates."""
    keys = ["partition", "trip_id", "observation_id"]
    grouped = rows.groupby(keys, sort=False)
    valid = grouped.reference_route_distance_m.min().le(50.0)
    has_ref = rows.assign(_has_ref=rows.candidate_arc_id.eq(rows.reference_arc_id)).groupby(keys, sort=False)._has_ref.any()
    denom = int(valid.sum())
    return float((valid & ~has_ref).sum() / denom) if denom else float("nan")


def config_write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

