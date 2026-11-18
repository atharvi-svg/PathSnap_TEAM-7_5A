"""Inference interface: candidate rows in, per-observation probabilities out."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from models.common import FEATURE_COLUMNS, ID_COLUMNS, MODEL_DIR, probabilities


def predict_candidates(candidate_rows: pd.DataFrame, model=None) -> pd.DataFrame:
    """Return a normalized candidate probability for each observation.

    The function consumes only candidate features and IDs. Reference routes and
    labels, if present in the input frame, are never passed to the classifier.
    """
    if model is None:
        model = joblib.load(MODEL_DIR / "best_model.pkl")
    missing = set(ID_COLUMNS + FEATURE_COLUMNS) - set(candidate_rows.columns)
    if missing:
        raise ValueError(f"Missing candidate columns: {sorted(missing)}")
    raw = probabilities(model, candidate_rows[FEATURE_COLUMNS])
    result = candidate_rows[ID_COLUMNS].copy()
    result["raw_positive_score"] = raw
    keys = ["partition", "trip_id", "observation_id"]
    denom = result.groupby(keys, sort=False).raw_positive_score.transform("sum")
    count = result.groupby(keys, sort=False).candidate_arc_id.transform("size")
    result["candidate_probability"] = np.divide(raw, denom, out=1.0 / count.to_numpy(dtype=float), where=denom.to_numpy() > 0)
    winner = result.groupby(keys, sort=False).candidate_probability.transform("max")
    result["is_top_candidate"] = result.candidate_probability.eq(winner)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/candidates/candidate_features.csv")
    parser.add_argument("--output", default="results/candidates/student_test_predictions.csv")
    parser.add_argument("--partition", default="student_test", choices=["train", "student_test"])
    parser.add_argument("--model", default=str(MODEL_DIR / "best_model.pkl"))
    args = parser.parse_args()
    frames = []
    for chunk in pd.read_csv(args.input, usecols=ID_COLUMNS + FEATURE_COLUMNS,
                             dtype={"trip_id": str}, chunksize=200_000):
        selected = chunk.loc[chunk.partition.eq(args.partition)]
        if not selected.empty:
            frames.append(selected)
    if not frames:
        raise ValueError(f"No rows found for partition {args.partition!r} in {args.input}")
    rows = pd.concat(frames, ignore_index=True)
    model = joblib.load(args.model)
    predictions = predict_candidates(rows, model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print(f"Wrote {len(predictions):,} candidate probabilities to {output}")


if __name__ == "__main__":
    main()
