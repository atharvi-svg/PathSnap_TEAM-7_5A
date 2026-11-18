# PathSnap - Person 2 handoff

Person 2's model-development work is included in this folder: the full candidate feature table, five-model comparison, trip-grouped validation, Bayesian-search history, selected model, student-test evaluation, and probability predictions.

## Key artifacts

- `results/candidates/candidate_features.csv` — 6,827,310 rows from 80 training and 10 student-test trips.
- `results/candidates/candidate_generation_summary.csv` — per-trip counts and candidate miss statistics.
- `models/experiment_results.csv` and `models/cv_fold_results.csv` — five baseline model results and fold records.
- `models/model_comparison.csv` and `models/model_comparison.png` — baseline and tuned comparison.
- `models/optimization_results.csv` — default baseline plus all ten Bayesian search trials.
- `models/best_model.pkl` and `models/best_config.json` — selected/refit HistGradientBoosting model and configuration.
- `models/student_test_metrics.csv` — evaluation on all ten student-test trips.
- `results/candidates/student_test_predictions.csv` — normalized candidate probabilities for all 574,800 candidate rows on the ten student-test trips.
- `models/predict.py` — callable `predict_candidates()` interface and CLI.
- `models/MODEL_NOTES.md` — measured results, label caveats, sampling, and metric definitions.

## Reproduce the workflow

From this folder:

```bash
python -m pip install -r requirements.txt
python scripts/build_candidate_dataset.py --data-root "Student folder" --output-dir results/candidates --top-k 30
python -m models.train_models
python -m models.bayesian_optimization
python -m models.evaluate_models
python -m models.model_comparison
python -m models.predict --partition student_test
```

The experiment code uses only the 80 training trips for fitting and grouped cross-validation. It uses all ten `student_test` trips for development evaluation and prediction. It does not read the evaluator-testing partition.

## Handoff to Person 3

Load `models/best_model.pkl` and call `models.predict.predict_candidates(candidate_rows, model)` with the eight feature columns and identifiers. Each output row retains `partition`, `trip_id`, `observation_id`, and `candidate_arc_id`; `candidate_probability` sums to one for each observation. The supplied prediction CSV is for development review. For evaluation-time inference, generate candidates from GPS and road-network data with the route-independent `generate_candidates(gps, nodes, arcs)` function, then add temporal features with `extract_features` before calling the prediction interface. Do not pass reference route labels to prediction.
