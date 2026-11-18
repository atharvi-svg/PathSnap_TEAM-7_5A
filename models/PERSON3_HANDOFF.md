# Person 2 to Person 3 handoff and integration status

## Model artifacts

- `best_model.pkl` — tuned HistGradientBoosting pipeline, including median imputation.
- `best_config.json` — parameters, feature contract, CV objective, random seed, and runtime versions.
- `predict.py` — `predict_candidates(candidate_rows, model=None)` returns one normalized probability per candidate.

## Inference contract

Provide a DataFrame with these identifiers: `partition`, `trip_id`, `observation_id`, `candidate_arc_id`; and these features: `distance_to_road_m`, `midpoint_distance_m`, `bearing_difference_deg`, `gps_speed_mps`, `arc_length_m`, `segment_fraction`, `candidate_rank`, `time_gap_s`.

The function returns the identifiers, `raw_positive_score`, `candidate_probability`, and `is_top_candidate`. Candidate probabilities sum to 1 per GPS observation. The function does not consume `reference_arc_id`, `reference_route_distance_m`, or `label` even if those columns exist on the input.

For evaluator-time input, Person 3 should create candidates from GPS and road-network data through `src.data.candidate_generation.generate_candidates`, add time-gap features with `src.data.feature_engineering.extract_features`, set `partition` and `trip_id`, then call the prediction function. Do not load route labels in the inference path. `results/candidates/student_test_predictions.csv` is provided only as a development output for the ten student-test trips.

## Integration status

Person 3's route reconstruction is now implemented in `../route_matching/`, and Person 4's API and UI are in `../backend/` and `../frontend/`. The shared inference boundary is route-independent: generate candidate rows from GPS and road-network data, add the documented features, and obtain candidate probabilities with `predict_candidates()`. The route reconstruction then uses those probabilities to produce ordered arc IDs. `backend.pipeline.predict_trip()` demonstrates the integrated path and keeps reference-route reads in the post-prediction metrics step.

Pooled student-test comparisons are recorded in `../results/route_matching/hmm_comparison.csv`. HMM route continuity is higher than classifier-only continuity in the pooled student-test results, while HMM top-1 is slightly lower. The detailed final report is `../reports/PathSnap_Final_Report.pdf`.
