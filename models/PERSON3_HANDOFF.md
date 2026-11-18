# Handoff from Person 2 to Person 3

## Model artifacts

- `best_model.pkl` — tuned HistGradientBoosting pipeline, including median imputation.
- `best_config.json` — parameters, feature contract, CV objective, random seed, and runtime versions.
- `predict.py` — `predict_candidates(candidate_rows, model=None)` returns one normalized probability per candidate.

## Inference contract

Provide a DataFrame with these identifiers: `partition`, `trip_id`, `observation_id`, `candidate_arc_id`; and these features: `distance_to_road_m`, `midpoint_distance_m`, `bearing_difference_deg`, `gps_speed_mps`, `arc_length_m`, `segment_fraction`, `candidate_rank`, `time_gap_s`.

The function returns the identifiers, `raw_positive_score`, `candidate_probability`, and `is_top_candidate`. Candidate probabilities sum to 1 per GPS observation. The function does not consume `reference_arc_id`, `reference_route_distance_m`, or `label` even if those columns exist on the input.

For evaluator-time input, Person 3 should create candidates from GPS and road-network data through `src.data.candidate_generation.generate_candidates`, add time-gap features with `src.data.feature_engineering.extract_features`, set `partition` and `trip_id`, then call the prediction function. Do not load route labels in the inference path. `results/candidates/student_test_predictions.csv` is provided only as a development output for the ten student-test trips.

The candidate classifier does not implement HMM/Viterbi decoding. Route continuity, sequence decoding, API/UI integration, and evaluator-server handling are outside Person 2's deliverables.
