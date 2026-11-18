# Person 2 model-development notes

## Experiment provenance

This implementation and its runs were prepared with Codex assistance at the team's direction. Review the code, configuration, and results before using them in a submission or presentation, and follow the instructor's disclosure requirements.

## Data and labels

The candidate table has 6,827,310 rows: 6,252,510 from 80 training trips and 574,800 from 10 student-test trips. It uses the Person 1 top-30 candidate generator. The label is a weak alignment: for each GPS fix, the geometrically nearest arc in the supplied reference route is positive if it is within 50 m; other candidates are negative. Fixes farther than 50 m are labelled -1. This is not a direct GPS-to-route correspondence.

Training excludes label -1 rows and observations with no positive arc in the generated candidate set. The latter are candidate misses and cannot supply a valid positive candidate for classifier fitting. Their rate is tracked separately. The student-test partition is used only for development metrics and candidate-probability demonstration; it is not used for fitting or Bayesian search. Evaluator-testing data is not read.

## Training and validation

The five compared models are logistic regression, random forest, extra trees, histogram gradient boosting, and Gaussian naive Bayes. The primary selection metric is observation-level top-1 accuracy because the downstream task chooses one road segment per GPS observation. Candidate-level accuracy, precision, recall, F1 at threshold 0.5, and log loss are also recorded.

Cross-validation is four-fold `GroupKFold` by `trip_id`, so observations from a trip never appear on both sides of a fold. To bound runtime, the workflow deterministically selects at most 100 evenly spaced GPS observations per training trip for CV and refitting. All 80 trip IDs participate. Bayesian search uses the same groups and observation sample. Its Gaussian-process surrogate uses a Matérn 5/2 kernel and expected improvement for ten trials, optimizing pooled fold top-1 accuracy. `optimization_results.csv` includes the untuned default baseline as trial 0.

The candidate table contains more than 6.8 million rows; the observation cap means candidate-level row metrics and training loss are based on a reproducible training sample, while student-test evaluation uses all student-test candidates. Student-test labels remain weak route-proximity labels, so reported accuracy should not be described as independently verified ground truth.

## Recorded results

| Model | Grouped CV top-1 | Student-test top-1 | Student-test F1 | Student-test log loss |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.7797 | 0.7862 | 0.4965 | 0.1524 |
| Random forest | 0.7840 | 0.7898 | 0.7469 | 0.0522 |
| Extra trees | 0.7829 | 0.7900 | 0.6218 | 0.1074 |
| Histogram gradient boosting | 0.7854 | 0.7903 | 0.6358 | 0.0979 |
| Gaussian naive Bayes | 0.7696 | 0.7528 | 0.4372 | 0.4446 |
| Histogram gradient boosting, Bayesian tuned | 0.7863 | 0.7907 | 0.6243 | 0.1006 |

The tuned model improved pooled grouped-CV top-1 by 0.0009 over its untuned default and student-test top-1 by 0.0005. The student-test F1 and log loss were worse after tuning. The top-1 gain is very small; do not present tuning as a broad improvement. Random forest had the strongest student-test F1 and log loss among the five untuned baselines, while histogram gradient boosting was selected by the predeclared grouped-CV top-1 criterion.

The student-test candidate miss rate was 1.94% among route-distance-valid observations. There were also 27 of 19,160 GPS fixes beyond the 50 m weak-label threshold. Candidate probabilities in the supplied prediction file sum to one within each observation; this normalization is a ranking interface and should not be interpreted as calibrated route certainty.

## Person 3 and Person 4 integration

The connected route pipeline is implemented under `route_matching/`: `road_graph.py` builds directed connectivity; `hmm.py` defines candidate-state emissions and transition scores; `viterbi.py` decodes candidate sequences; `route_reconstruction.py` composes inference; and the evaluation and visualization modules write route metrics and maps. The pooled student-test results are in `results/route_matching/hmm_comparison.csv`. Across 19,149 eligible adjacent transitions, HMM continuity was 0.9993 versus classifier-only continuity of 0.9662. HMM pooled top-1 was 0.7861 versus 0.7907 for the classifier, so route continuity improved while weak-label top-1 decreased slightly.

`backend/` exposes health, trip-list, and prediction endpoints, and `frontend/` provides the Streamlit evaluator interface. The selected model is loaded from `models/best_model.pkl`. The inference path loads a trip without its reference route; the route is loaded only after predictions to compute aggregate development metrics. Reference route IDs and geometry are not included in the API result. The integrated `backend.pipeline.predict_trip` was exercised on student-test trip `00000052`; it returned 262 observation results and a 93-arc predicted sequence with evaluation metrics.

## Limits

- The route-proximity labels can be wrong when GPS noise, parallel roads, or route geometry make the nearest arc ambiguous.
- Candidate generation can omit the reference arc; a classifier cannot recover an arc that is absent from its candidate list.
- CV and final fitting use up to 100 sampled observations per trip, not every training observation.
- The included archive has no evaluator partition. Run the configured API and UI against the evaluator files when they are provided; do not substitute student-test results for evaluator results.
- Folium is in `requirements.txt`; it was unavailable on the offline build host, so the included route image was generated with the offline visualization fallback.
