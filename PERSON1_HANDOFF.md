# Person 1 handoff: candidate generation + feature engineering

## Files added
- `src/data/candidate_generation.py`: spatial candidate search and arc-level geometric features.
- `src/data/feature_engineering.py`: temporal feature and training/development label derivation.
- `scripts/build_candidate_dataset.py`: builds candidate feature tables for `train` and `student_test` only.

## Run
```bash
python -m pip install -r requirements.txt
python scripts/build_candidate_dataset.py --data-root "PATH_TO_Student folder" --output-dir results/candidates --top-k 30
```

Quick smoke test:
```bash
python scripts/build_candidate_dataset.py --data-root "PATH_TO_Student folder" --output-dir results/candidates_smoke --top-k 20 --limit-trips 1
```

## Person 2: use the output
```python
import pandas as pd
from sklearn.impute import SimpleImputer

all_rows = pd.read_csv("results/candidates/candidate_features.csv")
train = all_rows[(all_rows.partition == "train") & (all_rows.label.isin([0, 1]))].copy()
student_dev = all_rows[(all_rows.partition == "student_test") & (all_rows.label.isin([0, 1]))].copy()

feature_cols = [
    "distance_to_road_m", "midpoint_distance_m", "bearing_difference_deg",
    "gps_speed_mps", "arc_length_m", "segment_fraction",
    "candidate_rank", "time_gap_s",
]
X_train, y_train = train[feature_cols], train["label"]
# Use trip_id as groups for group-aware cross-validation.
groups = train["trip_id"]
```

Do not fit on `student_dev`. It is for development evaluation/model comparison only.

## Label interpretation and limitation
The source data provides a GPS track and a chronological `.route` list, but does not provide a direct GPS-observation-to-route-position mapping. The script therefore derives a transparent weak label: for each GPS observation, find the geometrically closest arc among that trip's reference-route arcs; if it is within 50 m, the matching candidate gets label 1 and other candidates get label 0. If no route arc is within 50 m, the observation's candidates get label -1 and should be excluded from supervised training. This is an assumption that must be explained in the report, not presented as exact ground truth.

Candidate generation itself is independent of route labels. It queries the 30 nearest arc midpoints, then computes exact point-to-segment distances for those candidates. The candidate miss rate is recorded per trip; do not conceal it.

## Smoke-test performed
The code was compiled and run on one training trip (`00000001`) and one student-testing trip (`00000000`). It produced 76,360 candidate rows total (20 candidates per GPS observation). The measured candidate miss rates were 0.138 and 0.064 respectively for that smoke test only. These are not full-dataset results and must not be reported as final performance.
