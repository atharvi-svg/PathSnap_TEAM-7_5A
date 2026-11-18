# PathSnap

## Runtime version

Use Python 3.10.x for the bundled `best_model.pkl` (the integration was verified with Python 3.10.11). Requirements pin scikit-learn 1.7.2 and SciPy 1.14.1 to match the model environment and support the route graph implementation.

Create a virtual environment before installing packages:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, use `py -3.10 -m venv .venv`, then `.venv\Scripts\Activate.ps1` before installing requirements.

## Person 1: Dataset exploration

Run the reproducible data-quality and EDA workflow:

```bash
python -m pip install -r requirements.txt
python3 scripts/explore_dataset.py \
  --data-root '/path/to/Student folder' \
  --output-dir results/eda \
  --map-trip 00000001
```

The shared loader contract is `src.data.load_trip(data_root, partition, trip_id)`.
It returns GPS, nodes, arcs, and optionally the reference route with explicit IDs.

## Person 1: candidate generation and feature engineering

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Build candidate rows for the 80 training and 10 student-testing trips:

```bash
python scripts/build_candidate_dataset.py \
  --data-root "/path/to/Student folder" \
  --output-dir results/candidates \
  --top-k 30
```

For a quick smoke test on one trip from each partition:

```bash
python scripts/build_candidate_dataset.py \
  --data-root "/path/to/Student folder" \
  --output-dir results/candidates_smoke \
  --top-k 30 --limit-trips 1
```

Main output: `results/candidates/candidate_features.csv`.

- Keep `partition`, `trip_id`, `observation_id`, and `candidate_arc_id` as identifiers.
- Person 2 should train only on `partition == "train"` and `label` in `{0, 1}`.
- `label == 1` marks the candidate arc that is geometrically closest to the supplied reference route for that GPS observation; `label == 0` marks other candidates. `label == -1` means the closest reference-route arc was more than 50 m away, so exclude those observations from supervised fitting and report them.
- Student-testing rows are for development-stage evaluation only. Do not use them to fit the model.
- This is a nearest-route-geometry alignment heuristic, not a supplied GPS-to-route correspondence. Document the 50 m threshold and candidate miss rate when reporting results.
- The inference function `generate_candidates(gps, nodes, arcs)` does not accept or read route labels.

## Person 2: model development

Generate the full train and student-test feature table, then run the reproducible model workflow:

```bash
python scripts/build_candidate_dataset.py --data-root "Student folder" --output-dir results/candidates --top-k 30
python -m models.train_models
python -m models.bayesian_optimization
python -m models.evaluate_models
python -m models.model_comparison
python -m models.predict --partition student_test
```

The first stage compares five classifiers with four-fold `GroupKFold` grouped by `trip_id`. To keep this run tractable, training observations are deterministically sampled at up to 100 GPS observations per trip. All 80 training trip groups participate. Bayesian search uses the same folds and sample. Evaluation and prediction use the complete student-test candidate table. The model fit excludes weak-label observations without an in-candidate positive. Metrics include candidate-level classification and observation-level top-1 segment accuracy; candidate miss rates are reported separately.

Artifacts are written under `models/` and candidate predictions under `results/candidates/`. See `models/MODEL_NOTES.md` for the metric definitions, sampling details, and generated results after the run.

## Person 3: route reconstruction

Evaluate the classifier-only route against HMM/Viterbi decoding on the permitted student-test partition:

```bash
python -m scripts.evaluate_routes
```

This writes per-trip and aggregate results to `results/route_matching/`, along with an interactive Folium map and a route visualization for trip `00000000`. HMM hidden states are candidate road arcs, emissions are log candidate probabilities, and transitions score the discrepancy between GPS displacement and directed network-path distance. A 12-state candidate beam bounds decoding cost. GPS gaps above 60 seconds and impossible transition sequences start a new decode segment. Reference routes are used only after candidate probabilities are produced, for development scoring.

## Person 4: FastAPI and Streamlit

The backend loads `models/best_model.pkl`, generates candidates from the selected trip, predicts candidate probabilities, and applies Viterbi decoding. It calculates weak-label metrics only for the permitted `student_test` development partition. For evaluator partitions it does not open ground-truth routes, and returns no ground-truth-based metrics. The Streamlit UI calls the API and displays the observed GPS trace and predicted route.

For a local development demo on one laptop, use two terminals from the project root. In terminal 1, run the API:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --access-log
```

```bash
export PATHSNAP_API_URL="http://127.0.0.1:8000"
python -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

In Windows PowerShell, set the variable with `$env:PATHSNAP_API_URL="http://127.0.0.1:8000"` instead of `export`.

Open `http://127.0.0.1:8501`. The supplied bundle has `train/` and `student_test/`, so its default partition is for development demonstration only.

## Two-laptop presentation

The final evaluation partition must be present only on Laptop 1, under the dataset root using the exact partition directory name supplied by the organizers. The provided zip does not include evaluator-test data, so that final-partition run cannot be verified from this bundle alone.

**Laptop 1 (API and model host):** Keep the dataset and model on this computer. Set the dataset root and the exact partition name supplied by the organizers, then start the API on all network interfaces. Keep this terminal open; Uvicorn prints API request logs here.

macOS/Linux (use port 8000, or choose another free port consistently on both laptops):

```bash
export PATHSNAP_DATA_ROOT="$PWD/Student folder"
export PATHSNAP_TRIP_PARTITION="evaluator_test"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --access-log
```

Windows PowerShell:

```powershell
$env:PATHSNAP_DATA_ROOT="C:\PathSnap\Student folder"
$env:PATHSNAP_TRIP_PARTITION="evaluator_test"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --access-log
```

Find Laptop 1's Wi-Fi IPv4 address (`ipconfig` on Windows; `ipconfig getifaddr en0` on macOS). Check `http://127.0.0.1:8000/health` on Laptop 1. If Laptop 2 cannot connect, allow incoming connections to the API port on Laptop 1's private network.

**Laptop 2 (evaluator UI):** Use the same project bundle, install requirements, and point the UI to Laptop 1's IPv4 address:

macOS/Linux:

```bash
export PATHSNAP_API_URL="http://192.168.1.25:8000"
python -m streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
```

Windows PowerShell:

```powershell
$env:PATHSNAP_API_URL="http://192.168.1.25:8000"
python -m streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
```

Replace `192.168.1.25` with Laptop 1's actual IPv4 address. Open `http://localhost:8501` on Laptop 2, select a trip, and choose **Predict route**. The UI sends only the trip ID; Laptop 1 loads the trip and returns the predicted route. Ground-truth-based metrics are disabled for evaluator partitions. The ten evaluator trip options come from the API, not from hardcoded IDs. Keep both laptops on the same trusted private network.

For a development demonstration without the evaluator data, use `PATHSNAP_TRIP_PARTITION=student_test` on Laptop 1. Do not describe that demonstration as final evaluator-partition performance.

## Final project report and handoff

The submission-ready report, including measured results, limitations, and two-laptop setup steps, is `reports/PathSnap_Final_Report.pdf`. `PERSON4_HANDOFF.md` summarizes the integration and presentation arrangement. The supplied archive does not include an evaluator partition, so only the included student-test development data has been evaluated.
