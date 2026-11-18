# Person 4 integration and presentation handoff

## Components

- `backend/main.py` creates the FastAPI app.
- `backend/api.py` exposes `/health`, `/api/v1/trips`, and `/api/v1/predict`.
- `backend/pipeline.py` loads the selected model, generates probabilities, reconstructs a route, and calculates aggregate development metrics after inference.
- `backend/schemas.py` defines request and response models; `backend/config.py` reads dataset, model, and partition settings from environment variables.
- `frontend/app.py` provides the Streamlit trip selector and evaluator display; `frontend/map_view.py` renders predicted GPS and road paths.
- `requirements.txt` and `README.md` contain installation and run steps.

## Data separation

The prediction pipeline opens trips with `include_route=False` before inference. Weak-label development metrics are calculated only when `PATHSNAP_TRIP_PARTITION=student_test`; for other partitions the pipeline never opens reference routes and returns no ground-truth-based metrics. The API response includes predicted arc IDs and coordinates, but no reference route IDs or coordinates. The default `student_test` is for local demonstration only.

## Two-laptop demo

Laptop 1 runs the API with `--host 0.0.0.0`, the dataset root, and the exact evaluator partition supplied by the organizers. Keep the API terminal visible for request logs; use `ipconfig` (Windows) or `ipconfig getifaddr en0` (macOS) to find Laptop 1's Wi-Fi IPv4 address. Laptop 2 sets `PATHSNAP_API_URL` to `http://<Laptop-1-IPv4>:<API-port>`, runs Streamlit on port 8501, and opens `http://localhost:8501`. Both laptops must use the same API port and trusted private network.

The provided archive has no evaluator partition. Use `student_test` for rehearsal and clearly identify it as development data. Replace it only after receiving the organizer's partition and exact folder name. Full commands are in `README.md` and `reports/PathSnap_Final_Report.pdf`.

## Integration check

Local verification on Python 3.10.11 and scikit-learn 1.7.2: `GET /health` returned 200, `GET /api/v1/trips` returned 10 trips, and `POST /api/v1/predict` for `00000052` returned 200 with 262 observations and 93 predicted arcs. The endpoints were exercised in-process with FastAPI's test client; rehearse browser-to-API networking on the presentation laptops before the demonstration.
