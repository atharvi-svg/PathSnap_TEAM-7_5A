# PathSnap Person 3: HMM route reconstruction

This package takes Person 1's GPS/road graph and Person 2's candidate classifier scores and returns an ordered road-arc path for Person 4 to call from the API pipeline.

## Contents

- `road_graph.py`: directed arc graph, metric lengths, and shortest-path helpers.
- `hmm.py`: emission/transition scoring and Viterbi decoder.
- `route_reconstruction.py`: public `predict_route(...)` interface and route metrics.
- `route_evaluation.py`: development-only comparison against route-proximity weak labels.
- `route_visualization.py`: Folium map export, with a self-contained SVG fallback.
- `visualizations/`: ten development route maps comparing predictions with supplied reference routes.
- `hmm_comparison.csv`: per-trip classifier-only versus HMM development evaluation.

## Integration contract

Put this folder at repository root beside Person 2's `models/` and Person 1's `src/data/`. Install `pip install -r route_matching/requirements.txt`. Person 2 serialized the model with scikit-learn 1.7.2 under Python 3.10.11, so keep that version when loading `models/best_model.pkl`.

```python
from src.data.loader import load_trip
from route_matching import predict_route

trip = load_trip("Student folder", "student_test", "00000000", include_route=False)
result = predict_route(trip.gps, trip.nodes, trip.arcs, trip_id=trip.trip_id)
print(result.arc_ids)  # ordered road arc IDs, including connecting arcs
print(result.metrics)
```

`predict_route` does not load or consume `.route` labels. Its result contains:

- `arc_ids`: ordered route arc IDs, with shortest directed connecting arcs inserted when reachable.
- `observation_arc_ids`: one decoded arc ID per GPS fix.
- `metrics`: observation/candidate counts, route length and arc count, continuity ratio, and Viterbi score.
- `candidates`: per-observation candidate features and normalized model probabilities.

Person 4 can use `result.arc_ids` and `result.metrics` directly in the API response. To render a map, call `save_route_map(gps, nodes, arcs, result.arc_ids, "route.html")`. Without Folium, the function writes a self-contained SVG visualization. The included HTML files are development-only comparisons that show the supplied reference route.

## Reproduce development evaluation

Run from the combined project repository root, where `src/`, `models/`, and `route_matching/` are importable:

```bash
python -m route_matching.route_evaluation_cli \
  --data-root "Student folder" \
  --partition student_test \
  --output route_matching/hmm_comparison.csv
```

The evaluation-only module reads reference routes after predictions and derives a weak label by selecting the nearest reference-route arc to each GPS fix, valid only within 50 m. This follows Person 2's documented label definition; metrics are development diagnostics, not independently verified ground truth. Evaluator inference must call `load_trip(..., include_route=False)` or pass raw frames directly.

The transition score rewards a directed network path with distance close to GPS displacement. The configurable default scale is 45 m; disconnected transitions remain possible with a strong penalty to handle noisy GPS fixes and candidate misses.

## Recorded student-test comparison

Using Person 2's provided development candidate probabilities, the macro mean across the ten student-test trips was 0.809 classifier-only top-1 accuracy versus 0.877 after HMM decoding, with 0.700 versus 0.876 ordered-route edit similarity. The decoded transitions were connected on 99.98% of consecutive GPS-fix pairs. Per-trip metrics are in `hmm_comparison.csv`. These are weak-label development results: the route reference is used only by `route_evaluation.py`, after predictions have been made.
