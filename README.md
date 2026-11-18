# PathSnap

## Person 1: Dataset exploration

Run the reproducible data-quality and EDA workflow:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/explore_dataset.py \
  --data-root '/Users/abhinav/Downloads/Student folder' \
  --output-dir results/eda \
  --map-trip 00000001
```

The shared loader contract is `src.data.load_trip(data_root, partition, trip_id)`.
It returns GPS, nodes, arcs, and optionally the reference route with explicit IDs.
