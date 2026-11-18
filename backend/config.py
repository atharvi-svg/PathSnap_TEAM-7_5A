from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("PATHSNAP_DATA_ROOT", str(ROOT / "Student folder"))).resolve()
MODEL_PATH = Path(os.getenv("PATHSNAP_MODEL_PATH", str(ROOT / "models" / "best_model.pkl"))).resolve()
TRIP_PARTITION = os.getenv("PATHSNAP_TRIP_PARTITION", "student_test")
HOST = os.getenv("PATHSNAP_API_HOST", "0.0.0.0")
PORT = int(os.getenv("PATHSNAP_API_PORT", "8000"))
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("PATHSNAP_ALLOWED_ORIGINS", "*").split(",") if x.strip()]
