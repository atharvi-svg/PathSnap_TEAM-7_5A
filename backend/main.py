from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import router
from backend.config import ALLOWED_ORIGINS, PORT, TRIP_PARTITION

app = FastAPI(title="PathSnap Route Prediction API", version="1.0.0",
              description="Candidate scoring, route reconstruction, and server-side development metrics.")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["*"])
app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok", "partition": TRIP_PARTITION, "api_port": PORT}
