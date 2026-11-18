from fastapi import APIRouter, HTTPException

from backend.pipeline import available_trips, predict_trip
from backend.schemas import PredictRequest, PredictResponse, TripOption

router = APIRouter()


@router.get("/trips", response_model=list[TripOption])
def list_trips():
    try:
        return available_trips()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        return predict_trip(request.trip_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
