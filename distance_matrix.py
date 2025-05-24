from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

class MatrixLocation(BaseModel):
    id: int
    name: Optional[str] = None
    lat: float
    lon: float

class DistanceMatrixRequest(BaseModel):
    locations: List[MatrixLocation]
    mode: Optional[str] = "driving"  # driving, walking, bicycling
    units: Optional[str] = "metric"  # metric, imperial

class DistanceMatrixResponse(BaseModel):
    distance_matrix: List[List[float]]
    duration_matrix: List[List[float]]
    status: str
    raw_response: Optional[dict] = None

@router.post("/distance-matrix", response_model=DistanceMatrixResponse)
def get_distance_matrix(request: DistanceMatrixRequest):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="GOOGLE_API_KEY no encontrada en .env")
    origins = [f"{loc.lat},{loc.lon}" for loc in request.locations]
    destinations = origins
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": "|".join(origins),
        "destinations": "|".join(destinations),
        "mode": request.mode or "driving",
        "units": request.units or "metric",
        "key": api_key
    }
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    if data.get("status") != "OK":
        raise HTTPException(status_code=400, detail=f"Google Distance Matrix error: {data.get('status')} - {data.get('error_message')}")
    # Procesar la matriz
    distance_matrix = []
    duration_matrix = []
    for row in data["rows"]:
        distance_row = [el["distance"]["value"]/1000 if el.get("distance") else None for el in row["elements"]]  # km
        duration_row = [el["duration"]["value"]/60 if el.get("duration") else None for el in row["elements"]]    # min
        distance_matrix.append(distance_row)
        duration_matrix.append(duration_row)
    return DistanceMatrixResponse(
        distance_matrix=distance_matrix,
        duration_matrix=duration_matrix,
        status=data["status"],
        raw_response=data
    )
