import os
import pytest
from fastapi.testclient import TestClient
from main import app

# Mock GOOGLE_API_KEY for the endpoint
def setup_module(module):
    os.environ["GOOGLE_API_KEY"] = "FAKE_KEY"

client = TestClient(app)

def make_payload(num_locations=2, num_vehicles=1, strict_mode=False, **kwargs):
    locations = [
        {
            "id": i,
            "lat": 1.0 + i,
            "lon": 2.0 + i,
            "name": f"Loc{i}",
            "demand": 0,
            "weight": 0.0,
            "volume": 0.0,
            "service_time": 5,
            "required_skills": []
        } for i in range(num_locations)
    ]
    vehicles = [
        {
            "id": v,
            "start_lat": 1.0,
            "start_lon": 2.0,
            "end_lat": 1.0,
            "end_lon": 2.0,
            "provided_skills": [],
            "capacity_weight": 100.0,
            "capacity_volume": 100.0,
            "capacity_quantity": 100,
            "start_time": 420,
            "end_time": 1080,
            "use_quantity": True,
            "use_weight": True,
            "use_volume": True
        } for v in range(num_vehicles)
    ]
    payload = {
        "locations": locations,
        "vehicles": vehicles,
        "num_vehicles": num_vehicles,
        "depot": 0,
        "strict_mode": strict_mode,
        "buffer_minutes": 10,
        "peak_hours": None,
        "peak_buffer_minutes": 20,
        "mode": "driving",
        "units": "metric",
        "include_polylines": True,
        "detail_level": "full"
    }
    payload.update(kwargs)
    return payload

def test_vrp_v2_ok():
    # Forzar el fallback eliminando la API KEY antes de la prueba
    if "GOOGLE_API_KEY" in os.environ:
        del os.environ["GOOGLE_API_KEY"]
    payload = make_payload()
    response = client.post("/vrp-v2", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "solution" in data
    assert "metadata" in data
    # Permitir cualquier warning legítimo; imprimirlos si existen
    if data["warnings"]:
        print("VRP v2 warnings:", data["warnings"])
        assert all(isinstance(w, dict) and "code" in w for w in data["warnings"])
    assert data["diagnostics"] in (None, {})

def test_vrp_v2_too_many_locations():
    from vrp_utils import VRPConstants
    payload = make_payload(num_locations=VRPConstants.MAX_LOCATIONS + 1)
    response = client.post("/vrp-v2", json=payload)
    # Ahora debe ser 422 porque Pydantic valida el límite
    assert response.status_code == 422

def test_vrp_v2_invalid_depot():
    payload = make_payload()
    payload["depot"] = 99
    response = client.post("/vrp-v2", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert any(w["code"] == "INVALID_DEPOT_INDEX" for w in data["warnings"])

def test_vrp_v2_only_depot():
    payload = make_payload(num_locations=1)
    response = client.post("/vrp-v2", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert any(w["code"] == "ONLY_DEPOT" for w in data["warnings"])
    assert data["diagnostics"]["routes"] == []
    assert data["diagnostics"]["total_distance"] == 0

def test_vrp_v2_skills_not_covered():
    payload = make_payload(strict_mode=True)
    payload["locations"][1]["required_skills"] = ["foo"]
    response = client.post("/vrp-v2", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert any(w["code"] == "SKILLS_NOT_COVERED" for w in data["warnings"])

def test_vrp_v2_capacity_not_covered():
    payload = make_payload(strict_mode=True)
    payload["locations"][1]["weight"] = 1000
    response = client.post("/vrp-v2", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert any(w["code"] == "CAPACITY_NOT_COVERED" for w in data["warnings"])
