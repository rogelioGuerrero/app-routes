import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Caso exitoso: datos mínimos válidos
def test_vrp_v1_ok():
    payload = {
        "locations": [
            {
                "id": 0,
                "name": "Depósito",
                "lat": 13.7000,
                "lon": -89.2000,
                "demand": 0,
                "weight": 0,
                "volume": 0,
                "time_window": [420, 1080],
                "service_time": 5,
                "required_skills": []
            },
            {
                "id": 1,
                "name": "Cliente 1",
                "lat": 13.7100,
                "lon": -89.2100,
                "demand": 1,
                "weight": 5,
                "volume": 1,
                "time_window": [500, 1000],
                "service_time": 5,
                "required_skills": []
            }
        ],
        "vehicles": [
            {
                "id": 0,
                "start_lat": 13.7000,
                "start_lon": -89.2000,
                "end_lat": 13.7000,
                "end_lon": -89.2000,
                "provided_skills": [],
                "capacity_weight": 100,
                "capacity_volume": 10,
                "capacity_quantity": 10,
                "start_time": 420,
                "end_time": 1080,
                "use_quantity": False,
                "use_weight": False,
                "use_volume": False
            }
        ],
        "num_vehicles": 1,
        "depot": 0,
        "strict_mode": False,
        "buffer_minutes": 10,
        "peak_hours": [["07:00", "09:00"]],
        "peak_buffer_minutes": 20,
        "mode": "driving",
        "units": "metric",
        "include_polylines": True,
        "detail_level": "full"
    }
    response = client.post("/vrp-v1", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert "solution" in data
    assert "routes" in data["solution"] or "details" in data["solution"]

# Caso error: ventana de tiempo imposible
def test_vrp_v1_invalid_time_window():
    payload = {
        "locations": [
            {
                "id": 0,
                "name": "Depósito",
                "lat": 13.7000,
                "lon": -89.2000,
                "demand": 0,
                "weight": 0,
                "volume": 0,
                "time_window": [420, 1080],
                "service_time": 5,
                "required_skills": []
            },
            {
                "id": 1,
                "name": "Cliente 1",
                "lat": 13.7100,
                "lon": -89.2100,
                "demand": 1,
                "weight": 5,
                "volume": 1,
                "time_window": [1000, 900],  # Fin antes que inicio
                "service_time": 5,
                "required_skills": []
            }
        ],
        "vehicles": [
            {
                "id": 0,
                "start_lat": 13.7000,
                "start_lon": -89.2000,
                "end_lat": 13.7000,
                "end_lon": -89.2000,
                "provided_skills": [],
                "capacity_weight": 100,
                "capacity_volume": 10,
                "capacity_quantity": 10,
                "start_time": 420,
                "end_time": 1080,
                "use_quantity": False,
                "use_weight": False,
                "use_volume": False
            }
        ],
        "num_vehicles": 1,
        "depot": 0,
        "strict_mode": False,
        "buffer_minutes": 10,
        "peak_hours": [["07:00", "09:00"]],
        "peak_buffer_minutes": 20,
        "mode": "driving",
        "units": "metric",
        "include_polylines": True,
        "detail_level": "full"
    }
    response = client.post("/vrp-v1", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "ventana de tiempo inválida" in data["detail"].lower() or "no se pudo encontrar una solución factible" in data["detail"].lower()
