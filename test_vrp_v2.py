import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Caso exitoso: datos mínimos válidos para vrp_v2

def test_vrp_v2_ok():
    payload = {
        "locations": [
            {
                "id": 0,
                "client_uuid": "loc-0",
                "name": "Depósito",
                "lat": 13.6989,
                "lon": -89.1914,
                "demand": 0,
                "weight": 0,
                "volume": 0,
                "time_window": [400, 1800],
                "service_time": 5,
                "required_skills": []
            },
            {
                "id": 1,
                "client_uuid": "loc-1",
                "name": "Cliente A",
                "lat": 13.6731,
                "lon": -89.2797,
                "demand": 1,
                "weight": 10,
                "volume": 0.5,
                "time_window": [400, 800],
                "service_time": 10,
                "required_skills": ["refrigerado"]
            }
        ],
        "vehicles": [
            {
                "id": 0,
                "start_lat": 13.6989,
                "start_lon": -89.1914,
                "end_lat": 13.6989,
                "end_lon": -89.1914,
                "provided_skills": ["refrigerado"],
                "capacity_weight": 100,
                "capacity_volume": 10,
                "capacity_quantity": 20,
                "start_time": 400,
                "end_time": 1200,
                "use_quantity": False,
                "use_weight": False,
                "use_volume": False,
                "plate_number": "ABC-123"
            }
        ],
        "num_vehicles": 1,
        "depot": 0,
        "strict_mode": False,
        "buffer_minutes": 10,
        "peak_hours": [],
        "peak_multiplier": 1.3,
        "peak_buffer_minutes": 20,
        "mode": "driving",
        "units": "metric",
        "include_polylines": True,
        "detail_level": "full"
    }
    response = client.post("/vrp-v2", json=payload)
    print("Status code:", response.status_code)
    print("Response body:", response.text)
    assert response.status_code == 200, f"Error inesperado: {response.text}"

# Puedes ejecutar este test con: pytest -s test_vrp_v2.py

def test_vrp_v2_wait_for_client_window():
    """
    Verifica que el vehículo espera en el depósito y sale a las 09:45 para llegar al cliente a las 10:00.
    """
    payload = {
        "locations": [
            {
                "id": 0,
                "client_uuid": "loc-0",
                "name": "Depósito",
                "lat": 13.7000,
                "lon": -89.2000,
                "demand": 0,
                "weight": 0,
                "volume": 0,
                "time_window": [420, 720],  # 07:00 a 12:00
                "service_time": 5,
                "required_skills": []
            },
            {
                "id": 1,
                "client_uuid": "loc-1",
                "name": "Cliente A",
                "lat": 13.7100,
                "lon": -89.2000,
                "demand": 1,
                "weight": 10,
                "volume": 0.5,
                "time_window": [600, 780],  # 10:00 a 13:00
                "service_time": 10,
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
                "capacity_quantity": 20,
                "start_time": 420,  # 07:00
                "end_time": 1200,  # 20:00
                "use_quantity": False,
                "use_weight": False,
                "use_volume": False,
                "plate_number": "ABC-123"
            }
        ],
        "num_vehicles": 1,
        "depot": 0,
        "strict_mode": False,
        "buffer_minutes": 10,
        "peak_hours": [],
        "peak_multiplier": 1.0,
        "peak_buffer_minutes": 0,
        "mode": "driving",
        "units": "metric",
        "include_polylines": False,
        "detail_level": "full"
    }
    response = client.post("/vrp-v2", json=payload)
    print("Status code:", response.status_code)
    print("Response body:", response.text)
    assert response.status_code == 200, f"Error inesperado: {response.text}"
    data = response.json()
    # Busca el detalle de la parada del cliente
    stops = data["solution"]["details"][0]["stops"]
    cliente_stop = next((s for s in stops if s["location_id"] == 1), None)
    assert cliente_stop is not None, "No se encontró la parada del cliente"
    # Debe llegar a las 10:00 (600 min)
    assert cliente_stop["arrival_time"] == 600, f"El vehículo no llegó a las 10:00, llegó a las {cliente_stop['arrival_time']} min"
    # El service_start debe ser igual a arrival_time (sin espera extra)
    assert cliente_stop["service_start"] == 600, f"El servicio no inicia a las 10:00"
    # El stop anterior es el depósito, debe salir a las 09:45 (585 min)
    deposito_stop = stops[0]
    assert deposito_stop["service_end"] == 585, f"El vehículo no sale del depósito a las 09:45, sino a las {deposito_stop['service_end']} min"
    print("Test de ventana de espera OK")
