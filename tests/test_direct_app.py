import sys
import os
import json
import pytest
from fastapi.testclient import TestClient

# Añadir el directorio raíz del proyecto a sys.path para resolver las importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Es crucial importar la app DESPUÉS de ajustar el path
from main import app 

@pytest.fixture(scope="module")
def test_client():
    """Crea un cliente de prueba para la aplicación FastAPI."""
    client = TestClient(app)
    yield client


def test_cvrp_simple_direct_call(test_client):
    """Prueba un escenario CVRP simple usando el TestClient para obtener un error detallado."""
    print("\n--- INICIANDO PRUEBA DIRECTA CON TestClient ---")
    payload_path = os.path.join(os.path.dirname(__file__), 'cvrp_test_payload.json')
    
    with open(payload_path, 'r') as f:
        payload = json.load(f)

    print("--- Payload cargado, enviando petición ---")
    
    # Esta llamada debería fallar y darnos el traceback completo
    response = test_client.post("/api/v1/vrptw/solve-unified", json=payload)
    
    print(f"--- Respuesta recibida: Status Code {response.status_code} ---")
    if response.status_code != 200:
        error_log_path = os.path.join(os.path.dirname(__file__), 'error_response.log')
        with open(error_log_path, "w") as f:
            f.write(f"Status Code: {response.status_code}\n")
            f.write(f"Response Body: {response.text}\n")
    
    assert response.status_code == 200, f"Request failed with status {response.status_code}: {response.text}"
