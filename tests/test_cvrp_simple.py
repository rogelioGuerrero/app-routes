import pytest
import sys
import os
import aiohttp
import json
import logging

# Añadir el directorio raíz del proyecto a sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from schemas.vrp_models import VRPSolutionStatus

# Configuración de logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

@pytest.fixture
async def http_session():
    async with aiohttp.ClientSession() as session:
        yield session

@pytest.mark.asyncio
async def test_cvrp_simple_scenario(http_session):
    """Prueba un escenario CVRP simple para verificar la selección del CVRPSolver."""
    scenario_name = "CVRP_SIMPLE_SUCCESS"
    payload_path = os.path.join(os.path.dirname(__file__), 'cvrp_test_payload.json')
    
    with open(payload_path, 'r') as f:
        payload = json.load(f)

    logger.info("\n%s INICIANDO ESCENARIO: %s %s", "="*20, scenario_name.upper(), "="*20)
    logger.debug("--- Payload Enviado ---\n%s", json.dumps(payload, indent=2))

    url = f"{BASE_URL}/api/v1/vrptw/solve-unified"
    try:
        timeout = aiohttp.ClientTimeout(total=90)
        async with http_session.post(url, json=payload, timeout=timeout) as response:
            status_code = response.status
            response_text = await response.text()

            logger.info("\n--- Respuesta Recibida (Status: %s) ---", status_code)
            logger.debug("--- Raw Response Body ---\n%s", response_text)

            response_json = {}
            try:
                response_json = json.loads(response_text)
            except json.JSONDecodeError:
                logger.error("Failed to decode JSON from response body.")

            solution_status = response_json.get('status', 'N/A')

            assert status_code == 200, f"Código HTTP inesperado: {status_code}\nResponse Body:\n{response_text}"
            assert solution_status in [VRPSolutionStatus.FEASIBLE.value, VRPSolutionStatus.OPTIMAL.value], f"Estado del solver inesperado. Esperado: FEASIBLE u OPTIMAL, Obtenido: {solution_status}"

            logger.info("\n--- ESCENARIO '%s' EXITOSO (Estado: %s) ---", scenario_name, solution_status)

    except aiohttp.ClientError as e:
        pytest.fail(f"Error en la solicitud para el escenario '{scenario_name}': {e}")
