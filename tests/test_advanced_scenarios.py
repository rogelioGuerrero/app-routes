import pytest
import sys
import os
import aiohttp
import json
import logging
from schemas.vrp_models import CVRPSolution, VRPTWSolution, VRPSolutionStatus

# Añadir el directorio raíz del proyecto a sys.path para resolver las importaciones de la aplicación
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Configuración de logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

@pytest.fixture
async def http_session():
    async with aiohttp.ClientSession() as session:
        yield session

def time_to_seconds(time_str):
    """Convierte un string 'HH:MM' a segundos desde la medianoche."""
    try:
        h, m = map(int, time_str.split(':'))
        return h * 3600 + m * 60
    except ValueError:
        return 0

async def run_test_scenario_and_assert(http_session, scenario_name, payload, expected_final_statuses, expect_dropped_nodes=False):
    """Función para enviar un escenario de prueba al endpoint y verificar la respuesta con aserciones."""
    logger.info("\n%s INICIANDO ESCENARIO: %s %s", "="*20, scenario_name.upper(), "="*20)
    logger.debug("--- Payload Enviado ---\n%s", json.dumps(payload, indent=2))

    url = f"{BASE_URL}/api/v1/vrptw/solve-unified"
    try:
        timeout = aiohttp.ClientTimeout(total=90)  # 90 segundos de timeout, generoso para CI
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
            dropped_nodes = response_json.get('metadata', {}).get('dropped_node_ids', [])

            if not (200 <= status_code < 300):
                with open("tests/advanced_error_response.log", "w", encoding='utf-8') as f:
                    f.write(f"Status Code: {status_code}\nResponse Body: {response_text}")
                pytest.fail(f"Request failed with status {status_code}. See tests/advanced_error_response.log for details.")
            assert solution_status in expected_final_statuses, f"Estado del solver inesperado. Esperado: {expected_final_statuses}, Obtenido: {solution_status}"

            if expect_dropped_nodes:
                assert dropped_nodes, "Se esperaban nodos descartados, pero la lista está vacía."
                logger.info(f"Nodos descartados detectados (esperado): {dropped_nodes}")
            else:
                assert not dropped_nodes, f"No se esperaban nodos descartados, pero se encontraron: {dropped_nodes}"

            logger.info("\n--- ESCENARIO '%s' EXITOSO (Estado: %s) ---", scenario_name, solution_status)

    except Exception as e:
        logger.error("Error inesperado en el ejecutor de pruebas para el escenario '%s': %s", scenario_name, e, exc_info=True)
        pytest.fail(f"Error inesperado en el ejecutor de pruebas para el escenario '{scenario_name}': {e}")

@pytest.mark.asyncio
async def test_super_scenario_complejo(http_session):
    from api.models import VRPSolutionStatus
    """Test para el escenario complejo con múltiples depósitos, P&D y descarte de nodos."""
    super_scenario_payload = {
        "allow_skipping_nodes": True,
        "api_version": "1.0",
        "optimization_profile": {"name": "cost_saving"},
        "time_limit_seconds": 30,
        "solver_params": {"profile": "driving-car"},
        "locations": [
            {"id": "depot_norte", "coords": [-99.133, 19.485], "time_window_start": time_to_seconds("08:00"), "time_window_end": time_to_seconds("18:00")},
            {"id": "depot_sur", "coords": [-99.175, 19.345], "time_window_start": time_to_seconds("08:00"), "time_window_end": time_to_seconds("18:00")},
            {"id": "recogida_A", "coords": [-99.1653, 19.4285], "demand": 10, "volume_demand": 5, "time_window_start": time_to_seconds("09:00"), "time_window_end": time_to_seconds("10:00")},
            {"id": "entrega_A", "coords": [-99.1629, 19.4053], "demand": -10, "volume_demand": -5, "time_window_start": time_to_seconds("11:00"), "time_window_end": time_to_seconds("12:00")},
            {"id": "recogida_B", "coords": [-99.1332, 19.4426], "demand": 8, "volume_demand": 3, "time_window_start": time_to_seconds("10:00"), "time_window_end": time_to_seconds("11:00")},
            {"id": "entrega_B", "coords": [-99.1639, 19.3587], "demand": -8, "volume_demand": -3, "time_window_start": time_to_seconds("13:00"), "time_window_end": time_to_seconds("14:00")},
            {"id": "cliente_C", "coords": [-99.1409, 19.4511], "demand": 15, "volume_demand": 10, "time_window_start": time_to_seconds("14:00"), "time_window_end": time_to_seconds("15:00"), "required_skills": ["manejo_delicado"]}
        ],
        "vehicles": [
            {"id": "veh_norte_1", "capacity": 20, "weight_capacity": 20, "volume_capacity": 15, "start_location_id": "depot_norte", "end_location_id": "depot_norte", "skills": ["manejo_delicado"]},
            {"id": "veh_sur_1", "capacity": 25, "weight_capacity": 25, "volume_capacity": 20, "start_location_id": "depot_sur", "end_location_id": "depot_sur"}
        ],
        "depots": [0, 1],
        "pickups_deliveries": [
            ["recogida_A", "entrega_A"],
            ["recogida_B", "entrega_B"]
        ]
    }
    await run_test_scenario_and_assert(http_session, "SUPER_ESCENARIO_COMPLEJO", super_scenario_payload, 
                                 expected_final_statuses=[VRPSolutionStatus.NO_SOLUTION_FOUND.value, VRPSolutionStatus.FEASIBLE.value],
                                 expect_dropped_nodes=True)

@pytest.mark.asyncio
async def test_habilidades_exito(http_session):
    from api.models import VRPSolutionStatus
    """Test para un escenario de habilidades que debe resolverse con éxito."""
    skills_scenario_success_payload = {
        "api_version": "1.0",
        "optimization_profile": {"name": "cost_saving"},
        "time_limit_seconds": 10,
        "solver_params": {"profile": "driving-car"},
        "locations": [
            {"id": "depot_central", "coords": [-99.1332, 19.4326], "time_window_start": time_to_seconds("08:00"), "time_window_end": time_to_seconds("18:00")},
            {"id": "cliente_frio", "coords": [-99.1653, 19.4285], "demand": 5, "time_window_start": time_to_seconds("09:00"), "time_window_end": time_to_seconds("10:00"), "required_skills": ["refrigeracion"]},
            {"id": "cliente_normal", "coords": [-99.1629, 19.4053], "demand": 7, "time_window_start": time_to_seconds("11:00"), "time_window_end": time_to_seconds("12:00")}
        ],
        "vehicles": [
            {"id": "veh_refrigerado", "capacity": 15, "weight_capacity": 15, "volume_capacity": 10, "start_location_id": "depot_central", "end_location_id": "depot_central", "skills": ["refrigeracion", "carga_pesada"]},
            {"id": "veh_estandar", "capacity": 15, "weight_capacity": 15, "volume_capacity": 10, "start_location_id": "depot_central", "end_location_id": "depot_central", "skills": ["carga_ligera"]}
        ],
        "depots": [0]
    }
    await run_test_scenario_and_assert(http_session, "HABILIDADES_EXITO", skills_scenario_success_payload, 
                                 expected_final_statuses=[VRPSolutionStatus.FEASIBLE.value, VRPSolutionStatus.OPTIMAL.value],
                                 expect_dropped_nodes=False)

@pytest.mark.asyncio
async def test_habilidades_fallo_esperado(http_session):
    from api.models import VRPSolutionStatus
    """Test para un escenario de habilidades que debe fallar por falta de vehículos compatibles."""
    skills_scenario_failure_payload = {
        "allow_skipping_nodes": True,
        "api_version": "1.0",
        "optimization_profile": {"name": "cost_saving"},
        "time_limit_seconds": 10,
        "solver_params": {"profile": "driving-car"},
        "locations": [
            {"id": "depot_central_f", "coords": [-99.1332, 19.4326], "time_window_start": time_to_seconds("08:00"), "time_window_end": time_to_seconds("18:00")},
            {"id": "cliente_peligroso", "coords": [-99.1653, 19.4285], "demand": 5, "time_window_start": time_to_seconds("09:00"), "time_window_end": time_to_seconds("10:00"), "required_skills": ["material_peligroso"]},
            {"id": "cliente_normal_f", "coords": [-99.1629, 19.4053], "demand": 7, "time_window_start": time_to_seconds("11:00"), "time_window_end": time_to_seconds("12:00")}
        ],
        "vehicles": [
            {"id": "veh_refrigerado_f", "capacity": 15, "weight_capacity": 15, "volume_capacity": 10, "start_location_id": "depot_central_f", "end_location_id": "depot_central_f", "skills": ["refrigeracion"]},
            {"id": "veh_estandar_f", "capacity": 15, "weight_capacity": 15, "volume_capacity": 10, "start_location_id": "depot_central_f", "end_location_id": "depot_central_f"}
        ],
        "depots": [0]
    }
    await run_test_scenario_and_assert(http_session, "HABILIDADES_FALLO_ESPERADO", skills_scenario_failure_payload, 
                                 expected_final_statuses=[VRPSolutionStatus.NO_SOLUTION_FOUND.value, VRPSolutionStatus.INFEASIBLE.value, VRPSolutionStatus.FEASIBLE.value],
                                 expect_dropped_nodes=True)
