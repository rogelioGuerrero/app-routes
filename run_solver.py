print("RUN_SOLVER_PY_START", flush=True)
import asyncio
import json
import sys
import logging
import os
import argparse # Importado para parsear argumentos
import shutil
from dotenv import load_dotenv

load_dotenv() # Carga variables desde .env

from services.distance_matrix.cache import MatrixCache

# Configuración robusta de logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# StreamHandler para la consola
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO) # Mostrar solo INFO y superior en consola
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# FileHandler para el archivo de log
file_handler = logging.FileHandler('d:/vrp_solver/output/solver.log', mode='w')
file_handler.setLevel(logging.DEBUG) # Guardar todo (DEBUG y superior) en el archivo
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger = logging.getLogger(__name__)

from models.vrp_models import VRPSolution
from core.solver_service import solve_unified_service

async def main():
    matrix_cache = MatrixCache()
    # Limpiar caché antes de cada ejecución para asegurar datos frescos
    # (útil durante la depuración, podría eliminarse en producción)
    try:
        # matrix_cache.clear() # Llama al método de instancia del singleton matrix_cache
        # El logger.info ya está dentro del método clear de MatrixCache
        pass # Añadido para evitar error de indentación con try/except vacío
    except Exception as e:
        logger.error(f"Error al limpiar el caché de matrices: {e}")

    parser = argparse.ArgumentParser(description="Run VRP Solver from a JSON input file.")
    parser.add_argument('--file', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output-file', type=str, help='Path to the output JSON file.')
    args = parser.parse_args()

    input_path = args.file
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)

    print("--- RUN_SOLVER.PY: DATA LOADED ---", flush=True)
    print(json.dumps(data, indent=2), flush=True)
    print("--- RUN_SOLVER.PY: CALLING SERVICE ---", flush=True)

    # Map swagger style keys to service parameters
    result = await solve_unified_service(
        locations=data.get("locations"),
        vehicles=data.get("vehicles"),
        depots=data.get("depots"),
        starts_ends=data.get("starts_ends"),
        allow_skipping=data.get("allow_skipping_nodes", False),
        penalties=data.get("penalties"),
        max_route_duration=data.get("max_route_duration"),
        force_refresh=data.get("force_refresh", False),
        time_limit_seconds=data.get("time_limit_seconds", 30),
        solver_params=data.get("solver_params"),
        optimization_profile=data.get("optimization_profile"),
    )

    print("--- RUN_SOLVER.PY: SERVICE RETURNED ---", flush=True)
    print(f"Result type: {type(result)}", flush=True)
    print(f"Result content: {result}", flush=True)
    print("--- RUN_SOLVER.PY: DUMPING JSON ---", flush=True)

    logger.info(f"Resultado obtenido del servicio, tipo: {type(result)}")
    logger.debug(f"Contenido del resultado (antes de JSON dump): {result}")

    try:
        output_content = json.dumps(result, indent=2)
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(output_content)
            logger.info(f"Resultado escrito a {args.output_file} exitosamente.")
        else:
            print(output_content)
            logger.info("Resultado impreso a stdout exitosamente.")
    except TypeError as e:
        logger.error(f"Error al serializar el resultado a JSON: {e}")
        logger.error(f"Objeto que causó el error (type: {type(result)}): {str(result)[:500]}...") # Loguear solo una parte para evitar logs masivos
        # Imprimir un JSON de error
        print(f"{{ \"status\": \"SERIALIZATION_ERROR\", \"error_message\": \"{str(e)}\", \"raw_result_type\": \"{str(type(result))}\" }}")
    except Exception as e:
        logger.error(f"Error inesperado durante la impresión del resultado: {e}")
        # Imprimir un JSON de error
        print(f"{{ \"status\": \"UNEXPECTED_PRINT_ERROR\", \"error_message\": \"{str(e)}\" }}")

if __name__ == '__main__':
    asyncio.run(main())
