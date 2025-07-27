"""
API principal para el servicio de optimización de rutas VRP.

Este módulo implementa un servicio FastAPI que expone un endpoint para resolver
problemas de enrutamiento de vehículos (VRP) con múltiples restricciones.
"""
import logging
import sys
import io
import asyncio
import aiohttp
import os
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from vrp_validator import validate_vrp
from vrp_converter import JsonToVrpDataConverter
from vrp_solver import VRPSolver
from solution_presenter import JsonSolutionPresenter

# ===== CONFIGURACIÓN =====
# Cargar variables de entorno
load_dotenv()

# Constantes de configuración
DEFAULT_LOG_LEVEL = logging.INFO
UVICORN_LOG_LEVEL = logging.WARNING
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

# Configuración de logging
def configure_logging():
    """Configura el sistema de logging de la aplicación."""
    logging.basicConfig(
        level=DEFAULT_LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )
    
    # Configurar niveles de log para módulos específicos
    for module in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        logging.getLogger(module).setLevel(UVICORN_LOG_LEVEL)
    
    # Reducir ruido de dependencias
    for module in ["ortools", "google", "google.protobuf"]:
        logging.getLogger(module).setLevel(logging.ERROR)

# Inicializar logging
configure_logging()
logger = logging.getLogger(__name__)

class SuppressOutput:
    """Context manager para suprimir la salida estándar y de error."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="API de Optimización de Rutas VRP",
    description="API para resolver problemas de enrutamiento de vehículos con múltiples restricciones",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia para suprimir salida (usada en los endpoints)
suppress_output = SuppressOutput()

async def convert_scenario(scenario: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
    """Convierte el escenario al formato del solver."""
    async with aiohttp.ClientSession() as session:
        converter = JsonToVrpDataConverter(scenario, session)
        vrp_data, used_cache = await converter.convert()
    return vrp_data, used_cache

@app.post("/optimize")
async def optimize_vrp(request_data: Dict[str, Any] = Body(...)):
    """
    Endpoint principal para la optimización de rutas VRP.
    
    Args:
        request_data: Diccionario con los datos del problema VRP.
        
    Returns:
        Dict con la solución del problema o mensaje de error.
        
    Example:
        POST /optimize
        {
            "locations": [...],
            "vehicles": [...],
            "pickups_deliveries": [...]
        }
    """
    with suppress_output:
        try:
            # 1. Manejar estructura anidada
            data = request_data.get('data', request_data)
            logger.debug("Datos recibidos para optimización")
            
            # 2. Validar los datos
            logger.info("Validando datos de entrada...")
            validation_result = validate_vrp(data)
            
            if validation_result.get('errors'):
                logger.warning("Errores de validación encontrados")
                return {
                    "status": "error",
                    "message": "Error en los datos de entrada",
                    "errors": validation_result.get('errors', []),
                    "warnings": validation_result.get('warnings', [])
                }
            
            # 3. Loggear advertencias si las hay
            if warnings := validation_result.get('warnings'):
                logger.warning(f"Advertencias de validación: {warnings}")
            
            # 4. Convertir a formato del solver
            logger.info("Convirtiendo datos...")
            vrp_data, used_cache = await convert_scenario(validation_result['cleaned_data'])
            
            # 5. Resolver el VRP
            logger.info("Iniciando optimización...")
            solver = VRPSolver(
                vrp_data=vrp_data,
                distance_matrix=vrp_data.get('distance_matrix', []),
                time_matrix=vrp_data.get('time_matrix', [])
            )
            solution = solver.solve()
            
            # 6. Procesar y devolver la solución
            logger.info("Procesando solución...")
            result = JsonSolutionPresenter.present(solution, vrp_data)
            result.update({
                'used_cache': used_cache,
                'excluded_nodes': validation_result.get('excluded_nodes', [])
            })
            
            logger.info("Optimización completada exitosamente")
            return result
            
        except Exception as e:
            logger.exception("Error durante la optimización")
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "message": "Error interno del servidor durante la optimización",
                    "details": str(e)
                }
            )

def start_server():
    """Inicia el servidor de la API."""
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True,
        log_level="warning"
    )

if __name__ == "__main__":
    logger.info(f"Iniciando servidor en {SERVER_HOST}:{SERVER_PORT}")
    start_server()