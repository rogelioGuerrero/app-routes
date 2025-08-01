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

from fastapi import FastAPI, HTTPException, Body, UploadFile, File, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import tempfile
import os
from pathlib import Path

from vrp_validator import validate_vrp
from vrp_converter import JsonToVrpDataConverter
from vrp_solver import VRPSolver
from solution_presenter import JsonSolutionPresenter
from xlsx_to_json import xlsx_to_json

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
    description="API para resolver problemas de enrutamiento de vehículos con múltiples restricciones. Incluye soporte para carga de archivos XLSX.",
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

@app.post("/optimize/upload", status_code=status.HTTP_201_CREATED)
async def optimize_vrp_upload(file: UploadFile = File(...)):
    """
    Endpoint para optimizar rutas VRP a partir de un archivo XLSX.
    
    Args:
        file: Archivo XLSX con los datos del problema VRP.
        
    Returns:
        JSON con la solución del problema o mensaje de error.
        
    Example:
        POST /optimize/upload
        Body: form-data con archivo XLSX
    """
    # Validar extensión del archivo
    if not file.filename.lower().endswith(('.xlsx', '.xls')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Solo se permiten archivos Excel (.xlsx, .xls)"
        )
    
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
        try:
            # Guardar archivo subido
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error al procesar el archivo: {str(e)}"
            )
    
    try:
        # Convertir XLSX a JSON
        request_data = xlsx_to_json(temp_file_path)
        
        # Validar datos
        validate_vrp(request_data)
        
        # Convertir al formato del solver
        async with aiohttp.ClientSession() as session:
            converter = JsonToVrpDataConverter(request_data, session)
            vrp_data, _ = await converter.convert()
        
        # Resolver el problema
        solver = VRPSolver(vrp_data)
        solution = solver.solve()
        
        # Formatear la solución
        presenter = JsonSolutionPresenter()
        result = presenter.present(solution, request_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Error al procesar la solicitud: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error al procesar el archivo: {str(e)}"
        )
    finally:
        # Limpiar archivo temporal
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            logger.warning(f"No se pudo eliminar el archivo temporal {temp_file_path}: {str(e)}")

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