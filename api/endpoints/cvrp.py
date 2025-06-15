"""Endpoints para el problema CVRP."""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, List
import logging

from core.advanced_solver_adapter import AdvancedSolverAdapter
from schemas.vrp_models import CVRPRequest, CVRPSolution, ProfileDefinition, ProfileParameter

# Configurar logging
logger = logging.getLogger(__name__)

# Crear router
router = APIRouter(
    prefix="/cvrp",
    tags=["CVRP"],
    responses={
        404: {"description": "No encontrado"},
        500: {"description": "Error interno del servidor"}
    }
)

@router.post(
    "/solve",
    response_model=CVRPSolution,
    status_code=status.HTTP_200_OK,
    summary="Resuelve un problema CVRP",
    description="""
    Resuelve un problema de Enrutamiento de Vehículos con Restricción de Capacidad (CVRP).
    
    - **locations**: Lista de ubicaciones (la primera debe ser el depósito con demanda 0)
    - **vehicles**: Lista de vehículos disponibles con sus capacidades
    - **providers**: Proveedores de matrices a usar (opcional, por defecto usa ORS)
    - **time_limit_seconds**: Tiempo máximo de resolución (por defecto: 30s)
    - **force_refresh**: Indica si se debe invalidar la caché (opcional, por defecto: False)
    """
)
async def solve_cvrp(request: CVRPRequest) -> CVRPSolution:
    """
    Resuelve un problema CVRP con los parámetros proporcionados, utilizando el adaptador avanzado.
    """
    try:
        logger.info("Recibida solicitud para resolver CVRP vía Advanced Adapter")
        adapter = AdvancedSolverAdapter()
        
        # El adaptador unificado se encarga de la lógica de proveedores y depots.
        solution = await adapter.solve_unified(
            locations=[loc.dict() for loc in request.locations],
            vehicles=[veh.dict() for veh in request.vehicles],
            force_refresh=request.force_refresh,
            time_limit_seconds=request.time_limit_seconds,
            solver_params=request.optimization_profile.override_params
        )
        
        logger.info(f"Solución CVRP encontrada con estado: {solution.status}")
        return solution

    except Exception as e:
        logger.error(f"Error al resolver CVRP: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al resolver el problema: {str(e)}"
        )

@router.get("/providers", response_model=Dict[str, Any])
async def get_available_providers() -> Dict[str, Any]:
    """
    Obtiene la lista de proveedores de matrices disponibles y sus parámetros.
    
    Returns:
        Diccionario con información de los proveedores disponibles
    """
    return {
        "providers": [
            {
                "name": "ors",
                "description": "OpenRouteService - Requiere API key",
                "params": {
                    "profile": {
                        "type": "str",
                        "required": True,
                        "default": "driving-car",
                        "options": ["driving-car", "driving-hgv", "cycling-regular", "foot-walking"],
                        "description": "Perfil de ruta a utilizar"
                    },
                    "api_key": {
                        "type": "str",
                        "required": True,
                        "description": "API key de OpenRouteService"
                    }
                }
            },
            {
                "name": "google",
                "description": "Google Maps Distance Matrix API - Requiere API key",
                "params": {
                    "mode": {
                        "type": "str",
                        "required": False,
                        "default": "driving",
                        "options": ["driving", "walking", "bicycling", "transit"],
                        "description": "Modo de viaje"
                    },
                    "api_key": {
                        "type": "str",
                        "required": True,
                        "description": "API key de Google Cloud"
                    },
                    "traffic_model": {
                        "type": "str",
                        "required": False,
                        "options": ["best_guess", "pessimistic", "optimistic"],
                        "description": "Modelo de tráfico (solo para modo driving con departure_time)"
                    },
                    "departure_time": {
                        "type": "int",
                        "required": False,
                        "description": "Hora de salida en timestamp UNIX (afecta el tráfico)"
                    }
                }
            },
            {
                "name": "euclidean",
                "description": "Distancia euclidiana (sin API key, menos precisa)",
                "params": {
                    "speed_kmh": {
                        "type": "float",
                        "required": False,
                        "default": 50.0,
                        "description": "Velocidad promedio para calcular tiempos"
                    }
                }
            }
        ]
    }

@router.get(
    "/profiles",
    response_model=List[ProfileDefinition],
    status_code=status.HTTP_200_OK,
    summary="Obtiene perfiles de optimización disponibles"
)
async def get_vrp_profiles() -> List[ProfileDefinition]:
    """
    Devuelve metadatos de perfiles: cost_saving, punctuality, balanced.
    """
    return [
        ProfileDefinition(
            name="cost_saving",
            description="Minimiza coste total",
            parameters=[
                ProfileParameter(name="distance_weight", type="float", default=0.8, required=True, description="Peso para distancia"),
                ProfileParameter(name="time_weight", type="float", default=0.2, required=True, description="Peso para tiempo"),
            ]
        ),
        ProfileDefinition(
            name="punctuality",
            description="Prioriza cumplimiento de ventanas de tiempo",
            parameters=[
                ProfileParameter(name="lateness_penalty", type="int", default=1000, required=True, description="Penalización por retraso"),
            ]
        ),
        ProfileDefinition(
            name="balanced",
            description="Balance entre coste y puntualidad",
            parameters=[
                ProfileParameter(name="distance_weight", type="float", default=0.5, required=True, description="Peso para distancia"),
                ProfileParameter(name="time_weight", type="float", default=0.5, required=True, description="Peso para tiempo"),
            ]
        )
    ]
