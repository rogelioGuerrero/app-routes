from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError
import traceback
from typing import Dict, Any, List
import logging

from schemas.vrp_models import (
    VRPTWRequest, UnifiedVRPRequest, UnifiedVRPSolution, ProfileDefinition,
    VRPTWSolution as schemas_VRPTWSolution,
    CVRPSolution as schemas_CVRPSolution
) # Removed CVRPTWSolution, kept CVRPSolution
from core.advanced_solver_adapter import AdvancedSolverAdapter
from core.solver_service import solve_unified_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/vrptw",
    tags=["VRPTW"],
    responses={404: {"description": "No encontrado"}, 500: {"description": "Error interno"}}
)

@router.post(
    "/solve",
    response_model=UnifiedVRPSolution,
    status_code=status.HTTP_200_OK,
    summary="Resuelve VRP con ventanas, peso/volumen y PdD",
    description="""
    Optimización VRP con:
    - Ventanas de tiempo
    - Capacidades de peso y volumen
    - Pickup & Delivery
    """
)
async def solve_vrptw(request: VRPTWRequest) -> UnifiedVRPSolution:
    """
    Resuelve un problema VRPTW usando el adaptador unificado para consistencia.
    """
    logger.info("Recibida solicitud para resolver VRPTW vía Advanced Adapter")
    try:
        adapter = AdvancedSolverAdapter()

        # Mapear la solicitud VRPTW a los parámetros del solver unificado.
        # La lógica de detección de depósitos, pickups, etc., está en solve_unified.
        solution = await adapter.solve_unified(
            locations=[loc.dict() for loc in request.locations],
            vehicles=[veh.dict() for veh in request.vehicles],
            force_refresh=request.force_refresh,
            time_limit_seconds=request.time_limit_seconds,
            solver_params=request.optimization_profile.override_params,
            optimization_profile=request.optimization_profile.dict()
        )

        logger.info(f"Solución VRPTW encontrada con estado: {solution.status}")
        return solution

    except Exception as e:
        logger.error(f"Error al resolver VRPTW: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al resolver el problema: {str(e)}"
        )

# Nuevo endpoint unificado
@router.post(
    "/solve-unified",
    response_model=UnifiedVRPSolution,
    status_code=status.HTTP_200_OK,
    summary="Resuelve cualquier variante de VRP",
    description="Un endpoint para CVRP, VRPTW, PDP, MDVRP según campos en la solicitud."
)
async def solve_unified(request: UnifiedVRPRequest) -> UnifiedVRPSolution:
    """Endpoint que detecta y resuelve la variante VRP adecuada."""
    logger.info("Endpoint.solve_unified: inicio")
    try:
        # Usar servicio desacoplado para resolver
        # solve_unified_service ahora puede devolver RawSolutionData, Dict (para CVRPSolution) o Dict (para error)
        service_output = await solve_unified_service(
            locations=[loc.dict() for loc in request.locations],
            vehicles=[v.dict() for v in request.vehicles],
            depots=request.depots,
            starts_ends=request.starts_ends,
            pickups_deliveries=request.pickups_deliveries,
            allow_skipping=request.allow_skipping_nodes,
            penalties=request.penalties,
            max_route_duration=request.max_route_duration,
            force_refresh=request.force_refresh,
            time_limit_seconds=request.time_limit_seconds,
            solver_params=request.solver_params,
            optimization_profile=request.optimization_profile.dict()
        )
        
        # Detailed logging for service_output
        logger.info(f"ENDPOINT: service_output type: {type(service_output)}")
        # Log content carefully, it might be large or complex
        if isinstance(service_output, tuple) or isinstance(service_output, dict):
            logger.info(f"ENDPOINT: service_output content (first 500 chars): {str(service_output)[:500]}")
        elif hasattr(service_output, 'model_dump_json'): # For Pydantic models
            logger.info(f"ENDPOINT: service_output content (model dump, first 500 chars): {service_output.model_dump_json()[:500]}")
        else:
            logger.info(f"ENDPOINT: service_output content (str, first 500 chars): {str(service_output)[:500]}")

        if isinstance(service_output, tuple):
            logger.info("ENDPOINT: Service returned a tuple, expecting (PydanticSolutionModel, MatrixMetadata).")
            pydantic_solution_object, matrix_meta = service_output # Unpack the tuple
            
            final_metadata = pydantic_solution_object.metadata.copy() if pydantic_solution_object.metadata else {}
            final_metadata.update(matrix_meta)

            # Routes in VRPTWSolution/CVRPSolution are already lists of Pydantic Route models.
            unified_solution_data = {
                "status": pydantic_solution_object.status,
                "routes": pydantic_solution_object.routes,
                "total_distance": pydantic_solution_object.total_distance,
                "total_load": pydantic_solution_object.total_load,
                "total_vehicles_used": pydantic_solution_object.total_vehicles_used,
                "total_weight": getattr(pydantic_solution_object, "total_weight", None),
                "total_volume": getattr(pydantic_solution_object, "total_volume", None),
                "metadata": final_metadata
            }
            solution = UnifiedVRPSolution(**unified_solution_data)
            logger.info(f"ENDPOINT: Successfully created UnifiedVRPSolution from service tuple. Status: {solution.status}")
            return solution

        elif isinstance(service_output, dict):
            # This case handles error dictionaries returned by the service.
            logger.warning(f"ENDPOINT: Received dict from service, likely an error: {str(service_output)[:500]}")
            
            if 'error' in service_output:
                error_detail = f"Service returned an error: {service_output.get('error', 'Unknown error')}"
                if "actual_type" in service_output: # From our solver_service error structure
                     error_detail += f" (Type: {service_output['actual_type']})"
                logger.error(f"ENDPOINT: {error_detail}. Full service output dict: {service_output}")
                raise HTTPException(status_code=500, detail=error_detail)
            else:
                # If it's a dict but not an error, try to parse it. This is an unlikely path.
                logger.info("ENDPOINT: Received dict from service (not identified as error). Attempting to parse as UnifiedVRPSolution.")
                solution = UnifiedVRPSolution(**service_output) # This will raise ValidationError if malformed
                logger.info(f"ENDPOINT: Successfully created UnifiedVRPSolution from service dict. Status: {solution.status}")
                return solution

        else:
            # This case should ideally not be reached if solver_service behaves as expected.
            logger.error(f"ENDPOINT: Unexpected service_output type: {type(service_output)}. Content: {str(service_output)[:500]}")
            raise HTTPException(status_code=500, detail=f"Internal error: Unexpected data type from service: {type(service_output).__name__}")

        logger.info("Endpoint.solve_unified: retorno unificado")
        return solution
    except ValidationError as e:
        # This will catch ValidationErrors raised from UnifiedVRPSolution constructor or from .model_validate()
        logger.error(f"ENDPOINT: Top-level ValidationError caught. Errors: {e.errors()}", exc_info=True)
        # The traceback from the logger.error in the specific path (if that's where it happened) will be more specific.
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e.errors()}. Traceback: {traceback.format_exc()}")
    except Exception as e:
        logger.error(f"Error en Unified VRP: {e}", exc_info=True)
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"An internal error occurred: {e}. Traceback: {error_trace}"
        )

@router.get(
    "/profiles",
    response_model=List[ProfileDefinition],
    status_code=status.HTTP_200_OK,
    summary="Obtiene perfiles de optimización disponibles"
)
async def get_vrptw_profiles() -> List[ProfileDefinition]:
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
