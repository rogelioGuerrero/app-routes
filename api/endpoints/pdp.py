from fastapi import APIRouter, HTTPException, status
import logging
from schemas.vrp_models import UnifiedVRPRequest, UnifiedVRPSolution
from core.advanced_solver_adapter import AdvancedSolverAdapter

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/pdp",
    tags=["PDP"],
    responses={404: {"description": "No encontrado"}, 500: {"description": "Error interno"}}
)

@router.post(
    "/solve",
    response_model=UnifiedVRPSolution,
    status_code=status.HTTP_200_OK,
    summary="Resuelve Pickup & Delivery (PDP)",
    description="Resuelve problemas de Pickup & Delivery sin ventanas de tiempo."
)
async def solve_pdp(request: UnifiedVRPRequest) -> UnifiedVRPSolution:
    """Endpoint para resolver PDP."""
    logger.info("Endpoint.solve_pdp: inicio")
    try:
        adapter = AdvancedSolverAdapter()
        raw = await adapter.solve_unified(
            locations=[loc.dict() for loc in request.locations],
            vehicles=[v.dict() for v in request.vehicles],
            depots=request.depots,
            starts_ends=request.starts_ends,
            allow_skipping=request.allow_skipping_nodes,
            penalties=request.penalties,
            max_route_duration=request.max_route_duration,
            force_refresh=request.force_refresh,
            time_limit_seconds=request.time_limit_seconds,
            solver_params=request.solver_params
        )
        routes_api = [r.dict() for r in raw.routes]
        solution = UnifiedVRPSolution(
            status=raw.status,
            routes=routes_api,
            total_distance=raw.total_distance,
            total_load=raw.total_load,
            total_vehicles_used=raw.total_vehicles_used,
            total_weight=raw.total_weight,
            total_volume=raw.total_volume,
            metadata=raw.metadata
        )
        logger.info("Endpoint.solve_pdp: retorno solución")
        return solution
    except Exception as e:
        logger.error(f"Error en PDP: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
