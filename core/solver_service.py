from typing import List, Dict, Any, Optional
from core.advanced_solver_adapter import AdvancedSolverAdapter
from schemas.vrp_models import CVRPSolution, VRPTWSolution # Import VRPTWSolution
from typing import Union, Tuple # For type hinting
import logging

logger = logging.getLogger(__name__)

async def solve_unified_service(
    locations: List[Dict[str, Any]],
    vehicles: List[Dict[str, Any]],
    depots: Optional[List[int]],
    starts_ends: Optional[List[List[int]]],
    pickups_deliveries: Optional[List[List[str]]],
    allow_skipping: bool,
    penalties: Optional[List[int]],
    max_route_duration: Optional[int],
    force_refresh: bool,
    time_limit_seconds: int,
    solver_params: Optional[Dict[str, Any]],
    optimization_profile: Optional[Dict[str, Any]] = None
) -> Union[Tuple[Union[VRPTWSolution, CVRPSolution], Dict[str, Any]], Dict[str, Any]]: # Adjusted return type hint
    """
    Servicio que envuelve la lógica de resolución unificada y devuelve un dict listo para FastAPI.
    """
    adapter = AdvancedSolverAdapter()
    # solve_unified ahora devuelve una tupla: (solución_del_solver, metadatos_matriz)
    solver_solution, matrix_metadata = await adapter.solve_unified(
        locations=locations,
        vehicles=vehicles,
        depots=depots,
        starts_ends=starts_ends,
        pickups_deliveries=pickups_deliveries,
        allow_skipping=allow_skipping,
        penalties=penalties,
        max_route_duration=max_route_duration,
        force_refresh=force_refresh,
        time_limit_seconds=time_limit_seconds,
        solver_params=solver_params,
        optimization_profile=optimization_profile
    )
    # Devolvemos la solución Pydantic (VRPTWSolution o CVRPSolution) y los metadatos de la matriz.

    logger.info(f"SERVICE: Received solver_solution of type: {type(solver_solution)} from adapter.")

    if isinstance(solver_solution, (VRPTWSolution, CVRPSolution)):
        # Solution is already a Pydantic model (VRPTWSolution or CVRPSolution).
        # Pass it and matrix_metadata to the endpoint.
        logger.info(f"SERVICE: solver_solution is a recognized Pydantic model ({type(solver_solution).__name__}). Returning to endpoint.")
        return solver_solution, matrix_metadata
    # elif isinstance(solver_solution, CVRPSolution): # Combined into the check above
    #     # Si es CVRPSolution, ya está "formateada" (es un objeto Pydantic).
    #     # Devolvemos el objeto Pydantic y los metadatos de la matriz.
    #     logger.info(f"SERVICE: solver_solution is CVRPSolution. Returning to endpoint.")
    #     return solver_solution, matrix_metadata 
    else:
        # Handle unexpected cases or actual error dictionaries from the adapter.
        if isinstance(solver_solution, dict) and 'error' in solver_solution:
            logger.warning(f"SERVICE: solver_solution is an error dictionary from adapter: {solver_solution}")
            # Propagate error dict from adapter, ensure matrix_metadata is included.
            final_error_dict = solver_solution.copy()
            if "metadata" not in final_error_dict:
                final_error_dict["metadata"] = {}
            final_error_dict["metadata"].update(matrix_metadata) # Ensure matrix_meta is part of the error response
            return final_error_dict

        # If it's not a recognized Pydantic model and not an error dict from adapter,
        # then it's an unrecognized type from the solver/adapter.
        status = getattr(solver_solution, 'status', "UNKNOWN_ADAPTER_OUTPUT_TYPE")
        error_message = "Tipo de salida del adaptador no reconocido por el servicio."
        
        logger.error(f"SERVICE: {error_message} Type: {type(solver_solution)}. Content (partial): {str(solver_solution)[:200]}")
        
        error_response = {
            "status": status,
            "error": error_message,
            "actual_type": str(type(solver_solution)),
            "metadata": matrix_metadata # Al menos devolvemos los metadatos de la matriz
        }
        return error_response

