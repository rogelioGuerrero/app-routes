import logging
import time
import uuid
from fastapi import APIRouter, HTTPException, status
from typing import List, Dict, Any, Optional
from schemas_skills import VRPSkillsRequest, SkillsLocation, SkillsVehicle
from schemas import VRPAdvancedResponse, VehicleRoute, Location
from vrp_services import (
    VRPError,
    validate_vrp_request,
    generate_distance_matrices,
    build_vrp_response
)
from vrp.solver import SolverFactory, SolverType, VRPSolution
from vrp_utils import min_to_hhmm, add_warning

# Configuración de logging (solo consola para desarrollo)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Solo consola
)
logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/vrp-v2.5", response_model=VRPAdvancedResponse)
async def vrp_v2_5(request: VRPSkillsRequest):
    """
    Endpoint para el cálculo de rutas VRP con soporte para múltiples proveedores de matrices.
    
    Flujo de generación de matrices:
    1. Valida las coordenadas de entrada
    2. Intenta con ORS (OpenRouteService)
    3. Si falla, intenta con Google Distance Matrix
    4. Si falla, usa distancia euclidiana como último recurso
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] Iniciando solicitud VRP v2.5")
    
    try:
        # 1. Validar coordenadas
        logger.info(f"[{request_id}] Validando coordenadas...")
        # validate_vrp_request es asíncrona, usar await
        is_valid, error_response = await validate_vrp_request(request)
        if not is_valid:
            logger.error(f"[{request_id}] Error en validación: {error_response}")
            raise HTTPException(
                status_code=error_response.get("status_code", status.HTTP_400_BAD_REQUEST),
                detail={
                    "message": error_response.get("message", "Error de validación"),
                    "details": error_response.get("details", {}),
                    "request_id": request_id
                }
            )
        logger.info(f"[{request_id}] ✓ Coordenadas validadas")
        
        # 2. Generar matrices
        warnings = []
        try:
            dist_matrix, time_matrix, warnings = await generate_distance_matrices(
                request=request,
                request_id=request_id,
                warnings=warnings
            )
            
            # 3. Resolver el VRP
            logger.info(f"[{request_id}] Resolviendo VRP...")
            try:
                # Preparar restricciones del solver
                solver_options = request.solver_options.dict()
                
                # Configurar restricciones básicas
                constraints = {
                    'max_route_duration_sec': solver_options.pop('max_route_duration_min') * 60,  # Convertir a segundos
                    'time_limit_sec': solver_options.pop('time_limit_sec'),
                    **solver_options  # Incluir el resto de las opciones del solver
                }
                
                # Agregar restricciones de capacidad si están presentes en la solicitud
                if hasattr(request, 'demands') and request.demands and hasattr(request, 'vehicle_capacities') and request.vehicle_capacities:
                    constraints.update({
                        'demands': request.demands,
                        'vehicle_capacities': request.vehicle_capacities
                    })
                
                # Convertir vehículos a diccionarios para el solver
                vehicles_as_dicts = [
                    {
                        'id': v.id,
                        'depot_id': v.depot_id,
                        'start_lat': v.start_lat,
                        'start_lon': v.start_lon,
                        'end_lat': v.end_lat if v.end_lat is not None else v.start_lat,
                        'end_lon': v.end_lon if v.end_lon is not None else v.start_lon,
                        'capacity': v.capacity_quantity if v.use_quantity else None
                    }
                    for v in request.vehicles
                ]
                
                # Crear y ejecutar el solver
                solver = SolverFactory.create_solver(SolverType.OR_TOOLS)
                solution = await solver.solve(
                    distance_matrix=dist_matrix,
                    time_matrix=time_matrix,
                    locations=request.locations,
                    vehicles=vehicles_as_dicts,
                    constraints=constraints
                )
                
                # 4. Construir respuesta con las rutas
                response = build_vrp_solution_response(
                    solution=solution,
                    request=request,
                    request_id=request_id,
                    start_time=start_time,
                    warnings=warnings
                )
                
                logger.info(f"[{request_id}] ✓ VRP resuelto exitosamente")
                return response
                
            except Exception as e:
                error_msg = f"Error al resolver el VRP: {str(e)}"
                logger.error(f"[{request_id}] {error_msg}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "message": error_msg,
                        "request_id": request_id,
                        "error_type": type(e).__name__
                    }
                )
            
        except VRPError as e:
            logger.error(f"[{request_id}] Error en generación de matrices: {e.message}")
            raise HTTPException(
                status_code=e.status_code,
                detail={
                    "message": e.message,
                    "details": e.details,
                    "request_id": request_id
                }
            )
            
    except HTTPException:
        # Re-lanzar excepciones HTTP existentes
        raise
        
    except Exception as e:
        # Capturar cualquier otro error inesperado
        error_msg = f"Error inesperado en el servidor: {str(e)}"
        logger.critical(f"[{request_id}] {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": error_msg,
                "request_id": request_id,
                "error_type": type(e).__name__
            }
        )
    
    finally:
        # Log final con métricas
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] Tiempo total de procesamiento: {total_time:.2f}s")


def build_vrp_solution_response(
    solution: VRPSolution,
    request: VRPSkillsRequest,
    request_id: str,
    start_time: float,
    warnings: List[Dict[str, Any]]
) -> VRPAdvancedResponse:
    """
    Construye la respuesta del VRP a partir de la solución.
    
    Args:
        solution: Solución del VRP
        request: Solicitud original
        request_id: ID de la solicitud
        start_time: Tiempo de inicio de la solicitud
        warnings: Lista de advertencias
        
    Returns:
        VRPAdvancedResponse con la respuesta formateada
    """
    # Calcular métricas
    total_distance = sum(solution.distances)
    total_time_min = sum(solution.times)
    
    # Construir rutas
    routes = []
    for i, (route_indices, distance, route_time) in enumerate(zip(solution.routes, solution.distances, solution.times)):
        # Obtener las ubicaciones de la ruta
        route_locations = []
        for loc_idx in route_indices:
            if 0 <= loc_idx < len(request.locations):
                loc = request.locations[loc_idx]
                # Crear diccionario con todos los atributos relevantes
                location_dict = {
                    'id': loc.id,
                    'lat': loc.lat,
                    'lon': loc.lon,
                    'name': getattr(loc, 'name', f'Ubicación {loc.id}'),
                    'address': getattr(loc, 'address', ''),
                    'is_depot': getattr(loc, 'is_depot', False),
                    'demand': getattr(loc, 'demand', 0),
                    'weight': getattr(loc, 'weight', 0.0),
                    'volume': getattr(loc, 'volume', 0.0),
                    'time_window': getattr(loc, 'time_window', [0, 1440]),
                    'required_skills': getattr(loc, 'required_skills', []),
                    'service_time': getattr(loc, 'service_time', 0),
                    'priority': getattr(loc, 'priority', 1)
                }
                
                # Asegurar que los valores numéricos no sean None
                for field in ['demand', 'weight', 'volume', 'service_time', 'priority']:
                    if location_dict[field] is None:
                        location_dict[field] = 0.0 if field != 'demand' else 0
                
                route_locations.append(location_dict)
        
        # Crear objeto de ruta
        vehicle_id = i % len(request.vehicles) if request.vehicles else 0
        vehicle = request.vehicles[vehicle_id] if request.vehicles and vehicle_id < len(request.vehicles) else None
        
        route = VehicleRoute(
            vehicle_id=vehicle_id,
            vehicle_name=getattr(vehicle, 'name', f'Vehículo {vehicle_id}'),
            distance=distance,
            time=route_time,
            time_str=min_to_hhmm(route_time),
            locations=route_locations,
            load=solution.loads[i] if i < len(solution.loads) else 0
        )
        routes.append(route)
    
    # Construir metadatos
    metadata = {
        "request_id": request_id,
        "processing_time_seconds": round(time.time() - start_time, 4),
        "num_locations": len(request.locations),
        "num_vehicles": len(request.vehicles) if hasattr(request, 'vehicles') else 0,
        "total_distance": total_distance,
        "total_time_minutes": total_time_min,
        "total_time_formatted": min_to_hhmm(total_time_min),
        "unassigned_locations": solution.unassigned,
        "solver_metadata": solution.metadata
    }
    
    # Construir respuesta
    return VRPAdvancedResponse(
        routes=routes,
        total_distance=total_distance,
        total_time=total_time_min,
        unassigned_locations=solution.unassigned,
        warnings=warnings,
        metadata=metadata
    )
