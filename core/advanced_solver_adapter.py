import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import time
import logging
from datetime import datetime

from schemas.vrp_models import VRPTWSolution, CVRPSolution, ExtendedVehicle
from core.cvrp_solver import CVRPSolver
from core.vrptw_solver import VRPTWSolver, RawSolutionData
from core.matrix_adapter import MatrixAdapter

# --- DEBUGGING FILE LOAD ---
# Exception removed for further debugging
# --- END DEBUGGING ---


logger = logging.getLogger(__name__)

class AdvancedSolverAdapter:
    """
    Adapter para VRPTW (ventanas, peso, volumen, PdD).
    Aísla el solver avanzado del flujo básico.
    """
    def __init__(self):
        self.solver: Optional[VRPTWSolver] = None
        self.matrix_adapter = MatrixAdapter()

    async def solve_vrptw(
        self,
        locations: List[Dict[str, Any]],
        vehicles: List[Dict[str, Any]],
        providers: Optional[List[Tuple[str, dict]]] = None,
        time_limit_seconds: int = 30,
        force_refresh: bool = False,
        optimization_profile: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> VRPTWSolution:
        """
        Resuelve un problema VRPTW usando el solver avanzado.

        Args:
            locations: Lista de ubicaciones con ventanas, demanda, peso/volumen y pares PdD
            vehicles: Lista de vehículos con capacidades de peso y volumen
            matrix_provider_type = problem_data.solver_params.get('matrix_type', 'ors')
            profile = problem_data.solver_params.get('profile', 'driving-car')
            time_limit_seconds: Tiempo máximo de resolución
            force_refresh: Ignorar caché de matrices
            optimization_profile: Perfil de optimización
            **kwargs: Parámetros adicionales para proveedores

        Returns:
            VRPTWSolution: Solución optimizada

        """
        start_time = time.time()
        # Coordenadas para matrices
        coords = [{'lat': loc['lat'], 'lng': loc['lng']} for loc in locations]
        # Obtener matrices utilizando MatrixAdapter
        distances, durations, provider_used, from_cache = await self.matrix_adapter.get_matrices(
            coords, providers, force_refresh, **kwargs
        )
        # Cargar y resolver problema
        self.solver.load_problem(
            distance_matrix=distances,
            duration_matrix=durations,
            locations=locations,
            vehicles=vehicles,
            optimization_profile=optimization_profile
        )
        solution = await self.solver.solve(time_limit_seconds=time_limit_seconds)
        # Añadir metadatos
        solution.metadata.update({
            'matrix_provider': provider_used,
            'from_cache': from_cache,
            'num_locations': len(locations),
            'num_vehicles': solution.total_vehicles_used,
            'execution_time_seconds': round(time.time() - start_time, 2),
            'solved_at': datetime.utcnow().isoformat()
        })
        return solution

    async def solve_unified(
        self,
        locations: List[Dict[str, Any]],
        vehicles: List[Dict[str, Any]],
        depots: Optional[List[int]] = None,
        starts_ends: Optional[List[List[int]]] = None,
        pickups_deliveries: Optional[List[List[str]]] = None,
        allow_skipping: bool = False,
        penalties: Optional[List[int]] = None,
        max_route_duration: Optional[int] = None,
        force_refresh: bool = False,
        time_limit_seconds: int = 30,
        solver_params: Optional[Dict[str, Any]] = None,
        optimization_profile: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[Union[RawSolutionData, CVRPSolution], Dict[str, Any]]:
        # Validar depots y starts_ends
        if depots is not None:
            logger.info(f"AdvancedSolverAdapter.solve_unified: depots recibidos: {depots}")
        if starts_ends is not None:
            logger.info(f"AdvancedSolverAdapter.solve_unified: starts_ends recibidos: {starts_ends}")
        """
        Resuelve cualquier variante de VRP (CVRP, VRPTW, PDP). Detecta la variante según campos.
        """
        logger.info("AdvancedSolverAdapter.solve_unified: inicio")
        # Generar matrices internamente
        all_locations_with_coords = [{'lat': loc['coords'][1], 'lng': loc['coords'][0]} for loc in locations]
        profile = solver_params.get('profile', 'driving-car') if solver_params else 'driving-car'
        logger.info("AdvancedSolverAdapter.solve_unified: solicitando matrices")
        logger.debug(f"Locations sent to matrix_adapter: {all_locations_with_coords}, Profile: {profile}")
        matrix_data = await self.matrix_adapter.get_matrix(
            coordinates=all_locations_with_coords,
            profile=profile # El tipo de matriz (ors, google, etc.) se infiere en el adaptador
        )
        distances = matrix_data.distances
        durations = matrix_data.durations
        provider_used = matrix_data.provider

        # Aplicar factor de congestión si se especifica
        scaled_durations = durations # Por defecto, usar las originales
        if solver_params and durations:
            congestion_factor = float(solver_params.get('congestion_factor', 1.0))
            if congestion_factor < 0.1: # Evitar factores demasiado pequeños o negativos
                logger.warning(f"Factor de congestión {congestion_factor} es inválido, usando 1.0.")
                congestion_factor = 1.0
            
            if congestion_factor != 1.0: # Solo escalar si el factor es diferente de 1.0
                logger.info(f"Aplicando factor de congestión {congestion_factor} a la matriz de duraciones.")
                scaled_durations = [
                    [int(round(d * congestion_factor)) for d in row]
                    for row in durations
                ]
            else:
                logger.debug("Factor de congestión es 1.0, no se aplica escalado adicional a las duraciones.")
        elif durations:
            logger.debug("No se especificaron solver_params o congestion_factor, usando duraciones originales.")
        else:
            logger.debug("Matriz de duraciones no disponible, no se aplica factor de congestión.")
        logger.debug(f"Matrix data received: distances_shape={len(matrix_data.distances) if matrix_data.distances else 'None'}x{len(matrix_data.distances[0]) if matrix_data.distances and matrix_data.distances[0] else 'None'}, durations_shape={len(matrix_data.durations) if matrix_data.durations else 'None'}x{len(matrix_data.durations[0]) if matrix_data.durations and matrix_data.durations[0] else 'None'}, provider={provider_used}")
        # Acceder a 'from_cache' desde el diccionario de metadatos de MatrixResult
        from_cache = False # Default to False
        if matrix_data and hasattr(matrix_data, 'metadata') and isinstance(matrix_data.metadata, dict):
            from_cache = matrix_data.metadata.get('from_cache', False)
        logger.info(f"AdvancedSolverAdapter.solve_unified: matrices obtenidas de {provider_used}, desde cache: {from_cache}")
        has_tw = any(
            loc.get('time_window_start', 0) > 0 or loc.get('time_window_end', 24*3600) < 24*3600
            for loc in locations
        )
        has_pd = pickups_deliveries is not None and len(pickups_deliveries) > 0
        logger.info(f"AdvancedSolverAdapter.solve_unified: detección has_tw={has_tw}, has_pd={has_pd}, allow_skipping={allow_skipping}")
        # Ajustar optimization_profile con max_route_duration y penalizaciones de skipping
        if optimization_profile is None:
            optimization_profile = {}
        if max_route_duration:
            optimization_profile['max_route_duration'] = max_route_duration
        if allow_skipping and penalties:
            optimization_profile['skip_penalties'] = penalties

        if not has_tw and not has_pd:
            logger.info("AdvancedSolverAdapter.solve_unified: usando CVRPSolver")
            solver = CVRPSolver()
            solver.load_problem(
                distance_matrix=distances,
                duration_matrix=scaled_durations, # Usar matriz escalada
                locations=locations,
                vehicles=vehicles,
                optimization_profile=optimization_profile
            )
            sol = solver.solve(time_limit_seconds=time_limit_seconds)
        else: # This is for the VRPTW_ADVANCED case
            logger.info("AdvancedSolverAdapter.solve_unified: usando VRPTWSolver (avanzado)")
            pydantic_vehicles_list = [ExtendedVehicle(**v_data) for v_data in vehicles]
            if self.solver is None or not isinstance(self.solver, VRPTWSolver):
                 self.solver = VRPTWSolver(pydantic_vehicles_list=pydantic_vehicles_list)
            elif hasattr(self.solver, 'pydantic_vehicles_list'): # If solver exists, update its vehicle list if necessary
                 self.solver.pydantic_vehicles_list = pydantic_vehicles_list
            else: # Fallback if solver exists but has no pydantic_vehicles_list attr (should not happen with VRPTWSolver)
                 self.solver = VRPTWSolver(pydantic_vehicles_list=pydantic_vehicles_list)

            # Ensure depots has a default value if None
            final_depots = depots if depots is not None else [0]

            # Prepare starts_ends
            final_starts_ends = starts_ends # Initialize with provided value
            if not final_starts_ends and any(v_data.get('start_location_id') or v_data.get('end_location_id') for v_data in vehicles):
                logger.info("`starts_ends` no fue proporcionado o está incompleto. Intentando construir desde `start_location_id` y `end_location_id` en `vehicles`.")
                location_id_to_index = {loc['id']: i for i, loc in enumerate(locations)}
                calculated_starts_ends = []
                default_depot_index = final_depots[0]
                for v_idx, v_data_item in enumerate(vehicles):
                    start_id = v_data_item.get('start_location_id')
                    end_id = v_data_item.get('end_location_id')
                    
                    start_idx = location_id_to_index.get(start_id, default_depot_index)
                    # Default end to start_idx if end_id is None and start_id is valid, otherwise default to default_depot_index
                    if end_id is None:
                        end_idx = start_idx if start_id and location_id_to_index.get(start_id) is not None else default_depot_index
                    else:
                        end_idx = location_id_to_index.get(end_id, default_depot_index)

                    if start_id and location_id_to_index.get(start_id) is None:
                        logger.warning(f"ID de inicio '{start_id}' del vehículo '{v_data_item.get('id', f'veh_{v_idx}')}' no encontrado. Usando depot por defecto ({default_depot_index}).")
                    if end_id and location_id_to_index.get(end_id) is None:
                        logger.warning(f"ID de fin '{end_id}' del vehículo '{v_data_item.get('id', f'veh_{v_idx}')}' no encontrado. Usando depot por defecto ({default_depot_index}) o ubicación de inicio.")
                    calculated_starts_ends.append([start_idx, end_idx])
                final_starts_ends = calculated_starts_ends
                logger.info(f"Construido/actualizado `starts_ends` desde vehículos: {final_starts_ends}")
            elif not final_starts_ends: # If still not defined (not by payload, not by vehicle specifics)
                final_starts_ends = [[final_depots[0], final_depots[0]]] * len(vehicles)
                logger.info(f"`starts_ends` no definido, usando depots por defecto: {final_starts_ends}")

            # Prepare skills
            location_required_skills_list = [loc.get('required_skills') for loc in locations]
            vehicle_skills_list_for_load = [v_data.get('skills') for v_data in vehicles]

            # Prepare kwargs for load_problem
            load_problem_kwargs = {
                'allow_skipping_nodes': allow_skipping,
                'penalties': penalties,
                'location_required_skills': location_required_skills_list,
                'vehicle_skills': vehicle_skills_list_for_load,
                'optimization_profile': optimization_profile
            }
            load_problem_kwargs = {k: v for k, v in load_problem_kwargs.items() if v is not None}

            self.solver.load_problem(
                distance_matrix=distances,
                duration_matrix=scaled_durations, # Usar matriz escalada
                locations=locations,
                vehicles=vehicles,
                depots=final_depots,
                starts_ends=final_starts_ends,
                pickups_deliveries=pickups_deliveries,
                **load_problem_kwargs
            )
            sol = await self.solver.solve(time_limit_seconds=time_limit_seconds)

        logger.info(f"AdvancedSolverAdapter.solve_unified: solver devolvió estado={sol.status}")
        # Devolver la solución cruda/semi-procesada y los metadatos de la matriz por separado
        # El nombre del solver se añade para que el consumidor sepa cómo tratar 'sol'.
        return sol, {'matrix_provider': provider_used, 'from_cache': from_cache, 'solver_name': self.solver.__class__.__name__}
