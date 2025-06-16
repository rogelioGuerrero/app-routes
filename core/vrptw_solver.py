from __future__ import annotations
from pydantic import BaseModel, Field # Field might be used by RawSolutionData or other Pydantic models in this file

"""
Solver VRPTW con ventanas de tiempo, peso, volumen y pickup-delivery
"""
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
import time
import asyncio
import hashlib
import json
import os
import aiohttp

from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from .post_processors import add_suggested_departure_times

# Importar la instancia de caché existente
from services.distance_matrix.cache import matrix_cache

# --- Modelos Pydantic ---
from core.base_solver import BaseVRPSolver
from dataclasses import dataclass
from models.vrp_models import VRPSolutionStatus, VrpProblemData as VRPProblem, Vehicle, Location, RouteStop as ModelRouteStop # Renombrar para evitar colisión
from schemas.vrp_models import RouteStop as schemas_RouteStop, Route as schemas_Route, VRPTWSolution as schemas_VRPTWSolution, ExtendedVehicle

logger = logging.getLogger(__name__)

# --- Nueva función para obtener Polyline con Caché ---
def _generate_polyline_cache_key(coordinates: List[Tuple[float, float]], profile: str) -> str:
    """Genera una clave de caché única para un polyline."""
    key_data = json.dumps((coordinates, profile), sort_keys=False).encode('utf-8')
    return f"polyline_{hashlib.sha256(key_data).hexdigest()}"

async def get_ors_polyline_with_cache(
    session: aiohttp.ClientSession,
    coordinates: List[Tuple[float, float]], # Espera [(lng, lat), ...]
    profile: str, # e.g., "driving-car"
    api_key: str,
    cache_duration: int = 21600 # 6 horas, igual que MatrixCache
) -> Optional[str]:
    """
    Obtiene un polyline de ORS, utilizando un sistema de caché.
    Las coordenadas deben estar en formato [(lng1, lat1), (lng2, lat2), ...].
    """
    if len(coordinates) < 2:
        logger.debug("Menos de 2 coordenadas, no se puede generar polyline.")
        return None

    cache_key = _generate_polyline_cache_key(coordinates, profile)
    
    cached_polyline = matrix_cache.cache.get(cache_key)
    
    if cached_polyline is not None:
        logger.debug(f"Polyline encontrado en caché para la clave: {cache_key}")
        return cached_polyline

    logger.debug(f"Polyline no encontrado en caché para la clave: {cache_key}. Solicitando a ORS.")
    
    ors_directions_url = f"https://api.openrouteservice.org/v2/directions/{profile}/json"
    
    headers = {
        'Authorization': api_key,
        'Content-Type': 'application/json; charset=utf-8',
        'Accept': 'application/json',
    }
    payload = {
        "coordinates": coordinates,
        "preference": "fastest", 
        "instructions": "false",
        "geometry_simplify": "true"
    }
    
    try:
        async with session.post(ors_directions_url, json=payload, headers=headers, timeout=20) as response:
            if response.status == 200:
                data = await response.json()
                if 'routes' in data and len(data['routes']) > 0 and 'geometry' in data['routes'][0]:
                    polyline_string = data['routes'][0]['geometry']
                    matrix_cache.cache.set(cache_key, polyline_string, expire=cache_duration)
                    logger.debug(f"Polyline obtenido de ORS y guardado en caché: {cache_key}")
                    return polyline_string
                else:
                    logger.error(f"Respuesta inesperada de ORS Directions (polyline): {await response.text()}")
                    return None
            else:
                error_text = await response.text()
                logger.error(f"Error obteniendo polyline de ORS: {response.status} - {error_text}")
                return None
    except asyncio.TimeoutError:
        logger.error(f"Timeout obteniendo polyline de ORS para la clave: {cache_key}")
        return None
    except Exception as e:
        logger.error(f"Excepción obteniendo polyline de ORS para la clave {cache_key}: {e}", exc_info=True)
        return None


class RawSolutionData(BaseModel):
    solution: Optional[pywrapcp.Assignment] = Field(default=None, exclude=True)
    routing: pywrapcp.RoutingModel = Field(..., exclude=True)
    manager: pywrapcp.RoutingIndexManager = Field(..., exclude=True)
    data: Dict[str, Any]
    locations_input_models: List[Location] # models.Location (basic internal models)
    vehicles_input_models: List[Vehicle] # models.Vehicle (basic internal models)
    pydantic_vehicles_map: Dict[str, ExtendedVehicle] # Extended models for formatter
    depot_ids: Set[str]
    status_map: Dict[int, VRPSolutionStatus]
    status: int  # The raw integer status from OR-Tools solver
    current_status: VRPSolutionStatus

    class Config:
        arbitrary_types_allowed = True

RawSolutionData.model_rebuild() # Call rebuild after class definition


class VrpSolutionFormatter:
    # Esto es solo para referencia interna del formateador, la definición real está en schemas.vrp_models
    # from schemas.vrp_models import RouteStop as schemas_RouteStop (no se puede importar aquí directamente por circularidad potencial)
    # Se asumirá que schemas_RouteStop tiene: location_id, coords, arrival_time, departure_time, load, distance_from_previous
    def __init__(self, raw_solution: RawSolutionData): # Ensure this uses RawSolutionData
        self.raw_solution = raw_solution
        self.solution = raw_solution.solution
        self.routing = raw_solution.routing
        self.manager = raw_solution.manager
        self.data = raw_solution.data
        self.locations = raw_solution.locations_input_models
        self.pydantic_vehicles_map = raw_solution.pydantic_vehicles_map
        self.vehicles = raw_solution.vehicles_input_models
        self.depot_ids = raw_solution.depot_ids

    async def format(self) -> schemas_VRPTWSolution: # Convertido a async y tipo de retorno corregido
        d = self.data
        solution = self.solution
        status_code = self.routing.status()
        current_status = self.raw_solution.current_status

        if not solution or status_code in [
            routing_enums_pb2.RoutingSearchStatus.ROUTING_FAIL,
            routing_enums_pb2.RoutingSearchStatus.ROUTING_FAIL_TIMEOUT,
            routing_enums_pb2.RoutingSearchStatus.ROUTING_NOT_SOLVED
        ]:
            all_problem_location_ids = {loc.id for loc in self.locations if loc.id not in self.depot_ids}
            return schemas_VRPTWSolution(
                status=current_status.value,
                routes=[], total_distance=0, total_load=0, total_vehicles_used=0,
                total_weight=0, total_volume=0,
                metadata={'solver_status_code': status_code, 'dropped_node_ids': list(all_problem_location_ids)}
            )

        solution_routes = []
        overall_total_distance = 0
        overall_total_load_served = 0
        overall_total_weight_served = 0
        overall_total_volume_served = 0
        overall_vehicles_used = 0
        
        served_location_ids = set()

        time_dim = self.routing.GetDimensionOrDie('Time') if d.get('dur') else None
        capacity_dim = self.routing.GetDimensionOrDie('Capacity')
        weight_dim = self.routing.GetDimensionOrDie('Weight') if d.get('weight_caps') and d.get('weight_demands') else None
        volume_dim = self.routing.GetDimensionOrDie('Volume') if d.get('volume_caps') and d.get('volume_demands') else None

        ors_api_key = os.getenv('ORS_API_KEY')
        if not ors_api_key:
            logger.warning("ORS_API_KEY no encontrada. No se generarán polylines para las rutas.")

        # Usar una única sesión aiohttp para todas las llamadas a ORS
        async with aiohttp.ClientSession() as http_session:
            # Almacenaremos temporalmente los datos de cada ruta para construir el objeto Route después
            # junto con su polyline obtenido de forma asíncrona.
            # Cada elemento será un diccionario con los datos de la ruta y una tarea para su polyline.
            route_data_and_polyline_tasks = []

            for vehicle_idx in range(d['num_veh']):
                index = self.routing.Start(vehicle_idx)
                if self.routing.IsEnd(solution.Value(self.routing.NextVar(index))):
                    continue
                
                overall_vehicles_used += 1
                pydantic_vehicle = self.pydantic_vehicles_map.get(self.vehicles[vehicle_idx].id)
                
                route_stops_list = []
                route_total_distance = 0
                previous_index = -1
                previous_departure_time = 0

                while not self.routing.IsEnd(index):
                    problem_node_idx = self.manager.IndexToNode(index)
                    pydantic_loc = self.locations[problem_node_idx]
                    served_location_ids.add(pydantic_loc.id)

                    # The cumulative value at the *next* stop represents the quantity *after* the service at the current stop.
                    next_index = solution.Value(self.routing.NextVar(index))
                    load_after_stop = solution.Value(capacity_dim.CumulVar(next_index))
                    weight_after_stop = solution.Value(weight_dim.CumulVar(next_index)) if weight_dim else 0
                    volume_after_stop = solution.Value(volume_dim.CumulVar(next_index)) if volume_dim else 0
                    arrival_time_at_stop = int(solution.Min(time_dim.CumulVar(index))) if time_dim else 0
                    
                    actual_service_time = pydantic_loc.service_time or 0
                    departure_time_from_stop = int(arrival_time_at_stop + actual_service_time)

                    distance_from_previous = 0
                    travel_time_to_stop = 0
                    if previous_index != -1:
                        distance_from_previous = self.routing.GetArcCostForVehicle(previous_index, index, vehicle_idx)
                        if time_dim:
                            travel_time_to_stop = arrival_time_at_stop - previous_departure_time
                    
                    stop = ModelRouteStop(
                        location_id=pydantic_loc.id,
                        coords=pydantic_loc.coords,
                        arrival_time=arrival_time_at_stop,
                        departure_time=departure_time_from_stop,
                        load=load_after_stop,
                        weight=weight_after_stop,
                        volume=volume_after_stop,
                        distance_from_previous=distance_from_previous,
                        travel_time_to_stop=travel_time_to_stop,
                        service_time_at_stop=actual_service_time
                    )
                    route_stops_list.append(stop)
                    
                    route_total_distance += distance_from_previous
                    previous_index = index
                    previous_departure_time = departure_time_from_stop
                    index = solution.Value(self.routing.NextVar(index))

                # End of route loop, 'index' is now the end node for the vehicle
                end_node_idx = self.manager.IndexToNode(index)
                pydantic_loc = self.locations[end_node_idx]
                served_location_ids.add(pydantic_loc.id)
                
                arrival_time_at_stop = int(solution.Min(time_dim.CumulVar(index))) if time_dim else 0
                distance_from_previous = self.routing.GetArcCostForVehicle(previous_index, index, vehicle_idx)
                travel_time_to_stop = arrival_time_at_stop - previous_departure_time if time_dim else 0
                route_total_distance += distance_from_previous

                # Cumulative values at the end node represent the totals for the route
                route_total_load = solution.Value(capacity_dim.CumulVar(index))
                route_total_weight = solution.Value(weight_dim.CumulVar(index)) if weight_dim else 0
                route_total_volume = solution.Value(volume_dim.CumulVar(index)) if volume_dim else 0
                
                route_stops_list.append(ModelRouteStop(
                    location_id=pydantic_loc.id,
                    coords=pydantic_loc.coords,
                    arrival_time=arrival_time_at_stop,
                    departure_time=arrival_time_at_stop, # No service time at end depot
                    load=route_total_load,
                    weight=route_total_weight,
                    volume=route_total_volume,
                    distance_from_previous=distance_from_previous,
                    travel_time_to_stop=travel_time_to_stop,
                    service_time_at_stop=0
                ))

                route_total_time = arrival_time_at_stop

                # Determinar perfil del vehículo para ORS
                if pydantic_vehicle:
                    # Get vehicle ID safely
                    try:
                        vehicle_id = self.vehicles[vehicle_idx].id
                        logger.debug(f"VrpSolutionFormatter: pydantic_vehicle (type: {type(pydantic_vehicle)}) para ID '{vehicle_id}': {pydantic_vehicle!r}")
                        vehicle_profile_ors = pydantic_vehicle.ors_profile
                    except (IndexError, AttributeError) as e:
                        logger.warning(f"Error accessing vehicle attributes: {e}. Using default profile.")
                        vehicle_profile_ors = 'driving-car'
                else:
                    # Fallback si el modelo Pydantic del vehículo no se encontró (no debería ocurrir con datos válidos)
                    logger.warning("No pydantic_vehicle available. Using default profile.")
                    vehicle_profile_ors = 'driving-car'

                polyline_task = None
                if ors_api_key and route_stops_list: 
                    stop_coords_for_polyline = [
                        s.coords for s in route_stops_list if s.coords is not None
                    ]
                    if len(stop_coords_for_polyline) >= 2:
                        polyline_task = get_ors_polyline_with_cache(
                            http_session, 
                            stop_coords_for_polyline, 
                            vehicle_profile_ors, 
                            ors_api_key
                        )
                    else:
                        polyline_task = asyncio.create_task(asyncio.sleep(0, result=None))
                else:
                    polyline_task = asyncio.create_task(asyncio.sleep(0, result=None))

                # Convertir RouteStop de models a schemas para el almacenamiento temporal
                # Esto asume que schemas.vrp_models.RouteStop se llama schemas_RouteStop para evitar colisión de nombres
                # y que su estructura es: location_id, coords, arrival_time, departure_time, load, distance_from_previous
                route_stop_schemas_list = []
                for s_model in route_stops_list:
                    schema_stop_data = {
                        'location_id': s_model.location_id,
                        'coords': s_model.coords,
                        'arrival_time': s_model.arrival_time,
                        'departure_time': s_model.departure_time,
                        'load': s_model.load,
                        'weight': s_model.weight,
                        'volume': s_model.volume,
                        'distance_from_previous': s_model.distance_from_previous,
                        'travel_time_to_stop': s_model.travel_time_to_stop,
                        'service_time_at_stop': s_model.service_time_at_stop
                    }
                    route_stop_schemas_list.append(schema_stop_data)

                # Acumular la distancia total de la ruta actual
                overall_total_distance += route_total_distance

                route_data_and_polyline_tasks.append({
                    "vehicle_id": pydantic_vehicle.id if pydantic_vehicle else f"vehicle_{vehicle_idx}",
                    "stops_data": route_stop_schemas_list,
                    "total_distance": route_total_distance,
                    "total_load": route_total_load,
                    "total_weight": route_total_weight,
                    "total_volume": route_total_volume,
                    "total_time": route_total_time,
                    "polyline_task": polyline_task
                })

            # Esperar a que se completen todas las tareas de polyline
            polyline_results = await asyncio.gather(*[task['polyline_task'] for task in route_data_and_polyline_tasks])

            # Crear objetos Route con los datos y polylines obtenidos
            final_solution_routes = []
            for i, task_data in enumerate(route_data_and_polyline_tasks):
                polyline_result = polyline_results[i]
                stops = [schemas_RouteStop(**stop_data) for stop_data in task_data["stops_data"]]
                final_route = schemas_Route(
                    vehicle_id=task_data["vehicle_id"],
                    stops=stops,
                    total_distance=task_data["total_distance"],
                    total_load=task_data["total_load"],
                    total_weight=task_data["total_weight"],
                    total_volume=task_data["total_volume"],
                    total_time=task_data["total_time"],
                    polyline_ors=polyline_result
                )
                final_solution_routes.append(final_route)

        # Calcular totales de carga/peso/volumen servidos (solo entregas)
        pydantic_locations_map = {loc.id: loc for loc in self.locations}
        for loc_id in served_location_ids:
            if loc_id not in self.depot_ids:
                loc_model = pydantic_locations_map.get(loc_id)
                if loc_model:
                    if loc_model.demand and loc_model.demand > 0:
                        overall_total_load_served += loc_model.demand
                    if hasattr(loc_model, 'weight_demand') and loc_model.weight_demand and loc_model.weight_demand > 0:
                        overall_total_weight_served += loc_model.weight_demand
                    if hasattr(loc_model, 'volume_demand') and loc_model.volume_demand and loc_model.volume_demand > 0:
                        overall_total_volume_served += loc_model.volume_demand

        all_location_ids = {loc.id for loc in self.locations}
        non_depot_ids = all_location_ids - self.depot_ids
        dropped_node_ids = list(non_depot_ids - served_location_ids)

        return schemas_VRPTWSolution(
            status=current_status.value,
            routes=final_solution_routes, 
            total_distance=overall_total_distance,
            total_load=overall_total_load_served,
            total_vehicles_used=overall_vehicles_used,
            total_weight=overall_total_weight_served,
            total_volume=overall_total_volume_served,
            metadata={
                'solver_status_code': status_code,
                'dropped_node_ids': dropped_node_ids,
            }
        )


class VRPTWSolver(BaseVRPSolver):
    def __init__(self, pydantic_vehicles_list: List[ExtendedVehicle]):
        super().__init__()
        self.pydantic_vehicles_list = pydantic_vehicles_list # Store the list of extended vehicles
        self.manager: Optional[pywrapcp.RoutingIndexManager] = None
        self.routing: Optional[pywrapcp.RoutingModel] = None
        self.data: Dict[str, Any] = {}
        self.solver_params: Dict[str, Any] = {}
        self.status_map = {
            routing_enums_pb2.RoutingSearchStatus.ROUTING_NOT_SOLVED: VRPSolutionStatus.NO_SOLUTION_FOUND,
            routing_enums_pb2.RoutingSearchStatus.ROUTING_SUCCESS: VRPSolutionStatus.FEASIBLE,
            routing_enums_pb2.RoutingSearchStatus.ROUTING_FAIL: VRPSolutionStatus.ERROR_SOLVING_PROBLEM,
            routing_enums_pb2.RoutingSearchStatus.ROUTING_FAIL_TIMEOUT: VRPSolutionStatus.ERROR_SOLVING_PROBLEM,
            routing_enums_pb2.RoutingSearchStatus.ROUTING_INVALID: VRPSolutionStatus.INVALID_INPUT,
            routing_enums_pb2.RoutingSearchStatus.ROUTING_OPTIMAL: VRPSolutionStatus.OPTIMAL
        }
        self.locations: List[Location] = []
        self.vehicles: List[Dict[str, Any]] = []
        self.allow_skipping_nodes = False
        self.transit_callbacks = [] # Para mantener referencias a los callbacks y evitar GC
        self.pydantic_locations_map: Dict[str, Location] = {}
        self.pydantic_vehicles_map: Dict[str, Vehicle] = {} # For models.vrp_models.Vehicle (basic internal models)
        self.location_id_to_index: Dict[str, int] = {}
        logger.debug(f"VRPTWSolver.__init__: Received pydantic_vehicles (count: {len(self.pydantic_vehicles_map) if self.pydantic_vehicles_map else 0}). Types: {[type(v) for v in self.pydantic_vehicles_map.values()] if self.pydantic_vehicles_map else []}")

    def load_problem(
        self,
        distance_matrix: List[List[float]],
        duration_matrix: List[List[float]],
        locations: List[Dict[str, Any]],
        vehicles: List[Dict[str, Any]],
        depots: Optional[List[int]] = None,
        starts_ends: Optional[List[List[int]]] = None,
        pickups_deliveries: Optional[List[List[str]]] = None,
        **kwargs
    ) -> None:
        logger.debug(f"VRPTWSolver.load_problem: Received kwargs keys: {list(kwargs.keys())}")
        n = len(distance_matrix)
        # Store original Pydantic models for access in _format_solution
        self.pydantic_locations_map = {loc['id']: Location(**loc) for loc in locations}
        self.pydantic_vehicles_map = {veh['id']: Vehicle(**veh) for veh in vehicles} # Populate with models.Vehicle
        self.location_id_to_index = {loc['id']: i for i, loc in enumerate(locations)}

        self.locations = [Location(**loc) for loc in locations]
        self.vehicles = vehicles
        self.depot_ids = {loc['id'] for i, loc in enumerate(locations) if i in (depots or [])}
        self.allow_skipping_nodes = kwargs.get('allow_skipping_nodes', False)
        self.solver_params = kwargs.get('solver_params', {})
        self.penalties = kwargs.get('penalties', None) # Almacenar penalizaciones
        self.location_required_skills = kwargs.get('location_required_skills', None)
        self.vehicle_skills = kwargs.get('vehicle_skills', None)

        dist = [[int(round(d)) for d in row] for row in distance_matrix]
        dur = [[int(round(d)) for d in row] for row in duration_matrix] if duration_matrix else None

        # --- Dimensiones de Capacidad ---
        # Capacidad/Demanda Genérica
        caps = [int(round(v.get('capacity', 0))) for v in self.vehicles]
        demands = [int(round(loc.demand)) for loc in self.locations]

        # Capacidad/Demanda de Peso
        weight_caps = [int(round(v.get('weight_capacity', 0))) for v in self.vehicles]
        weight_demands = [int(round(getattr(loc, 'weight_demand', 0))) for loc in self.locations]

        # Capacidad/Demanda de Volumen
        volume_caps = [int(round(v.get('volume_capacity', 0))) for v in self.vehicles]
        volume_demands = [int(round(getattr(loc, 'volume_demand', 0))) for loc in self.locations]

        tw = [(loc.time_window_start, loc.time_window_end) for loc in self.locations]

        location_id_to_index = self.location_id_to_index
        pd_pairs = []
        if pickups_deliveries:
            for pickup_id, delivery_id in pickups_deliveries:
                pickup_idx = location_id_to_index.get(pickup_id)
                delivery_idx = location_id_to_index.get(delivery_id)
                if pickup_idx is not None and delivery_idx is not None:
                    pd_pairs.append([pickup_idx, delivery_idx])

        # Extract starts_indices and ends_indices from kwargs, expected from AdvancedSolverAdapter
        s_indices = kwargs.get('starts_indices')
        e_indices = kwargs.get('ends_indices')
        starts_ends = kwargs.get('starts_ends')

        num_vehicles = len(vehicles)

        if starts_ends and len(starts_ends) == num_vehicles:
            logger.debug(f"Using 'starts_ends' from kwargs to define start/end nodes: {starts_ends}")
            starts_indices = [item[0] for item in starts_ends]
            ends_indices = [item[1] for item in starts_ends]
        else:
            logger.warning(f"'starts_ends' not found or invalid in kwargs (expected len {num_vehicles}, got {len(starts_ends) if starts_ends else 'None'}). Falling back to default depot assignment.")
            # Fallback logic based on vehicle's home depot, which is more robust
            starts_indices = [self.location_id_to_index.get(v.get('start_location_id'), depots[i % len(depots)]) for i, v in enumerate(self.vehicles)]
            ends_indices = [self.location_id_to_index.get(v.get('end_location_id'), depots[i % len(depots)]) for i, v in enumerate(self.vehicles)]
            logger.warning(f"Fallback starts_indices: {starts_indices}")
            logger.warning(f"Fallback ends_indices: {ends_indices}")

        # Determine the default node (typically the first depot, or 0 if no depots specified)
        # This default_node is used if starts/ends information is missing or incomplete.
        # Skipping explicit skill checks for nodes identified as depots.
        effective_depots = depots if depots is not None else [0]
        default_node_for_vehicles = effective_depots[0] if effective_depots else 0

        if s_indices is None:
            logger.error("'starts_indices' not found in kwargs for VRPTWSolver.load_problem. This is unexpected.")
            if starts_ends: # Fallback to starts_ends if available
                s_indices = [pair[0] for pair in starts_ends]
            else: # Fallback to default_node_for_vehicles for all vehicles
                s_indices = [default_node_for_vehicles] * len(vehicles)
            logger.warning(f"Falling back to generated starts_indices: {s_indices}")

        if e_indices is None:
            logger.error("'ends_indices' not found in kwargs for VRPTWSolver.load_problem. This is unexpected.")
            if starts_ends: # Fallback to starts_ends if available
                e_indices = [pair[1] for pair in starts_ends]
            else: # Fallback to default_node_for_vehicles for all vehicles
                e_indices = [default_node_for_vehicles] * len(vehicles)
            logger.warning(f"Falling back to generated ends_indices: {e_indices}")
        
        # Ensure starts_ends also has a robust default if not provided
        final_starts_ends = starts_ends
        if final_starts_ends is None:
            final_starts_ends = [[s_indices[i], e_indices[i]] for i in range(len(vehicles))]
            logger.warning(f"'starts_ends' was None, constructed from s_indices and e_indices: {final_starts_ends}")

        self.data = {
            'n': n,
            'num_veh': len(vehicles),
            'depot': default_node_for_vehicles, # Primary depot for some OR-Tools constraints
            'dist': dist,
            'dur': dur,
            'caps': caps,
            'demands': demands,
            'weight_caps': weight_caps,
            'weight_demands': weight_demands,
            'volume_caps': volume_caps,
            'volume_demands': volume_demands,
            'tw': tw,
            'pd': pd_pairs,
            'depots': effective_depots, # List of all depot indices
            'starts_ends': final_starts_ends, # List of [start, end] pairs for each vehicle
            'starts': s_indices, # List of start nodes for each vehicle
            'ends': e_indices    # List of end nodes for each vehicle
        }
        logger.debug(f"VRPTWSolver.load_problem: self.data keys after assignment: {list(self.data.keys()) if self.data else 'None'}")

        # Create Routing Index Manager
        # Ensure starts and ends are populated, which should be guaranteed by prior logic
        if not isinstance(self.data.get('starts'), list) or not isinstance(self.data.get('ends'), list):
            logger.error("Critical error: 'starts' or 'ends' not properly initialized as lists in self.data.")
            # This indicates a flaw in the data preparation logic before this point.
            # Raising an error or returning early might be appropriate depending on desired robustness.
            # For now, we'll let it proceed, but it will likely fail in RoutingIndexManager construction.
            # A more robust solution would raise ValueError here.

        self.manager = pywrapcp.RoutingIndexManager(
            self.data['n'],
            self.data['num_veh'],
            self.data['starts'],
            self.data['ends']
        )

        # Create Routing Model
        self.routing = pywrapcp.RoutingModel(self.manager)
        logger.debug("VRPTWSolver.load_problem: RoutingIndexManager and RoutingModel created.")

    def _add_constraints(self) -> None:
        if not self.routing or not self.manager or not self.data:
            logger.error("_add_constraints called before problem loaded or manager/routing initialized.")
            raise ValueError("Problem data, manager, or routing model not initialized.")

        # Clear previous callbacks to avoid issues if called multiple times (though not typical)
        self.transit_callbacks.clear()

        # --- Transit Callbacks (Distance and Duration) ---
        def distance_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return self.data['dist'][from_node][to_node]

        dist_callback_index = self.routing.RegisterTransitCallback(distance_callback)
        self.transit_callbacks.append(dist_callback_index) # Keep reference
        self.routing.SetArcCostEvaluatorOfAllVehicles(dist_callback_index)

        if self.data.get('dur'):
            def time_callback_with_service(from_index, to_index):
                """Returns the total time (travel + service) between two locations."""
                from_node = self.manager.IndexToNode(from_index)
                to_node = self.manager.IndexToNode(to_index)
                
                # Travel time from the duration matrix
                travel_time = self.data['dur'][from_node][to_node]
                
                # Service time at the 'from' node.
                # Depots have 0 service time, so this is safe for the first arc.
                service_time = self.locations[from_node].service_time or 0
                
                return int(service_time + travel_time)

            time_callback_index = self.routing.RegisterTransitCallback(time_callback_with_service)
            self.transit_callbacks.append(time_callback_index) # Keep reference

            # Determine a suitable horizon for the time dimension (max duration of a route)
            # Use the maximum time window end provided, or default to 24 hours.
            max_tw_end = 0
            for loc_obj in self.locations:
                if loc_obj.time_window_end is not None and loc_obj.time_window_end > max_tw_end:
                    max_tw_end = loc_obj.time_window_end
            
            time_dimension_capacity = int(max(max_tw_end, 24 * 3600)) # Ensure at least 24 hours if TWs are small or not set

            self.routing.AddDimension(
                time_callback_index,
                time_dimension_capacity,  # Slack (waiting time) - Allow waiting up to horizon
                time_dimension_capacity, # Horizon
                False,  # Start cumul to zero for all vehicles
                'Time'
            )
            time_dimension = self.routing.GetDimensionOrDie('Time')
            
            # Add time window constraints for each location.
            # This applies to all nodes, ensuring that if a depot is visited mid-route,
            # it also respects its time window.
            for i, loc_obj in enumerate(self.locations):
                if loc_obj.time_window_start is not None and loc_obj.time_window_end is not None:
                    node_idx_in_manager = self.manager.NodeToIndex(i)
                    time_dimension.CumulVar(node_idx_in_manager).SetRange(
                        int(loc_obj.time_window_start),
                        int(loc_obj.time_window_end)
                    )

            # Add time window constraints for each vehicle's START node.
            # This is the critical part to ensure routes start within the depot's time window.
            for vehicle_id in range(self.data['num_veh']):
                start_node_idx = self.data['starts'][vehicle_id]
                start_loc_obj = self.locations[start_node_idx]
                
                if start_loc_obj.time_window_start is not None and start_loc_obj.time_window_end is not None:
                    index = self.routing.Start(vehicle_id)
                    time_dimension.CumulVar(index).SetRange(
                        int(start_loc_obj.time_window_start),
                        int(start_loc_obj.time_window_end)
                    )
            
            # Apply costs for slack (waiting time) and span (total route duration).
            # Additionally, maximise the start time of each vehicle to encourage late departures,
            # reducing waiting at the first customer while still respecting all windows.
            time_slack_cost = self.solver_params.get("time_slack_cost_coefficient", 0)
            time_span_cost = self.solver_params.get("time_span_cost_coefficient", 0)

            if time_slack_cost > 0 or time_span_cost > 0:
                for v_idx in range(self.data['num_veh']):
                    if time_slack_cost > 0:
                        time_dimension.SetSlackCostCoefficientForVehicle(time_slack_cost, v_idx)
                        logger.debug(f"Vehicle {v_idx}: Set time_slack_cost_coefficient to {time_slack_cost}")
                    if time_span_cost > 0:
                        time_dimension.SetSpanCostCoefficientForVehicle(time_span_cost, v_idx)
                        logger.debug(f"Vehicle {v_idx}: Set time_span_cost_coefficient to {time_span_cost}")

                    # Encourage vehicles to depart as late as feasible (just-in-time) by
                    # maximising the start cumul variable during the finalization phase.
                    start_index = self.routing.Start(v_idx)
                    self.routing.AddVariableMaximizedByFinalizer(time_dimension.CumulVar(start_index))

        # --- Capacity Dimensions (Generic, Weight, Volume) ---
        def demand_callback(from_index):
            from_node = self.manager.IndexToNode(from_index)
            return self.data['demands'][from_node]
        
        demand_callback_index = self.routing.RegisterUnaryTransitCallback(demand_callback)
        self.transit_callbacks.append(demand_callback_index) # Keep reference
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # Slack for capacity
            self.data['caps'],  # Vehicle capacities
            True,  # Start cumul to zero
            'Capacity'
        )

        if self.data.get('weight_demands') and self.data.get('weight_caps'):
            def weight_demand_callback(from_index):
                from_node = self.manager.IndexToNode(from_index)
                return self.data['weight_demands'][from_node]
            weight_demand_callback_index = self.routing.RegisterUnaryTransitCallback(weight_demand_callback)
            self.transit_callbacks.append(weight_demand_callback_index)
            self.routing.AddDimensionWithVehicleCapacity(
                weight_demand_callback_index, 0, self.data['weight_caps'], True, 'Weight'
            )

        if self.data.get('volume_demands') and self.data.get('volume_caps'):
            def volume_demand_callback(from_index):
                from_node = self.manager.IndexToNode(from_index)
                return self.data['volume_demands'][from_node]
            volume_demand_callback_index = self.routing.RegisterUnaryTransitCallback(volume_demand_callback)
            self.transit_callbacks.append(volume_demand_callback_index)
            self.routing.AddDimensionWithVehicleCapacity(
                volume_demand_callback_index, 0, self.data['volume_caps'], True, 'Volume'
            )

        # --- Pickup and Delivery Constraints ---
        if self.data.get('pd'):
            time_dim_for_pd = self.routing.GetDimensionOrDie('Time') if self.data.get('dur') else None
            
            for pickup_idx, delivery_idx in self.data['pd']:
                pickup_node_manager_idx = self.manager.NodeToIndex(pickup_idx)
                delivery_node_manager_idx = self.manager.NodeToIndex(delivery_idx)

                # Core constraints: same vehicle and time precedence.
                self.routing.solver().Add(
                    self.routing.VehicleVar(pickup_node_manager_idx) == self.routing.VehicleVar(delivery_node_manager_idx)
                )
                if time_dim_for_pd:
                    self.routing.solver().Add(
                        time_dim_for_pd.CumulVar(pickup_node_manager_idx) <= time_dim_for_pd.CumulVar(delivery_node_manager_idx)
                    )
                
                # Link active state: if one is served, the other must be too.
                # This is crucial for making them optional *as a pair* when disjunctions are used.
                if self.allow_skipping_nodes:
                    self.routing.solver().Add(
                        self.routing.ActiveVar(pickup_node_manager_idx) == self.routing.ActiveVar(delivery_node_manager_idx)
                    ) 

        # --- Node Skipping (Penalties) ---
        if self.allow_skipping_nodes:
            base_penalty = 1_000_000
            base_penalty_pd = 2_000_000  # Higher penalty for P&D pairs, can be customized

            pd_pickup_nodes = {pair[0] for pair in self.data.get('pd', [])}
            all_pd_nodes = {node for pair in self.data.get('pd', []) for node in pair}

            for i in range(self.data['n']):
                if i in self.data.get('depots', []):
                    continue

                node_manager_idx = self.manager.NodeToIndex(i)

                # Handle P&D pairs by adding a disjunction to the pickup node.
                # The linked active status (defined in P&D constraints) ensures the whole pair is dropped.
                if i in pd_pickup_nodes:
                    # TODO: Get penalty from input data for the pair.
                    penalty_for_pair = base_penalty_pd
                    self.routing.AddDisjunction([node_manager_idx], penalty_for_pair)

                # Handle standalone nodes (nodes not in any P&D pair).
                elif i not in all_pd_nodes:
                    loc_id = self.locations[i].id
                    penalty_value = base_penalty
                    if self.penalties and isinstance(self.penalties, dict) and loc_id in self.penalties:
                        penalty_value = self.penalties[loc_id]
                    elif isinstance(self.penalties, (int, float)):
                        penalty_value = self.penalties
                    
                    penalty_value = max(0, int(penalty_value))
                    self.routing.AddDisjunction([node_manager_idx], penalty_value)

        # --- Skill Constraints ---
        if self.location_required_skills and self.vehicle_skills:
            num_nodes = self.data['n']
            num_vehicles = self.data['num_veh']

            for node_idx_problem in range(num_nodes): # Iterate through problem node indices (0 to n-1)
                # Depots are generally not subject to skill constraints in the same way as customer locations.
                # If a depot had required skills, it would imply vehicles starting/ending there must possess them.
                # This logic assumes depot compatibility is handled by vehicle definitions or start/end constraints.
                # Skipping explicit skill checks for nodes identified as depots.
                if node_idx_problem in self.data.get('depots', []):
                    continue
                    
                # Get required skills for the current location (problem node index)
                # self.location_required_skills is a list where index corresponds to problem node_idx
                required_skills_for_loc = self.location_required_skills[node_idx_problem]
                
                if not required_skills_for_loc: # No skills required for this location, any vehicle can service
                    continue

                allowed_vehicles_for_this_node = []
                for veh_idx in range(num_vehicles):
                    # self.vehicle_skills is a list where index corresponds to vehicle_idx
                    vehicle_has_skills = self.vehicle_skills[veh_idx]
                    
                    can_service_node = True
                    if not vehicle_has_skills: # Vehicle has no skills defined
                        can_service_node = False
                    else:
                        for req_skill in required_skills_for_loc:
                            if req_skill not in vehicle_has_skills:
                                can_service_node = False
                                break
                    
                    if can_service_node:
                        allowed_vehicles_for_this_node.append(veh_idx)
                
                # Get the OR-Tools internal index for the node (manager index)
                or_tools_node_manager_idx = self.manager.NodeToIndex(node_idx_problem)

                if not allowed_vehicles_for_this_node:
                    # No vehicle possesses all required skills for this node.
                    # If node skipping (disjunction) is allowed and a penalty is set, OR-Tools might drop this node.
                    # If the node must be visited (no disjunction or infinite penalty), this will make the problem infeasible.
                    # Setting an empty list of allowed vehicles effectively forbids any vehicle from servicing this node.
                    logger.warning(
                        f"Node '{self.locations[node_idx_problem].id}' (problem idx {node_idx_problem}) requires skills {required_skills_for_loc}, "
                        f"but no vehicle has all of them. This node cannot be serviced by any vehicle."
                    )
                    self.routing.SetAllowedVehiclesForIndex([], or_tools_node_manager_idx)
                else:
                    self.routing.SetAllowedVehiclesForIndex(allowed_vehicles_for_this_node, or_tools_node_manager_idx)
                    logger.debug(
                        f"Node '{self.locations[node_idx_problem].id}' (problem idx {node_idx_problem}) requires skills {required_skills_for_loc}. "
                        f"Allowed vehicle indices: {allowed_vehicles_for_this_node}"
                    )

        logger.debug("_add_constraints completed.")

    async def solve(self, time_limit_seconds=30) -> schemas_VRPTWSolution: 
        logger.debug(f"VRPTWSolver.solve: self.pydantic_vehicles (count: {len(self.pydantic_vehicles_map) if self.pydantic_vehicles_map else 0}). Types: {[type(v) for v in self.pydantic_vehicles_map.values()] if self.pydantic_vehicles_map else []}") 
        if not self.data or not self.manager:
            raise ValueError("Problem data not loaded. Call load_problem first.")

        # --- Lógica de Restricciones ---
        self._add_constraints()

        # --- Búsqueda de Solución ---
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        params.time_limit.FromSeconds(time_limit_seconds)

        if self.solver_params.get("use_metaheuristic", False):
            params.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

        start_time = time.time()
        solution_assignment = self.routing.SolveWithParameters(params)
        
        end_time = time.time()
        logger.info(f"VRPTWSolver.solve: Resolución completada en {end_time - start_time:.2f} segundos.")
        
        status_code = self.routing.status()
        current_status = self.status_map.get(status_code, VRPSolutionStatus.ERROR_SOLVING_PROBLEM)

        # Crear el mapa de vehículos extendidos para el formateador
        extended_vehicles_map_for_formatter: Dict[str, schemas.ExtendedVehicle] = {
            v.id: v for v in self.pydantic_vehicles_list # Use the stored list of schemas.ExtendedVehicle
        }

        raw_solution_data = RawSolutionData(
            solution=solution_assignment,
            routing=self.routing,
            manager=self.manager,
            data=self.data,
            locations_input_models=self.locations,
            vehicles_input_models=self.vehicles,
            pydantic_vehicles_map=extended_vehicles_map_for_formatter, # Usar el mapa con ExtendedVehicle
            depot_ids=self.depot_ids,
            status_map=self.status_map,
            status=status_code,
            current_status=current_status
        )
        
        formatter = VrpSolutionFormatter(raw_solution_data)
        formatted_solution = await formatter.format()

        # Añadir tiempo de resolución a metadatos
        if formatted_solution.metadata is None:
            formatted_solution.metadata = {}
        formatted_solution.metadata['solver_time_seconds'] = round(end_time - start_time, 2)

        # Post-process: desplazar salidas para minimizar esperas.
        try:
            add_suggested_departure_times(
                formatted_solution,
                self.locations,
                self.location_id_to_index,
                self.data.get('dur', [])
            )
        except Exception as e:
            logger.warning(f"shift_departure_times post-processing failed: {e}")
        
        return formatted_solution

