import logging
from typing import List, Dict, Any, Optional, Union

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from schemas.vrp_models import CVRPSolution, Route, RouteStop, VRPSolutionStatus, Location, Vehicle
from core.base_solver import BaseVRPSolver

logger = logging.getLogger(__name__)

class CVRPSolver(BaseVRPSolver):
    """Solucionador para el Problema de Enrutamiento de Vehículos con Capacidad (CVRP)."""

    def __init__(self):
        """Inicializa el CVRPSolver."""
        super().__init__()
        self.manager = None
        self.routing = None
        self.assignment = None
        self.search_parameters = None
        self.logger = logger
        self.data = {}

    def load_problem(
        self,
        distance_matrix: List[List[Union[int, float]]],
        locations: List[Union[Dict[str, Any], Location]],
        vehicles: List[Union[Dict[str, Any], Vehicle]],
        duration_matrix: Optional[List[List[Union[int, float]]]] = None,
        optimization_profile: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Carga los datos del problema y prepara el modelo de enrutamiento."""
        super().load_problem(
            distance_matrix, locations, vehicles, duration_matrix, optimization_profile, **kwargs
        )
        
        self._prepare_or_tools_data(**kwargs)

        # Create the routing index manager.
        depots = self.data['depots']
        num_vehicles = self.data['num_vehicles']
        
        # OR-Tools requiere que los depots, starts y ends sean listas de la misma longitud que el número de vehículos.
        starts = self.data['vehicle_starts']
        ends = self.data['vehicle_ends']

        self.manager = pywrapcp.RoutingIndexManager(
            self.data['num_locations'],
            num_vehicles,
            starts,
            ends
        )
        # Create Routing Model.
        self.routing = pywrapcp.RoutingModel(self.manager)

        self._add_dimensions()
        self._add_constraints_and_settings()
        
        self.logger.info("Modelo de enrutamiento CVRP creado y cargado.")

    def _prepare_or_tools_data(self, **kwargs):
        """Prepara el diccionario 'data' para OR-Tools para un problema CVRP simple."""

        # OR-Tools requiere que las demandas y capacidades sean enteros.
        demands = [int(loc.demand or 0) for loc in self.locations]
        vehicle_capacities = [int(v.capacity or 0) for v in self.vehicles]

        # Para un CVRP simple, el depósito es cualquier ubicación con demanda cero.
        depots = [i for i, loc in enumerate(self.locations) if loc.demand == 0]
        if not depots:
            depots = [0]  # Si no se encuentra, se asume que el primer índice es el depósito.
        default_depot = depots[0]

        num_vehicles = len(self.vehicles)

        # En un CVRP simple, todos los vehículos empiezan y terminan en el mismo depósito.
        vehicle_starts = [default_depot] * num_vehicles
        vehicle_ends = [default_depot] * num_vehicles

        is_depot_list = [i in depots for i in range(len(self.locations))]

        self.data = {
            'distance_matrix': self.distance_matrix,
            'num_locations': len(self.locations),
            'demands': demands,
            'num_vehicles': num_vehicles,
            'vehicle_capacities': vehicle_capacities,
            'depots': depots,
            'vehicle_starts': vehicle_starts,
            'vehicle_ends': vehicle_ends,
            'is_depot': is_depot_list,
            'allow_skipping_nodes': self.optimization_profile.get('allow_skipping_nodes', False),
            'default_penalty_for_dropping_nodes': self.optimization_profile.get('default_penalty_for_dropping_nodes', 100000)
        }
        if hasattr(self, 'duration_matrix') and self.duration_matrix:
            self.data['duration_matrix'] = self.duration_matrix

    def _create_distance_callback(self):
        def distance_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            try:
                return int(self.data['distance_matrix'][from_node][to_node])
            except Exception:
                self.logger.error(f"Error al acceder a distance_matrix para from_node {from_node}, to_node {to_node}")
                return 0 # Fallback, debería investigarse si ocurre
        return distance_callback

    def _create_demand_callback(self, demands):
        def demand_callback(from_index):
            from_node = self.manager.IndexToNode(from_index)
            try:
                return int(demands[from_node])
            except Exception:
                self.logger.error(f"Error al acceder a demands para from_node {from_node}")
                return 0 # Fallback, debería investigarse si ocurre
        return demand_callback

    def _create_duration_callback(self):
        """Crea y devuelve el callback de tránsito para las duraciones."""
        def duration_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            try:
                # Durations are typically in seconds. Service times are added by the dimension itself.
                return int(self.data['duration_matrix'][from_node][to_node])
            except KeyError: # Específicamente para KeyErrors si la matriz no está completa
                self.logger.error(f"KeyError al acceder a duration_matrix para from_node {from_node}, to_node {to_node}. Verifique la estructura de la matriz.")
                return 0 # Fallback
            except IndexError: # Específicamente para IndexErrors si los nodos están fuera de rango
                self.logger.error(f"IndexError al acceder a duration_matrix para from_node {from_node}, to_node {to_node}. Verifique los índices de nodos.")
                return 0 # Fallback
            except Exception as e:
                self.logger.error(f"Error inesperado ({type(e).__name__}) al acceder a duration_matrix para from_node {from_node}, to_node {to_node}: {e}")
                return 0 # Fallback general
        return duration_callback

    def _add_dimensions(self):
        transit_callback_index = self.routing.RegisterTransitCallback(self._create_distance_callback())
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Dimensión de tiempo (si duration_matrix está presente)
        if 'duration_matrix' in self.data and self.data['duration_matrix']:
            duration_callback_index = self.routing.RegisterTransitCallback(self._create_duration_callback())
            time_dimension_name = "Time"
            self.routing.AddDimension(
                duration_callback_index,
                0,  # holgura (slack)
                self.optimization_profile.get('max_route_duration', 86400),  # tiempo máximo por ruta
                False,  # no empezar acumulado a cero para el tiempo
                time_dimension_name
            )
            self.time_dimension = self.routing.GetDimensionOrDie(time_dimension_name)
            self.logger.info(f"Dimensión de tiempo '{time_dimension_name}' agregada.")
        else:
            self.logger.info("No se agregó dimensión de tiempo (no se proporcionó duration_matrix).")

        if any(d > 0 for d in self.data['demands']):
            demand_callback_index = self.routing.RegisterUnaryTransitCallback(self._create_demand_callback(self.data['demands']))
            self.routing.AddDimensionWithVehicleCapacity(
                demand_callback_index, 0, self.data['vehicle_capacities'], True, 'Capacity'
            )

    def _add_constraints_and_settings(self):
        if self.data.get('allow_skipping_nodes'):
            penalty = self.data.get('default_penalty_for_dropping_nodes', 100000)
            for node_idx in range(self.data['num_locations']):
                if not self.data['is_depot'][node_idx]:
                    self.routing.AddDisjunction([self.manager.NodeToIndex(node_idx)], penalty)

        if self.data.get('vehicle_skills') and self.data.get('location_required_skills'):
            for loc_idx, required_skills in enumerate(self.data['location_required_skills']):
                if not self.data['is_depot'][loc_idx] and required_skills:
                    allowed_vehicles = [
                        veh_idx for veh_idx, vehicle_skills in enumerate(self.data['vehicle_skills'])
                        if all(req in vehicle_skills for req in required_skills)
                    ]
                    if allowed_vehicles:
                        self.routing.SetAllowedVehiclesForIndex(allowed_vehicles, self.manager.NodeToIndex(loc_idx))
                    else:
                        self.routing.AddDisjunction([self.manager.NodeToIndex(loc_idx)], self.data['default_penalty_for_dropping_nodes'])

    def solve(self, time_limit_seconds: int = 30, **kwargs) -> CVRPSolution:
        super().solve(time_limit_seconds, **kwargs)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        profile = self.optimization_profile or {}
        
        strategy = profile.get("first_solution_strategy", "AUTOMATIC").upper()
        if hasattr(routing_enums_pb2.FirstSolutionStrategy, strategy):
            search_parameters.first_solution_strategy = getattr(routing_enums_pb2.FirstSolutionStrategy, strategy)

        metaheuristic = profile.get("local_search_metaheuristic", "AUTOMATIC").upper()
        if hasattr(routing_enums_pb2.LocalSearchMetaheuristic, metaheuristic):
            search_parameters.local_search_metaheuristic = getattr(routing_enums_pb2.LocalSearchMetaheuristic, metaheuristic)

        search_parameters.time_limit.FromSeconds(time_limit_seconds)
        
        self.logger.info("Iniciando la resolución del problema CVRP...")
        self.assignment = self.routing.SolveWithParameters(search_parameters)
        
        self._solution = self._format_solution()
        return self._solution

    def _format_solution(self) -> CVRPSolution:
        """Formatea la solución de OR-Tools en el formato CVRPSolution."""
        if not self.assignment:
            self.logger.warning("Formateo de solución: No hay asignación (assignment) disponible.")
            return CVRPSolution(status=VRPSolutionStatus.NO_SOLUTION_FOUND.value, routes=[], total_distance=0, total_load=0, total_vehicles_used=0)

        status_code = self._get_solver_status()
        if status_code not in [VRPSolutionStatus.OPTIMAL.value, VRPSolutionStatus.FEASIBLE.value]:
            self.logger.warning(f"Formateo de solución: Estado del solver no es OPTIMAL ni FEASIBLE: {status_code}")
            return CVRPSolution(status=status_code, routes=[], total_distance=0, total_load=0, total_vehicles_used=0)

        solution_total_distance = self.assignment.ObjectiveValue() # Esto es si el costo objetivo es solo distancia
        solution_total_load = 0
        routes_data = []
        
        capacity_dimension = self.routing.GetDimensionOrDie('Capacity')
        time_dimension = None
        if hasattr(self, 'time_dimension') and self.time_dimension:
            time_dimension = self.time_dimension
        else:
            try:
                time_dimension = self.routing.GetDimensionOrDie("Time")
                self.logger.info("Dimensión 'Time' obtenida directamente del routing model para formateo.")
            except Exception:
                self.logger.info("No se pudo obtener la dimensión 'Time' para formateo.")

        for vehicle_idx in range(self.data['num_vehicles']):
            index = self.routing.Start(vehicle_idx)
            # Si la ruta está vacía (solo inicio y fin son el mismo y no hay paradas intermedias para este vehículo)
            if self.routing.IsEnd(self.assignment.Value(self.routing.NextVar(index))):
                continue

            current_route_stops = []
            current_route_distance = 0
            current_route_load = 0
            current_route_total_time = 0 # Para la duración de la ruta
            previous_routing_index = index # Usar un nombre diferente para el índice de enrutamiento

            while not self.routing.IsEnd(index):
                node_manager_index = self.manager.IndexToNode(index)
                location_obj = self.locations[node_manager_index]
                
                # Distancia desde la parada anterior a la actual
                # GetArcCostForVehicle usa el índice de enrutamiento, no el índice del nodo del manager
                arc_distance = self.routing.GetArcCostForVehicle(previous_routing_index, index, vehicle_idx)
                
                # Carga acumulada en esta parada
                cumulative_load = self.assignment.Value(capacity_dimension.CumulVar(index))

                arrival_time_val = None
                departure_time_val = None
                service_time = location_obj.service_time or 0

                if time_dimension:
                    arrival_time_val = self.assignment.Value(time_dimension.CumulVar(index))
                    departure_time_val = arrival_time_val + service_time
                
                current_route_stops.append(RouteStop(
                    location_id=location_obj.id,
                    arrival_time=arrival_time_val,
                    departure_time=departure_time_val,
                    load=cumulative_load, # Carga acumulada en este punto
                    distance_from_previous=arc_distance
                ))
                
                current_route_distance += arc_distance
                # La carga de la ruta se acumula solo para las demandas de los clientes
                if node_manager_index not in self.data['depots']:
                    # Asegurarse que self.data['demands'] existe y tiene el índice correcto
                    if 'demands' in self.data and node_manager_index < len(self.data['demands']):
                         current_route_load += self.data['demands'][node_manager_index]
                    else:
                        self.logger.error(f"Error al acceder a demands para node_manager_index {node_manager_index}")

                previous_routing_index = index
                index = self.assignment.Value(self.routing.NextVar(index))
            
            # Procesar la última parada de la ruta (generalmente el depósito final)
            final_node_manager_index = self.manager.IndexToNode(index)
            final_location_obj = self.locations[final_node_manager_index]
            final_arc_distance = self.routing.GetArcCostForVehicle(previous_routing_index, index, vehicle_idx)
            final_cumulative_load = self.assignment.Value(capacity_dimension.CumulVar(index))
            
            final_arrival_time = None
            if time_dimension:
                final_arrival_time = self.assignment.Value(time_dimension.CumulVar(index))
            
            current_route_stops.append(RouteStop(
                location_id=final_location_obj.id,
                arrival_time=final_arrival_time,
                departure_time=final_arrival_time, # En el depósito final, la salida es la llegada
                load=final_cumulative_load,
                distance_from_previous=final_arc_distance
            ))
            current_route_distance += final_arc_distance

            # Calcular el tiempo total de la ruta
            if time_dimension and current_route_stops:
                start_route_time = current_route_stops[0].departure_time if current_route_stops[0].departure_time is not None else 0
                end_route_time = current_route_stops[-1].arrival_time if current_route_stops[-1].arrival_time is not None else 0
                current_route_total_time = end_route_time - start_route_time
                if current_route_total_time < 0: current_route_total_time = 0 # Sanity check
            
            solution_total_load += current_route_load # Acumular la carga de esta ruta a la carga total de la solución
            
            routes_data.append(Route(
                vehicle_id=self.vehicles[vehicle_idx].id,
                stops=current_route_stops,
                total_distance=current_route_distance,
                total_load=current_route_load,
                total_time=current_route_total_time if time_dimension else None
            ))

        return CVRPSolution(
            status=status_code,
            routes=routes_data,
            total_distance=solution_total_distance, # Usar la distancia total de la solución directamente del objetivo
            total_load=solution_total_load,
            total_vehicles_used=len(routes_data),
            metadata={'solver': 'CVRPSolver'}
        )

    def _get_solver_status(self) -> str:
        if not self.assignment:
            return VRPSolutionStatus.INFEASIBLE.value

        status = self.routing.status()
        status_mapping = {
            routing_enums_pb2.RoutingSearchStatus.ROUTING_SUCCESS: VRPSolutionStatus.OPTIMAL.value,
            routing_enums_pb2.RoutingSearchStatus.ROUTING_FAIL: VRPSolutionStatus.NO_SOLUTION_FOUND.value,
            routing_enums_pb2.RoutingSearchStatus.ROUTING_FAIL_TIMEOUT: VRPSolutionStatus.NO_SOLUTION_FOUND.value,
            routing_enums_pb2.RoutingSearchStatus.ROUTING_INVALID: VRPSolutionStatus.ERROR_SOLVING_PROBLEM.value,
            routing_enums_pb2.RoutingSearchStatus.ROUTING_OPTIMAL: VRPSolutionStatus.OPTIMAL.value,
            routing_enums_pb2.RoutingSearchStatus.ROUTING_INFEASIBLE: VRPSolutionStatus.INFEASIBLE.value,
        }
        return status_mapping.get(status, VRPSolutionStatus.ERROR_SOLVING_PROBLEM.value)
