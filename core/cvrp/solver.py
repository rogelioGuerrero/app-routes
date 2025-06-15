"""Implementación del solucionador CVRP utilizando Google OR-Tools."""

# Deshabilitar logs detallados de OR-Tools
import os
os.environ['CPLOG_TO_LOGGING'] = '0'
os.environ['GLOG_minloglevel'] = '2'  # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
import time

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from core.base_solver import BaseVRPSolver
from models import Location, Vehicle, VRPSolution, Route, RouteStop, VRPSolutionStatus

# Configurar logging
logger = logging.getLogger(__name__)

# Configurar logging para OR-Tools
os.environ['GLOG_minloglevel'] = '2'  # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL

# Constantes
DEFAULT_TIME_LIMIT = 30  # segundos
MAX_VEHICLES = 1000  # Número máximo de vehículos soportados

# Mapeo de estados de OR-Tools a nuestros estados
# Valores tomados de la documentación de OR-Tools
# https://github.com/google/or-tools/blob/stable/ortools/constraint_solver/routing_enums.proto
# Los valores de retorno deben coincidir con el tipo SolutionStatus definido en models/vrp_models.py
STATUS_MAPPING = {
    0: "ROUTE_FAILED",        # ROUTING_NOT_SOLVED
    1: "OPTIMAL",            # ROUTING_SUCCESS
    2: "ROUTE_FAILED",       # ROUTING_FAIL
    3: "NO_SOLUTION_FOUND",  # ROUTING_FAIL_TIMEOUT
    4: "INVALID",            # ROUTING_INVALID
    5: "FEASIBLE",           # ROUTING_PARTIAL_TIMEOUT
    6: "INFEASIBLE",         # ROUTING_INFEASIBLE
    7: "OPTIMAL",            # ROUTING_OPTIMAL
    8: "INFEASIBLE"          # ROUTING_INFEASIBLE_TIMEOUT
}

class CVRPSolver(BaseVRPSolver):
    """
    Implementación de un solucionador para el Problema de Ruteo de Vehículos con Restricción de Capacidad (CVRP)
    utilizando Google OR-Tools.
    """
    
    def __init__(self):
        """Inicializa el solucionador CVRP."""
        super().__init__()
        self.manager = None
        self.routing = None
        self.solution = None
        self._search_parameters = None
        self._time_dimension = None
        self._demand_callback_index = None
        self._time_callback_index = None
        self._distance_callback_index = None
        logger.debug("Inicializado CVRPSolver")

    def load_problem(
        self,
        distance_matrix: List[List[Union[int, float]]],
        locations: List[Union[Dict[str, Any], Location]],
        vehicles: List[Union[Dict[str, Any], Vehicle]],
        duration_matrix: Optional[List[List[Union[int, float]]]] = None,
        **kwargs
    ) -> None:
        """
        Carga un problema CVRP en el solucionador.
        
        Args:
            distance_matrix: Matriz de distancias entre ubicaciones (NxN)
            locations: Lista de ubicaciones (el primer elemento debe ser el depósito)
            vehicles: Lista de vehículos disponibles
            duration_matrix: Matriz opcional de duraciones entre ubicaciones (NxN)
            **kwargs: Argumentos adicionales específicos de la implementación
            
        Raises:
            ValueError: Si los datos de entrada no son válidos
            RuntimeError: Si no se puede cargar el problema
        """
        try:
            # Validar y cargar los datos básicos usando la implementación de la clase base
            super().load_problem(distance_matrix, locations, vehicles, duration_matrix, **kwargs)
            
            # Inicializar el administrador de índices de OR-Tools
            num_nodes = len(self.distance_matrix)
            num_vehicles = len(self.vehicles)
            
            # Crear el administrador de índices
            self.manager = pywrapcp.RoutingIndexManager(
                num_nodes,  # número de ubicaciones
                num_vehicles,  # número de vehículos
                [0] * num_vehicles,  # índices de inicio (todos en el depósito)
                [0] * num_vehicles   # índices de fin (todos terminan en el depósito)
            )
            
            # Crear el modelo de enrutamiento
            self.routing = pywrapcp.RoutingModel(self.manager)
            
            # Configurar parámetros de búsqueda por defecto
            self._setup_search_parameters()
            
            # Registrar callbacks para distancias y demandas
            self._register_callbacks()
            
            # Configurar dimensiones (capacidad y tiempo si está disponible)
            self._setup_dimensions()
            
            logger.info(f"Problema CVRP cargado con {num_nodes} ubicaciones y {num_vehicles} vehículos")
            
        except Exception as e:
            self.clear()
            logger.error(f"Error al cargar el problema CVRP: {str(e)}", exc_info=True)
            raise RuntimeError(f"No se pudo cargar el problema CVRP: {str(e)}") from e
    
    def _setup_search_parameters(self, time_limit_seconds: int = DEFAULT_TIME_LIMIT) -> None:
        """Configura los parámetros de búsqueda para el solucionador OR-Tools."""
        parameters = pywrapcp.DefaultRoutingSearchParameters()
        
        # Configurar estrategia de búsqueda
        parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        
        # Configurar metaheurística
        parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        
        # Configurar límite de tiempo
        parameters.time_limit.FromSeconds(time_limit_seconds)
        
        # Configurar parámetros de búsqueda local
        parameters.use_depth_first_search = True
        parameters.log_search = True
        
        self._search_parameters = parameters
    
    def _register_callbacks(self) -> None:
        """Registra las funciones de callback para distancias, tiempos y demandas."""
        # Callback para la matriz de distancias
        def distance_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return int(round(self.distance_matrix[from_node][to_node]))
        
        # Callback para la matriz de duraciones (si está disponible)
        def time_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return int(round(self.duration_matrix[from_node][to_node])) if self.duration_matrix else 0
        
        # Callback para las demandas
        def demand_callback(from_index):
            from_node = self.manager.IndexToNode(from_index)
            return int(round(self.locations[from_node].demand)) if hasattr(self.locations[from_node], 'demand') else 0
        
        # Registrar callbacks de demanda
        self._demand_callback_index = self.routing.RegisterUnaryTransitCallback(demand_callback)
        # Registrar callbacks de distancia y tiempo
        self._distance_callback_index = self.routing.RegisterTransitCallback(distance_callback)
        if self.duration_matrix:
            self._time_callback_index = self.routing.RegisterTransitCallback(time_callback)
        else:
            self._time_callback_index = None
        
        # Registrar y establecer evaluador de coste ponderado (distancia y tiempo)
        def cost_callback(from_index, to_index):
            dist = distance_callback(from_index, to_index)
            time_val = time_callback(from_index, to_index) if self._time_callback_index else 0
            # Extraer override_params o usar defaults según perfil
            params = getattr(self.optimization_profile, 'override_params', {}) or {}
            if 'distance_weight' in params or 'time_weight' in params:
                w_dist = float(params.get('distance_weight', 0.0))
                w_time = float(params.get('time_weight', 0.0))
            else:
                profile_name = getattr(self.optimization_profile, 'name', 'cost_saving')
                if profile_name == 'punctuality':
                    w_dist, w_time = 0.0, 1.0
                elif profile_name == 'balanced':
                    w_dist, w_time = 1.0, 1.0
                else:  # cost_saving
                    w_dist, w_time = 1.0, 0.0
            
            return int(round(w_dist * dist + w_time * time_val))
        
        self._cost_callback_index = self.routing.RegisterTransitCallback(cost_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(self._cost_callback_index)
    
    def _setup_dimensions(self) -> None:
        """Configura las dimensiones del problema (capacidad, tiempo, etc.)."""
        # Dimensión de capacidad
        self.routing.AddDimensionWithVehicleCapacity(
            self._demand_callback_index,
            0,  # slack máximo
            [int(round(v.capacity)) for v in self.vehicles],  # capacidades de los vehículos
            True,  # iniciar en cero
            'Capacity'
        )
        
        # Dimensión de tiempo (si está disponible la matriz de duraciones)
        if hasattr(self, '_time_callback_index') and self._time_callback_index is not None:
            # Configurar slack y horizonte desde override_params o usar defaults
            params = getattr(self.optimization_profile, 'override_params', {}) or {}
            # default 24h horizon
            time_horizon = int(params.get('time_horizon', 24 * 3600))
            # slack default to horizon to avoid infeasibility
            time_slack = int(params.get('time_slack', time_horizon))
            self.routing.AddDimension(
                self._time_callback_index,
                time_slack,
                time_horizon,
                True,  # fijar inicio en cero
                'Time'
            )
            self._time_dimension = self.routing.GetDimensionOrDie('Time')
    
    def _create_solution(self, solution, execution_time: float) -> VRPSolution:
        """
        Crea un objeto VRPSolution a partir de la solución de OR-Tools.
        
        Args:
            solution: Solución devuelta por OR-Tools
            execution_time: Tiempo de ejecución en segundos
            
        Returns:
            VRPSolution: Solución formateada
        """
        routes = []
        total_distance = 0
        total_load = 0
        
        for vehicle_id in range(len(self.vehicles)):
            # Inicializar variables para la ruta
            route_stops = []
            route_distance = 0
            route_load = 0
            
            # Añadir el depósito como primera parada
            route_stops.append(RouteStop(
                location_id=str(self.routing.Start(vehicle_id)),
                location_name="Depósito",
                arrival_time=0,
                departure_time=0,
                load=0,
                service_time=0,
                time_window=(0, 24 * 60 * 60),  # Ventana de 24 horas para el depósito
                distance_from_previous=0,
                wait_time=0
            ))
            
            # Variables para seguimiento de la ruta
            index = self.routing.Start(vehicle_id)
            prev_node_index = self.manager.IndexToNode(index)
            
            while not self.routing.IsEnd(index):
                # Obtener el siguiente nodo
                next_index = solution.Value(self.routing.NextVar(index))
                next_node_index = self.manager.IndexToNode(next_index)
                
                # Obtener distancia de la matriz y convertir a float con fallback
                raw_dist = self.distance_matrix[prev_node_index][next_node_index]
                try:
                    distance = float(raw_dist)
                except Exception:
                    distance = 0.0
                
                # Obtener la ubicación
                location = self.locations[next_node_index]
                
                # Calcular tiempo de llegada
                arrival_time = None
                departure_time = None
                if hasattr(self, '_time_dimension') and self._time_dimension:
                    time_var = self._time_dimension.CumulVar(index)
                    arrival_time = solution.Min(time_var)
                    departure_time = arrival_time + getattr(location, 'service_time', 0)
                
                # Actualizar carga
                if hasattr(location, 'demand'):
                    route_load += location.demand
                
                # Obtener ventana de tiempo
                time_window = getattr(location, 'time_window', (0, 24 * 60 * 60))
                
                # No calculamos tiempo de espera (no usado en CVRP básico)
                wait_time = 0
                
                # Crear parada de ruta
                route_stop = RouteStop(
                    location_id=str(next_node_index),
                    location_name=getattr(location, 'name', f'Ubicación {next_node_index}'),
                    arrival_time=arrival_time,
                    departure_time=departure_time,
                    load=route_load,
                    service_time=getattr(location, 'service_time', 0),
                    time_window=time_window,
                    distance_from_previous=distance,
                    wait_time=wait_time
                )
                
                route_stops.append(route_stop)
                route_distance += distance
                
                # Actualizar índices para la siguiente iteración
                index = next_index
                prev_node_index = next_node_index
            
            # Calcular la distancia total de la ruta sumando las distancias entre paradas
            route_distance = sum(
                stop.distance_from_previous 
                for stop in route_stops
            )
            
            # Calcular tiempo total de la ruta si está disponible
            route_duration = None
            if hasattr(self, '_time_dimension') and self._time_dimension:
                route_duration = (
                    solution.Min(self._time_dimension.CumulVar(index)) - 
                    solution.Min(self._time_dimension.CumulVar(self.routing.Start(vehicle_id)))
                )
            
            # Crear la ruta con los nombres de campos correctos
            route = Route(
                vehicle_id=str(vehicle_id),
                stops=route_stops,
                total_distance=route_distance,  # Campo requerido
                total_load=route_load,          # Campo requerido
                total_time=route_duration,      # Opcional
                total_cost=None                 # Opcional
            )
            
            routes.append(route)
            total_distance += route_distance
            total_load += route_load
        
        # Determinar estado de la solución usando el status del modelo
        try:
            status_code = self.routing.status()
            status = STATUS_MAPPING.get(status_code, "NO_SOLUTION_FOUND")
        except Exception:
            # Fallback: marcar como OPTIMAL si hay rutas, sino NO_SOLUTION_FOUND
            status = "OPTIMAL" if len(routes) > 0 else "NO_SOLUTION_FOUND"
        
        # Contar vehículos usados (rutas con más de 2 paradas: depósito + al menos una entrega + depósito)
        vehicles_used = len([r for r in routes if len(r.stops) > 2])
        
        # Crear y devolver la solución
        return VRPSolution(
            status=status,
            routes=routes,
            total_distance=total_distance,
            total_load=total_load,
            total_vehicles_used=vehicles_used,  # Campo requerido
            execution_time=execution_time,
            metadata={
                'solver': 'OR-Tools CVRP',
                'version': '1.0',
                'timestamp': datetime.utcnow().isoformat(),
                'num_vehicles_used': vehicles_used  # También en metadata por compatibilidad
            }
        )
    
    def solve(
        self,
        time_limit_seconds: int = DEFAULT_TIME_LIMIT,
        **kwargs
    ) -> VRPSolution:
        """
        Resuelve el problema CVRP cargado.
        
        Args:
            time_limit_seconds: Tiempo máximo de resolución en segundos
            **kwargs: Argumentos adicionales específicos de la implementación
            
        Returns:
            VRPSolution: Solución del problema
            
        Raises:
            RuntimeError: Si no se ha cargado ningún problema o no se puede resolver
        """
        if not self._is_loaded:
            raise RuntimeError("No se ha cargado ningún problema. Use load_problem() primero.")
        
        if not self.routing or not self.manager:
            raise RuntimeError("El problema no se ha inicializado correctamente.")
        
        try:
            # Actualizar el límite de tiempo si se proporciona
            if time_limit_seconds != DEFAULT_TIME_LIMIT:
                self._setup_search_parameters(time_limit_seconds)
            
            logger.info(f"Iniciando resolución del problema CVRP (límite: {time_limit_seconds}s)")
            start_time = time.time()
            
            # Resolver el problema
            solution = self.routing.SolveWithParameters(self._search_parameters)
            
            # Calcular tiempo de ejecución
            execution_time = time.time() - start_time
            
            if solution:
                logger.info(
                    f"Solución encontrada en {execution_time:.2f}s. "
                    f"Distancia total: {solution.ObjectiveValue()}"
                )
                
                # Crear y devolver la solución formateada
                self._solution = self._create_solution(solution, execution_time)
                return self._solution
            else:
                logger.warning(
                    f"No se encontró solución en {time_limit_seconds}s. "
                    f"Tiempo de ejecución: {execution_time:.2f}s"
                )
                return VRPSolution(
                    status="NO_SOLUTION_FOUND",
                    routes=[],
                    total_distance=0,
                    total_load=0,
                    num_vehicles_used=0,
                    total_vehicles_used=0,
                    execution_time=execution_time,
                    metadata={
                        'solver': 'OR-Tools CVRP',
                        'version': '1.0',
                        'timestamp': datetime.utcnow().isoformat(),
                        'error': 'No se encontró solución dentro del tiempo límite',
                        'status_code': 3  # ROUTING_FAIL_TIMEOUT
                    }
                )
                
        except Exception as e:
            error_msg = f"Error al resolver el problema CVRP: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return VRPSolution(
                status="INVALID",
                routes=[],
                total_distance=0,
                total_load=0,
                num_vehicles_used=0,
                total_vehicles_used=0,
                execution_time=execution_time if 'execution_time' in locals() else 0,
                metadata={
                    'solver': 'OR-Tools CVRP',
                    'version': '1.0',
                    'timestamp': datetime.utcnow().isoformat(),
                    'error': error_msg,
                    'status_code': 4  # ROUTING_INVALID
                }
            )
    
    def clear(self) -> None:
        """Limpia el estado del solucionador para resolver un nuevo problema."""
        super().clear()
        self.manager = None
        self.routing = None
        self.solution = None
        self._search_parameters = None
        self._time_dimension = None
        self._demand_callback_index = None
        self._time_callback_index = None
        self.routing = None
        self.solution = None
        self._search_parameters = None
        self._time_dimension = None
        self._demand_callback_index = None
        self._time_callback_index = None
        self._distance_callback_index = None
        logger.debug("CVRPSolver reiniciado")
        
    def _get_solver_status(self) -> str:
        """Obtiene el estado actual del solucionador."""
        if not self.solution:
            return "NO_SOLUTION_FOUND"
        
        # Verificar el estado del solver OR-Tools
        status = self.routing.status()
        
        # Mapear los estados de OR-Tools a nuestros estados
        status_mapping = {
            routing_enums_pb2.RoutingSearchStatus.ROUTING_SUCCESS: "OPTIMAL",
            routing_enums_pb2.RoutingSearchStatus.ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED: "FEASIBLE",
            routing_enums_pb2.RoutingSearchStatus.ROUTING_OPTIMAL: "OPTIMAL",
            routing_enums_pb2.RoutingSearchStatus.ROUTING_INFEASIBLE: "INFEASIBLE",
            routing_enums_pb2.RoutingSearchStatus.ROUTING_FAIL: "FAILED",
            routing_enums_pb2.RoutingSearchStatus.ROUTING_FAIL_TIMEOUT: "TIMEOUT",
            routing_enums_pb2.RoutingSearchStatus.ROUTING_INVALID: "INVALID",
            routing_enums_pb2.RoutingSearchStatus.ROUTING_NOT_SOLVED: "NOT_SOLVED"
        }
        
        # Verificar si hay rutas válidas
        routes = []
        for vehicle_id in range(len(self.vehicles)):
            index = self.routing.Start(vehicle_id)
            route = []
            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                route.append(node_index)
                index = self.solution.Value(self.routing.NextVar(index))
            routes.append(route)
        
        # Contar rutas no vacías (más de 2 nodos: depósito + al menos un cliente + depósito)
        valid_routes = sum(1 for route in routes if len(route) > 2)
        
        # Si hay rutas válidas, la solución es factible
        if valid_routes > 0:
            # Si el estado es óptimo o éxito, mantenerlo
            if status in [routing_enums_pb2.RoutingSearchStatus.ROUTING_OPTIMAL, routing_enums_pb2.RoutingSearchStatus.ROUTING_SUCCESS]:
                return "OPTIMAL"
            # De lo contrario, marcar como factible
            return "FEASIBLE"
        
        # Si no hay rutas válidas, verificar el estado
        if status in [routing_enums_pb2.RoutingSearchStatus.ROUTING_OPTIMAL, routing_enums_pb2.RoutingSearchStatus.ROUTING_SUCCESS]:
            # Si el solver dice que es óptimo pero no hay rutas, algo anda mal
            return "FEASIBLE" # La lógica original devuelve FEASIBLE aquí
            
        # Usar el mapeo de estados para otros casos
        return status_mapping.get(status, "UNKNOWN")
        
    def _format_solution(self, execution_time: float = 0.0) -> VRPSolution:
        """
        Formatea la solución en el formato estándar.
        
        Args:
            execution_time: Tiempo de ejecución en segundos
            
        Returns:
            VRPSolution: Solución formateada
            
        Raises:
            RuntimeError: Si no hay solución disponible para formatear
        """
        from models.vrp_models import Route, RouteStop
        from typing import List
        
        if not self.solution:
            raise RuntimeError("No hay solución disponible para formatear")
            
        routes: List[Route] = []
        total_distance = 0
        total_time = 0
        total_load = 0
        total_vehicles_used = 0
        
        # Obtener la dimensión de tiempo si existe
        time_dimension = None
        if hasattr(self, 'duration_matrix') and self.duration_matrix is not None:
            time_dimension = self.routing.GetDimensionOrDie('Time')
        
        for vehicle_id in range(len(self.vehicles)):
            index = self.routing.Start(vehicle_id)
            
            # Si no hay siguiente nodo, saltar este vehículo
            if self.routing.IsEnd(self.solution.Value(self.routing.NextVar(index))):
                continue
                
            stops: List[RouteStop] = []
            route_distance = 0
            route_time = 0
            route_load = 0
            cumulative_distance = 0
            
            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                
                # Obtener información de la ubicación
                location = self.locations[node_index]
                location_id = str(location.id) if hasattr(location, 'id') else str(node_index)
                
                # Crear parada con valores iniciales
                stop = RouteStop(
                    location_id=location_id,
                    load=route_load,
                    distance_from_previous=0,  # Se actualizará en la siguiente iteración
                    cumulative_distance=cumulative_distance,
                    arrival_time=None,  # Se actualizará si hay dimensión de tiempo
                    departure_time=None,  # Se actualizará si hay dimensión de tiempo
                    wait_time=None  # Opcional, se puede establecer si es necesario
                )
                
                # Actualizar distancias
                previous_index = index
                index = self.solution.Value(self.routing.NextVar(index))
                
                # Obtener distancia y asegurar valor numérico
                distance_raw = self.routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
                distance = float(distance_raw) if distance_raw is not None else 0.0
                
                # Actualizar distancias acumuladas
                cumulative_distance += distance
                route_distance += distance
                
                # Actualizar carga
                if node_index > 0:  # No contar el depósito
                    demand = location.demand if hasattr(location, 'demand') else 0
                    route_load += demand
                    total_load += demand
                
                # Actualizar tiempo si hay dimensión de tiempo
                if time_dimension is not None:
                    stop.arrival_time = self.solution.Min(time_dimension.CumulVar(previous_index))
                    stop.departure_time = stop.arrival_time  # Por defecto igual a la llegada
                    
                    # Añadir tiempo de servicio si existe
                    if hasattr(location, 'service_time') and location.service_time:
                        stop.departure_time += location.service_time
                    
                    route_time = self.solution.Min(time_dimension.CumulVar(index))
                
                # Actualizar distancia desde el nodo anterior
                if stops:
                    stops[-1].distance_from_previous = distance
                
                stops.append(stop)
            
            # Añadir la última parada (depósito final)
            if stops:
                node_index = self.manager.IndexToNode(index)
                location = self.locations[node_index]
                location_id = str(location.id) if hasattr(location, 'id') else str(node_index)
                
                final_stop = RouteStop(
                    location_id=location_id,
                    load=0,  # Al final, la carga debe ser 0
                    distance_from_previous=0,  # No hay siguiente parada
                    cumulative_distance=cumulative_distance,
                    arrival_time=None,
                    departure_time=None,
                    wait_time=None
                )
                
                if time_dimension is not None:
                    final_stop.arrival_time = self.solution.Min(time_dimension.CumulVar(index))
                    final_stop.departure_time = final_stop.arrival_time
                
                stops.append(final_stop)
                
                # Crear ruta
                route = Route(
                    vehicle_id=str(self.vehicles[vehicle_id].id) if hasattr(self.vehicles[vehicle_id], 'id') else str(vehicle_id),
                    stops=stops,
                    total_distance=route_distance,
                    total_load=route_load,
                    total_time=route_time if time_dimension is not None else None,
                    total_cost=route_distance  # Por defecto, el costo es la distancia
                )
                
                routes.append(route)
                total_distance += route_distance
                total_vehicles_used += 1
        
        # Obtener el estado de la solución
        status = self._get_solver_status()
        
        # Crear metadatos adicionales
        metadata = {
            'total_time': execution_time,  # Tiempo real de ejecución
            'has_time_dimension': time_dimension is not None,
            'total_vehicles_available': len(self.vehicles),
            'total_locations': len(self.locations),
            'matrix_provider': getattr(self, 'matrix_provider', 'unknown')
        }
        
        # Si no se usaron vehículos pero hay solución, es un problema
        if total_vehicles_used == 0 and len(self.vehicles) > 0:
            status = "NO_SOLUTION_FOUND"
        
        return VRPSolution(
            status=status,
            routes=routes,
            total_distance=total_distance,
            total_load=total_load,
            total_vehicles_used=total_vehicles_used,
            total_cost=total_distance,  # Por defecto, el costo total es la distancia total
            execution_time=execution_time,
            metadata=metadata
        )

# Mapeo de estados de OR-Tools para referencia
# Este mapeo es solo para documentación y no se utiliza en el código
# Los estados reales se manejan en STATUS_MAPPING
_ORT_STATUS_MAPPING = {
    0: "ROUTING_NOT_SOLVED",        # No se ha intentado resolver
    1: "ROUTING_SUCCESS",          # Solución encontrada correctamente
    2: "ROUTING_FAIL",             # Error durante la resolución
    3: "ROUTING_FAIL_TIMEOUT",     # Tiempo de espera agotado
    4: "ROUTING_INVALID",          # Parámetros inválidos
    5: "ROUTING_PARTIAL_TIMEOUT",  # Tiempo agotado, solución factible encontrada
    6: "ROUTING_INFEASIBLE",       # Problema infactible
    7: "ROUTING_OPTIMAL",          # Solución óptima encontrada
    8: "ROUTING_INFEASIBLE_TIMEOUT" # Tiempo agotado, problema infactible
}
