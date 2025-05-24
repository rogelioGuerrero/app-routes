import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

from .base_solver import BaseVRPSolver, VRPSolution, SolverType
from ....schemas_skills import OptimizationProfile

logger = logging.getLogger(__name__)


class ORToolsSolver(BaseVRPSolver):
    """Implementación de VRP usando OR-Tools."""
    
    def __init__(self):
        self._solver_type = SolverType.OR_TOOLS
        self._default_constraints = self._get_default_constraints()
    
    def _get_default_constraints(self) -> Dict[str, Any]:
        """Retorna las restricciones por defecto para el solver."""
        return {
            'optimization_objective': 'minimize_distance',
            'time_limit_sec': 30,
            'first_solution_strategy': 'PARALLEL_CHEAPEST_INSERTION',
            'local_search_metaheuristic': 'GUIDED_LOCAL_SEARCH',
            'solution_limit': 1,
            'log_search': False,
        }
    
    def _apply_optimization_profile(
        self, 
        constraints: Optional[Dict[str, Any]] = None,
        profile: Optional[Union[OptimizationProfile, str]] = None
    ) -> Dict[str, Any]:
        """Aplica un perfil de optimización predefinido a las restricciones.
        
        Args:
            constraints: Restricciones existentes (pueden sobrescribir el perfil)
            profile: Perfil de optimización a aplicar
            
        Returns:
            Diccionario con las restricciones actualizadas
        """
        # Si no hay perfil, usar el valor por defecto o el de las restricciones
        if profile is None:
            profile = constraints.get('profile', OptimizationProfile.STANDARD_OPERATIONS) if constraints else OptimizationProfile.STANDARD_OPERATIONS
        
        # Convertir a instancia de OptimizationProfile si es un string
        if isinstance(profile, str):
            try:
                profile = OptimizationProfile[profile]
            except KeyError:
                logger.warning(f"Perfil '{profile}' no encontrado. Usando STANDARD_OPERATIONS.")
                profile = OptimizationProfile.STANDARD_OPERATIONS
        
        # Configuración base del perfil
        profile_config = {
            OptimizationProfile.STANDARD_OPERATIONS: {
                'first_solution_strategy': 'PARALLEL_CHEAPEST_INSERTION',
                'local_search_metaheuristic': 'GUIDED_LOCAL_SEARCH',
                'time_limit_sec': 30,
                'optimization_objective': 'minimize_distance',
                'description': 'Configuración equilibrada para operaciones estándar durante el día.'
            },
            OptimizationProfile.FAST_DELIVERY: {
                'first_solution_strategy': 'PATH_CHEAPEST_ARC',
                'local_search_metaheuristic': 'GUIDED_LOCAL_SEARCH',
                'time_limit_sec': 15,
                'optimization_objective': 'minimize_time',
                'description': 'Prioriza la velocidad de entrega sobre la distancia.'
            },
            OptimizationProfile.COST_EFFICIENT: {
                'first_solution_strategy': 'SAVINGS',
                'local_search_metaheuristic': 'GREEDY_DESCENT',
                'time_limit_sec': 60,
                'optimization_objective': 'minimize_distance',
                'description': 'Optimizado para minimizar distancias y costos operativos.'
            },
            OptimizationProfile.EXTENDED_OPERATIONS: {
                'first_solution_strategy': 'CHRISTOFIDES',
                'local_search_metaheuristic': 'SIMULATED_ANNEALING',
                'time_limit_sec': 300,  # 5 minutos
                'optimization_objective': 'minimize_distance',
                'description': 'Para planificación detallada con tiempo de cómputo extendido.'
            },
            OptimizationProfile.RAPID_RESPONSE: {
                'first_solution_strategy': 'BEST_INSERTION',
                'local_search_metaheuristic': 'GREEDY_DESCENT',
                'time_limit_sec': 5,
                'optimization_objective': 'minimize_time',
                'description': 'Obtener soluciones rápidas para aplicaciones interactivas.'
            },
            OptimizationProfile.BALANCED: {
                'first_solution_strategy': 'PARALLEL_CHEAPEST_INSERTION',
                'local_search_metaheuristic': 'GUIDED_LOCAL_SEARCH',
                'time_limit_sec': 45,
                'optimization_objective': 'minimize_cost',
                'description': 'Equilibrio general entre distancia, tiempo y costos.'
            }
        }.get(profile, {})
        
        # Combinar configuraciones: por defecto -> perfil -> restricciones personalizadas
        result = self._default_constraints.copy()
        result.update(profile_config)
        
        # Aplicar restricciones personalizadas (sobrescriben el perfil)
        if constraints:
            # No sobrescribir la descripción del perfil
            profile_description = result.pop('description', None)
            result.update(constraints)
            if profile_description and 'description' not in constraints:
                result['description'] = profile_description
        
        logger.info(f"Aplicando perfil: {profile.name} - {result.get('description', '')}")
        return result
    
    @property
    def solver_type(self) -> SolverType:
        return self._solver_type
        
    def _get_strategy_enum(self, strategy_name: str):
        """Obtiene la enumeración de estrategia correspondiente al nombre."""
        from ortools.constraint_solver import routing_enums_pb2
        
        strategy_map = {
            'AUTOMATIC': routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
            'PATH_CHEAPEST_ARC': routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
            'PATH_MOST_CONSTRAINED_ARC': routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
            'SAVINGS': routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
            'SWEEP': routing_enums_pb2.FirstSolutionStrategy.SWEEP,
            'CHRISTOFIDES': routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
            'ALL_UNPERFORMED': routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED,
            'BEST_INSERTION': routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION,
            'PARALLEL_CHEAPEST_INSERTION': routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
            'SEQUENTIAL_CHEAPEST_INSERTION': routing_enums_pb2.FirstSolutionStrategy.SEQUENTIAL_CHEAPEST_INSERTION,
            'LOCAL_CHEAPEST_INSERTION': routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
            'GLOBAL_CHEAPEST_ARC': routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
            'LOCAL_CHEAPEST_ARC': routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC,
            'FIRST_UNBOUND_MIN_VALUE': routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE,
        }
        
        return strategy_map.get(strategy_name, routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    
    def _get_metaheuristic_enum(self, metaheuristic_name: str):
        """Obtiene la enumeración de metaheurística correspondiente al nombre."""
        from ortools.constraint_solver import routing_enums_pb2
        
        metaheuristic_map = {
            'AUTOMATIC': routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
            'GREEDY_DESCENT': routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
            'GUIDED_LOCAL_SEARCH': routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
            'SIMULATED_ANNEALING': routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
            'TABU_SEARCH': routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
        }
        
        return metaheuristic_map.get(metaheuristic_name, routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    
    def _configure_search_parameters(self, constraints: Dict[str, Any]) -> Any:
        """Configura los parámetros de búsqueda avanzados."""
        from ortools.constraint_solver import routing_enums_pb2
        
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        
        # Obtener configuración del perfil o usar valores por defecto
        first_solution_strategy = constraints.get('first_solution_strategy', 'PARALLEL_CHEAPEST_INSERTION')
        local_search_metaheuristic = constraints.get('local_search_metaheuristic', 'GUIDED_LOCAL_SEARCH')
        time_limit_sec = int(constraints.get('time_limit_sec', 30))
        solution_limit = constraints.get('solution_limit', 1)
        log_search = constraints.get('log_search', False)
        
        # Configurar tiempo límite
        search_parameters.time_limit.seconds = time_limit_sec
        
        # Configurar estrategia de primera solución
        search_parameters.first_solution_strategy = self._get_strategy_enum(first_solution_strategy)
        
        # Configurar metaheurística de búsqueda local
        search_parameters.local_search_metaheuristic = self._get_metaheuristic_enum(local_search_metaheuristic)
        
        # Configurar límite de soluciones
        search_parameters.solution_limit = solution_limit
        
        # Configurar log de búsqueda
        search_parameters.log_search = log_search
        
        # Configuración de operadores de búsqueda local
        search_parameters.local_search_operators.use_path_lns = True
        search_parameters.local_search_operators.use_inactive_lns = True
        
        # Deshabilitar verificaciones costosas para mejor rendimiento
        search_parameters.use_full_propagation = False
        
        # Configuración específica para la versión actual de OR-Tools
        if hasattr(search_parameters, 'log_search'):
            search_parameters.log_search = log_search
        
        logger.info(f"Configuración de búsqueda: strategy={first_solution_strategy}, "
                   f"metaheuristic={local_search_metaheuristic}, "
                   f"time_limit={time_limit_sec}s")
        
        return search_parameters
    
    def _configure_optimization_objective(self, routing, constraints, transit_callback_index, time_callback_index):
        """Configura el objetivo de optimización basado en las restricciones.
        
        Args:
            routing: Modelo de enrutamiento de OR-Tools
            constraints: Diccionario con restricciones y configuración
            transit_callback_index: Índice del callback de distancia
            time_callback_index: Índice del callback de tiempo
            
        Los posibles valores para optimization_objective son:
        - 'minimize_distance': Minimiza la distancia total (por defecto)
        - 'minimize_time': Minimiza el tiempo total
        - 'minimize_cost': Minimiza el costo (puede ser una combinación de distancia y tiempo)
        """
        optimization_objective = constraints.get('optimization_objective', 'minimize_distance')
        
        if optimization_objective == 'minimize_time':
            routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)
            logger.info("Objetivo de optimización: Minimizar tiempo total")
        elif optimization_objective == 'minimize_cost':
            # En un escenario real, aquí podrías combinar distancia y tiempo con ponderaciones
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            logger.info("Objetivo de optimización: Minimizar costo (distancia)")
        else:  # minimize_distance (por defecto)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            logger.info("Objetivo de optimización: Minimizar distancia total")
    
    async def solve(
        self,
        distance_matrix: List[List[float]],
        time_matrix: List[List[float]],
        locations: List[Any],
        vehicles: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> VRPSolution:
        """
        Resuelve un problema VRP usando OR-Tools con soporte para perfiles de optimización.
        
        Args:
            distance_matrix: Matriz de distancias entre ubicaciones en metros
            time_matrix: Matriz de tiempos de viaje entre ubicaciones en segundos
            locations: Lista de ubicaciones (incluyendo depósitos)
            vehicles: Lista de vehículos con sus características
            constraints: Restricciones adicionales y configuración del solver
                - profile: Perfil de optimización a utilizar (ver OptimizationProfile)
                - optimization_objective: 'minimize_distance', 'minimize_time' o 'minimize_cost'
                - time_limit_sec: Tiempo límite en segundos
                - first_solution_strategy: Estrategia para la primera solución
                - local_search_metaheuristic: Metaheurística para búsqueda local
                - solution_limit: Límite de soluciones a explorar
                - log_search: Habilitar logs detallados de búsqueda
            **kwargs: Parámetros adicionales
                
        Returns:
            VRPSolution con las rutas y métricas
                
        Raises:
            ValueError: Si los parámetros de entrada no son válidos
        """
        """
        Resuelve un problema VRP usando OR-Tools.
        
        Args:
            distance_matrix: Matriz de distancias entre ubicaciones
            time_matrix: Matriz de tiempos entre ubicaciones
            locations: Lista de ubicaciones (incluyendo depósitos)
            vehicles: Lista de vehículos con sus características
            constraints: Restricciones adicionales
            **kwargs: Parámetros adicionales
                
        Returns:
            VRPSolution con las rutas y métricas
            
        Raises:
            ValueError: Si los parámetros de entrada no son válidos
        """
        # Validación básica
        if not distance_matrix or not time_matrix:
            raise ValueError("Las matrices de distancia y tiempo son requeridas")
            
        if not locations:
            raise ValueError("Se requiere al menos una ubicación")
            
        if not vehicles:
            raise ValueError("Se requiere al menos un vehículo")
            
        # Aplicar perfil de optimización si se especifica
        profile = constraints.pop('profile', None) if constraints else None
        constraints = self._apply_optimization_profile(constraints, profile)
        
        # Extraer parámetros del perfil
        optimization_objective = constraints.get('optimization_objective', 'minimize_distance')
        
        # Validación básica
        num_vehicles = len(vehicles)
        num_locations = len(locations)
        
        # Validar que las matrices sean cuadradas y del tamaño correcto
        if len(distance_matrix) != num_locations or any(len(row) != num_locations for row in distance_matrix):
            raise ValueError("La matriz de distancia no es cuadrada o no coincide con el número de ubicaciones")
            
        if len(time_matrix) != num_locations or any(len(row) != num_locations for row in time_matrix):
            raise ValueError("La matriz de tiempo no es cuadrada o no coincide con el número de ubicaciones")
        
        # 1. Crear el modelo de datos con múltiples depósitos
        # Obtener los índices de inicio y fin de cada vehículo
        starts = []
        ends = []
        
        for vehicle in vehicles:
            depot_id = vehicle.get('depot_id', 0)
            # Validar que el depot_id sea un índice válido
            if depot_id < 0 or depot_id >= num_locations:
                logger.warning(f"Depot_id {depot_id} fuera de rango. Usando 0 por defecto.")
                depot_id = 0
            
            starts.append(depot_id)
            # Por defecto, el vehículo termina en el mismo depósito
            end_depot = vehicle.get('end_depot_id', depot_id)
            if end_depot < 0 or end_depot >= num_locations:
                end_depot = depot_id
            ends.append(end_depot)
        
        # Crear el administrador de índices con múltiples depósitos
        manager = pywrapcp.RoutingIndexManager(
            num_locations,  # Número de ubicaciones
            num_vehicles,   # Número de vehículos
            starts,         # Puntos de inicio (uno por vehículo)
            ends            # Puntos de fin (uno por vehículo)
        )
        
        logger.info(f"Depósitos de inicio: {starts}")
        logger.info(f"Depósitos de fin: {ends}")
        
        # 2. Crear el modelo de enrutamiento
        routing = pywrapcp.RoutingModel(manager)
        
        # 3. Definir callbacks de distancia y tiempo
        def distance_callback(from_index, to_index):
            """Retorna la distancia entre los dos nodos."""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node] * 1000)  # Convertir a metros
        
        def time_callback(from_index, to_index):
            """Retorna el tiempo de viaje entre los dos nodos."""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(time_matrix[from_node][to_node] * 60)  # Convertir a segundos
            
        # Registrar callbacks
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        
        # 4. Configurar la función de costo (distancia)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Configurar el objetivo de optimización
        self._configure_optimization_objective(
            routing, 
            constraints, 
            transit_callback_index, 
            time_callback_index
        )
        
        # Configurar restricción de duración máxima de ruta
        max_route_duration_sec = constraints.get('max_route_duration_sec', 8 * 3600)  # 8 horas por defecto
        routing.AddDimension(
            time_callback_index,
            0,  # Slack máximo en segundos
            max_route_duration_sec,
            False,  # No forzar el inicio a cero
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # 6. Agregar restricciones de capacidad (si corresponde)
        demands = constraints.get('demands')
        vehicle_capacities = constraints.get('vehicle_capacities')
        
        if demands is not None and vehicle_capacities is not None:
            if len(demands) != num_locations:
                raise ValueError("La longitud de 'demands' debe coincidir con el número de ubicaciones")
                
            if len(vehicle_capacities) != num_vehicles:
                raise ValueError("La longitud de 'vehicle_capacities' debe coincidir con el número de vehículos")
            
            def demand_callback(from_index):
                """Retorna la demanda del nodo."""
                from_node = manager.IndexToNode(from_index)
                return demands[from_node]
                
            demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
            
            routing.AddDimensionWithVehicleCapacity(
                demand_callback_index,
                0,  # Slack máximo
                vehicle_capacities,  # Capacidad por vehículo
                True,  # Iniciar en cero
                'Capacity'
            )
        
        # 7. Configurar parámetros de búsqueda
        search_parameters = self._configure_search_parameters(constraints)
        
        # 8. Resolver el problema
        logger.info(f"Resolviendo VRP con {num_locations} ubicaciones y {num_vehicles} vehículos...")
        solution = routing.SolveWithParameters(search_parameters)
        
        # 9. Procesar la solución
        if not solution:
            logger.warning("No se encontró solución factible")
            # Obtener todos los índices de depósitos
            depot_indices = {v.get('depot_id', 0) for v in vehicles}
            # Todas las ubicaciones excepto los depósitos se consideran no asignadas
            unassigned = [i for i in range(num_locations) if i not in depot_indices]
            return VRPSolution(
                routes=[], 
                distances=[], 
                times=[], 
                loads=[], 
                unassigned=unassigned,
                metadata={"status": "NO_SOLUTION_FOUND"}
            )
            
        routes = []
        route_distances = []
        route_times = []
        route_loads = []
        all_used = set()
        
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0
            route_time = 0
            route_load = 0
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                all_used.add(node_index)
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                
                # Acumular distancia y tiempo
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
                route_time += time_matrix[manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
                
                # Acumular carga (si aplica)
                if demands is not None:
                    route_load += demands[manager.IndexToNode(previous_index)]
            
            # Agregar la ruta solo si tiene más que solo el depósito
            if len(route) > 1:
                routes.append(route)
                route_distances.append(route_distance / 1000.0)  # Convertir a km
                route_times.append(route_time)
                route_loads.append(route_load)
        
        # Identificar ubicaciones no asignadas (todas excepto las usadas y los depósitos)
        depot_indices = {v.get('depot_id', 0) for v in vehicles}  # Índices de todos los depósitos
        unassigned = [i for i in range(num_locations) if i not in all_used and i not in depot_indices]
        
        # Recolectar métricas
        metadata = {
            "solver": "OR-Tools",
            "status": routing.status(),
            "solution_count": solution.solutions() if hasattr(solution, 'solutions') else 1,
            "unassigned_count": len(unassigned),
            "routes_count": len(routes)
        }
        
        return VRPSolution(
            routes=routes,
            distances=route_distances,
            times=route_times,
            loads=route_loads,
            unassigned=unassigned,
            metadata=metadata
        )
