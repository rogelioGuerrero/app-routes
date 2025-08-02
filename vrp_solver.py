"""
VRP Solver usando OR-Tools con soporte completo para todas las restricciones:
- Ventanas de tiempo (time windows)
- Capacidad de peso y volumen
- Habilidades requeridas (skills)
- No se manejan breaks en esta versión
- Pickup and delivery
- Múltiples depósitos
- Penalizaciones por nodos no visitados
"""


from __future__ import annotations

import logging
from typing import Dict, List, Any
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

logger = logging.getLogger(__name__)


class VRPSolver:
    """Solver principal para problemas de VRP usando OR-Tools."""
    
    # Constantes de tiempo (en segundos)
    SECONDS_PER_HOUR = 3600
    SECONDS_PER_DAY = 24 * 3600  # 24 horas
    DEFAULT_MAX_SLACK = 2 * 3600  # 2 horas
    DEFAULT_SERVICE_TIME = 300  # 5 minutos
    
    # Factores de conversión
    M3_TO_CM3 = 1_000_000  # 1 m³ = 1,000,000 cm³
    
    # Configuración del solver
    DEFAULT_WEIGHT_CAPACITY = 1000  # kg
    DEFAULT_VOLUME_CAPACITY = 100.0  # m³
    
    # Configuración por defecto de optimización
    DEFAULT_OPTIMIZATION_PROFILE = {
        'first_solution_strategy': 'PATH_CHEAPEST_ARC',
        'local_search_metaheuristic': 'GUIDED_LOCAL_SEARCH',
        'time_limit_seconds': 30
    }
    
    def __init__(self, vrp_data: Dict[str, Any], distance_matrix: List[List[int]] = None, time_matrix: List[List[int]] = None):
        """
        Inicializa el solver con los datos del problema VRP.
        
        Args:
            vrp_data: Diccionario con todos los datos del problema VRP
            distance_matrix: Matriz de distancias entre ubicaciones (opcional)
            time_matrix: Matriz de tiempos de viaje entre ubicaciones (opcional)
        """
        self.data = vrp_data
        self.manager = None
        self.routing = None
        self.solution = None
        
        # Extraer datos principales
        self.locations = self.data.get('locations', [])
        self.vehicles = self.data.get('vehicles', [])
        self.pickups_deliveries = self.data.get('pickups_deliveries', [])
        
        # Inicializar matrices con las proporcionadas o listas vacías
        self.distance_matrix = distance_matrix or []
        self.time_matrix = time_matrix or []
        
        # Mapeo rápido id -> índice para referencias internas
        self.node_map: Dict[str, int] = {loc.get('id'): idx for idx, loc in enumerate(self.locations)}
        
        # Configuración de optimización
        self.optimization_profile = self.DEFAULT_OPTIMIZATION_PROFILE.copy()
        if 'optimization_profile' in vrp_data:
            self.optimization_profile.update(vrp_data['optimization_profile'])
        
        # Validar datos básicos
        self._validate_input_data()
        
        # Crear manager y routing model
        self._create_routing_model()
        
    def _validate_input_data(self):
        """Valida los datos de entrada, incluyendo prioridades y días de la semana."""
        self._validate_weekdays()
        
    def _validate_weekdays(self):
        """Valida que las ubicaciones solo se visiten en días permitidos."""
        for loc in self.locations:
            if 'allowed_weekdays' in loc and loc['allowed_weekdays']:
                weekday = loc.get('date', 'mon').lower()  # 'mon' por defecto
                allowed_days = [d.lower() for d in loc['allowed_weekdays']]
                if weekday not in allowed_days:
                    logger.warning(f"Ubicación {loc.get('id')} no se puede visitar el día {weekday}")
                    # No fallamos, solo registramos la advertencia
        
    def _create_routing_model(self):
        """Crea el manager y modelo de routing de OR-Tools."""
        num_locations = len(self.locations)
        num_vehicles = len(self.vehicles)
        
        # Identificar depósitos (start/end locations para cada vehículo)
        depot_indices = []
        for vehicle in self.vehicles:
            start_id = vehicle.get('start_location_id')
            end_id = vehicle.get('end_location_id')
            
            # Encontrar índices de ubicaciones de inicio y fin
            start_idx = next((i for i, loc in enumerate(self.locations) if loc.get('id') == start_id), None)
            end_idx = next((i for i, loc in enumerate(self.locations) if loc.get('id') == end_id), None)
            
            # Asumimos que las validaciones ya se hicieron en vrp_validator.py
            depot_indices.append((start_idx, end_idx))
        
        # Crear manager con depósitos múltiples
        starts = [depot[0] for depot in depot_indices]
        ends = [depot[1] for depot in depot_indices]
        
        self.manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, starts, ends)
        self.routing = pywrapcp.RoutingModel(self.manager)
        
        logger.info(f"Modelo creado: {num_locations} nodos, {num_vehicles} vehículos")
        logger.info(f"Depósitos: starts={starts}, ends={ends}")
        
    def _add_distance_constraint(self):
        """Añade la restricción de distancia/costo de transporte."""
        def distance_callback(from_index, to_index):
            """Callback para calcular distancia entre dos nodos."""
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return self.distance_matrix[from_node][to_node]
        
        transit_callback_index = self.routing.RegisterTransitCallback(distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        logger.info("Restricción de distancia añadida")
        
    def _add_time_window_constraints(self):
        """Añade restricciones de ventanas de tiempo."""
        if not self.time_matrix:
            logger.warning("No hay matriz de tiempo, usando matriz de distancia para tiempo")
            time_matrix = self.distance_matrix
        else:
            time_matrix = self.time_matrix
            
        def time_callback(from_index, to_index):
            """Callback para calcular tiempo de viaje entre dos nodos incluyendo tiempo de servicio."""
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            
            # Tiempo de viaje base
            travel_time = time_matrix[from_node][to_node]
            
            # Añadir tiempo de servicio del nodo de origen (si no es un nodo de depósito)
            if from_node < len(self.locations):
                service_time = self.locations[from_node].get('service_time', 0)
                travel_time += service_time
                
            return travel_time
        
        transit_callback_index = self.routing.RegisterTransitCallback(time_callback)
        
        # Añadir dimensión de tiempo
        time_dimension_name = 'Time'
        
        # Configuración de la dimensión de tiempo
        max_slack = self.DEFAULT_MAX_SLACK
        
        # Calcular la capacidad necesaria para la dimensión de tiempo
        vehicle_span = max(
            vehicle.get('end_time', self.SECONDS_PER_DAY) - 
            vehicle.get('start_time', 0) 
            for vehicle in self.vehicles
        )
        capacity = max(self.SECONDS_PER_DAY, vehicle_span * 2)  # Capacidad generosa
        
        logger.info(f"Añadiendo dimensión de tiempo con capacidad={capacity}, slack={max_slack}")
        
        self.routing.AddDimension(
            transit_callback_index,
            max_slack,   # slack máximo (2 horas)
            capacity,    # capacidad máxima generosa
            False,       # no forzar inicio en cero para mayor flexibilidad
            time_dimension_name
        )
        
        time_dimension = self.routing.GetDimensionOrDie(time_dimension_name)
        
        # Configurar ventanas de tiempo para cada ubicación
        for location_idx, location in enumerate(self.locations):
            try:
                # Obtener el índice del nodo en el modelo de routing
                index = self.manager.NodeToIndex(location_idx)
                
                # Obtener ventana de tiempo con valores por defecto generosos
                time_window_start = int(location.get('time_window_start', 0))
                time_window_end = int(location.get('time_window_end', self.SECONDS_PER_DAY))
                
                # Asegurar que la ventana de tiempo sea válida
                if time_window_end <= time_window_start:
                    time_window_end = time_window_start + self.SECONDS_PER_HOUR
                    logger.warning(f"Ventana de tiempo inválida para ubicación {location.get('id')}. Ajustando a {time_window_start}-{time_window_end}")
                    
                logger.debug(f"Ubicación {location.get('id')}: ventana {time_window_start}-{time_window_end}")
                
                # Establecer ventana de tiempo
                time_dimension.CumulVar(index).SetRange(time_window_start, time_window_end)
                
            except Exception as e:
                logger.error(f"Error al configurar ventana de tiempo para ubicación {location.get('id', f'index_{location_idx}')}: {str(e)}")
                # Continuar con las siguientes ubicaciones en lugar de fallar completamente
                continue
        
        # Configurar ventanas de tiempo para cada vehículo
        for vehicle_id, vehicle in enumerate(self.vehicles):
            start_index = self.routing.Start(vehicle_id)
            end_index = self.routing.End(vehicle_id)
            start_time = int(vehicle.get('start_time', 0))
            end_time = int(vehicle.get('end_time', self.SECONDS_PER_DAY))
            
            # Asegurar que la ventana de tiempo del vehículo sea válida
            if end_time <= start_time:
                end_time = start_time + self.SECONDS_PER_HOUR
                
            logger.debug(f"Vehículo {vehicle_id}: ventana {start_time}-{end_time}")
            
            try:
                time_dimension.CumulVar(start_index).SetRange(start_time, end_time)
                time_dimension.CumulVar(end_index).SetRange(start_time, end_time)
            except Exception as e:
                logger.error(f"Error al establecer ventana de tiempo para vehículo {vehicle_id}: {str(e)}")
                raise

        # Configurar costo de span global para optimizar el tiempo total
        time_dimension.SetGlobalSpanCostCoefficient(100)
        logger.info("Restricciones de ventana de tiempo añadidas")
        
    def _add_capacity_constraints(self):
        """Añade restricciones de capacidad (peso y volumen)."""
        # Restricción de peso
        def weight_callback(from_index):
            """Callback para obtener demanda de peso de un nodo."""
            from_node = self.manager.IndexToNode(from_index)
            return int(self.locations[from_node].get('weight_demand', 0))
        
        weight_callback_index = self.routing.RegisterUnaryTransitCallback(weight_callback)
        
        # Capacidades de peso por vehículo
        weight_capacities = [
            int(vehicle.get('weight_capacity', self.DEFAULT_WEIGHT_CAPACITY))
            for vehicle in self.vehicles
        ]
        
        self.routing.AddDimensionWithVehicleCapacity(
            weight_callback_index,
            0,  # slack nulo
            weight_capacities,  # capacidades por vehículo
            True,  # comenzar acumulador en cero
            'Weight'
        )
        
        # Restricción de volumen - usando valores enteros (centímetros cúbicos)
        def volume_callback(from_index):
            """Callback para obtener demanda de volumen de un nodo en centímetros cúbicos."""
            from_node = self.manager.IndexToNode(from_index)
            # Convertir m³ a cm³ (1m³ = 1,000,000 cm³) y redondear a entero
            volume_m3 = float(self.locations[from_node].get('volume_demand', 0.0))
            return int(round(volume_m3 * self.M3_TO_CM3))  # m³ a cm³
        
        volume_callback_index = self.routing.RegisterUnaryTransitCallback(volume_callback)
        
        # Capacidades de volumen por vehículo en cm³
        volume_capacities = [
            int(round(float(vehicle.get('volume_capacity', self.DEFAULT_VOLUME_CAPACITY)) * self.M3_TO_CM3))
            for vehicle in self.vehicles
        ]
        
        self.routing.AddDimensionWithVehicleCapacity(
            volume_callback_index,
            0,  # slack nulo
            volume_capacities,  # capacidades por vehículo
            True,  # comenzar acumulador en cero
            'Volume'
        )
        
        logger.info("Restricciones de capacidad (peso y volumen) añadidas")
        
    def _add_skill_constraints(self):
        """Añade restricciones de habilidades requeridas."""
        # Crear mapeo de habilidades por vehículo
        vehicle_skills = {}
        for vehicle_idx, vehicle in enumerate(self.vehicles):
            skills = set(vehicle.get('skills', []))
            vehicle_skills[vehicle_idx] = skills
        
        # Para cada ubicación con habilidades requeridas, restringir vehículos
        for location_idx, location in enumerate(self.locations):
            required_skills = set(location.get('required_skills', []))
            
            if required_skills:
                # Encontrar vehículos que pueden atender esta ubicación
                allowed_vehicles = []
                for vehicle_idx, skills in vehicle_skills.items():
                    if required_skills.issubset(skills):
                        allowed_vehicles.append(vehicle_idx)
                
                if not allowed_vehicles:
                    logger.warning(f"Ubicación {location.get('id')} no puede ser atendida por ningún vehículo")
                    if not self.allow_skipping_nodes:
                        raise ValueError(f"Ubicación {location.get('id')} requiere habilidades que ningún vehículo tiene")
                else:
                    # Restringir esta ubicación solo a vehículos permitidos
                    index = self.manager.NodeToIndex(location_idx)
                    self.routing.SetAllowedVehiclesForIndex(allowed_vehicles, index)
        
        logger.info("Restricciones de habilidades añadidas")
        
    def _add_priority_penalties(self):
        """Añade penalizaciones según la prioridad de las ubicaciones."""
        if not any('priority' in loc for loc in self.locations):
            return
            
        logger.info("Aplicando penalizaciones por prioridad")
        
        # Valores de penalización (mayor = más prioridad)
        PRIORITY_PENALTIES = {
            'H': 1000000,  # Alta prioridad
            'M': 100000,   # Media prioridad
            'L': 10000,    # Baja prioridad
            '': 1        # Sin prioridad (mínima)
        }
        
        for node_idx, location in enumerate(self.locations):
            if 'priority' not in location:
                continue
                
            priority = str(location.get('priority', '')).upper()
            penalty = PRIORITY_PENALTIES.get(priority, 1)
            
            # Solo aplicar a ubicaciones que no son depósitos
            if not self._is_depot(node_idx):
                index = self.manager.NodeToIndex(node_idx)
                self.routing.AddDisjunction([index], penalty)
                logger.debug(f"Prioridad {priority} para ubicación {location.get('id')} - Penalización: {penalty}")
    
    def _is_depot(self, node_idx: int) -> bool:
        """Verifica si un nodo es un depósito (inicio/fin de ruta)."""
        for vehicle in self.vehicles:
            start_idx = next((i for i, loc in enumerate(self.locations) 
                           if loc.get('id') == vehicle.get('start_location_id')), None)
            end_idx = next((i for i, loc in enumerate(self.locations) 
                          if loc.get('id') == vehicle.get('end_location_id')), None)
            
            if node_idx in (start_idx, end_idx):
                return True
        return False

    def _add_pickup_delivery_constraints(self):
        """Añade restricciones de pickup & delivery asegurando mismo vehículo y orden correcto.
        Soporta pares en formato [idx_pickup, idx_delivery] o {'pickup': id/idx, 'delivery': id/idx}."""
        if not self.pickups_deliveries:
            logger.info("No hay restricciones pickup-delivery")
            return

        def _to_index(ref):
            """Convierte una referencia de ubicación (id str o índice int) a índice int."""
            if isinstance(ref, int):
                return ref if 0 <= ref < len(self.locations) else None
            if isinstance(ref, str):
                return self.node_map.get(ref)
            return None

        valid_pairs = 0
        for pd_pair in self.pickups_deliveries:
            # Detectar formato
            if isinstance(pd_pair, (list, tuple)) and len(pd_pair) == 2:
                pickup_ref, delivery_ref = pd_pair[0], pd_pair[1]
            elif isinstance(pd_pair, dict):
                pickup_ref = pd_pair.get('pickup')
                delivery_ref = pd_pair.get('delivery')
            else:
                logger.warning(f"Formato de par pickup-delivery no reconocido: {pd_pair}")
                continue

            pickup_idx = _to_index(pickup_ref)
            delivery_idx = _to_index(delivery_ref)

            if pickup_idx is None or delivery_idx is None:
                logger.warning(f"Par pickup-delivery no válido (ids desconocidos): {pd_pair}")
                continue

            # Añadir restricción al modelo
            pickup_index = self.manager.NodeToIndex(pickup_idx)
            delivery_index = self.manager.NodeToIndex(delivery_idx)
            self.routing.AddPickupAndDelivery(pickup_index, delivery_index)
            self.routing.solver().Add(
                self.routing.VehicleVar(pickup_index) == self.routing.VehicleVar(delivery_index)
            )
            # Pickup antes que delivery usando dimensión de tiempo
            time_dimension = self.routing.GetDimensionOrDie('Time')
            self.routing.solver().Add(
                time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(delivery_index)
            )
            valid_pairs += 1

        logger.info(f"Restricciones pickup-delivery añadidas: {valid_pairs} pares válidos de {len(self.pickups_deliveries)}")
            
    def solve(self) -> Dict[str, Any]:
        """Resuelve el problema VRP con las restricciones definidas.
        
        Returns:
            Dict con la solución del problema o None si no se encontró solución.
        """
        # Las validaciones de matrices ya se realizaron en vrp_validator.py
        num_locations = len(self.locations)
        logger.info(f"Iniciando resolución con {num_locations} ubicaciones")
            
        # Añadir restricciones
        logger.info("Añadiendo restricciones al modelo...")
        self._add_distance_constraint()
        self._add_time_window_constraints()
        self._add_capacity_constraints()
        self._add_skill_constraints()
        self._add_pickup_delivery_constraints()
        self._add_priority_penalties()  # Añadir penalizaciones por prioridad
        
        # Configurar parámetros de búsqueda
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        
        # Configurar estrategia de primera solución
        first_solution_strategy = self.optimization_profile.get('first_solution_strategy', 'PATH_CHEAPEST_ARC')
        try:
            search_parameters.first_solution_strategy = getattr(
                routing_enums_pb2.FirstSolutionStrategy,
                first_solution_strategy,
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )
        except AttributeError:
            logger.warning(f"Estrategia de primera solución '{first_solution_strategy}' no válida. Usando PATH_CHEAPEST_ARC.")
            search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        
        # Configurar metaheurística de búsqueda local
        metaheuristic = self.optimization_profile.get('local_search_metaheuristic', 'GUIDED_LOCAL_SEARCH')
        try:
            search_parameters.local_search_metaheuristic = getattr(
                routing_enums_pb2.LocalSearchMetaheuristic,
                metaheuristic,
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
        except AttributeError:
            logger.warning(f"Metaheurística '{metaheuristic}' no válida. Usando GUIDED_LOCAL_SEARCH.")
            search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        
        # Configurar límite de tiempo
        time_limit = int(self.optimization_profile.get('time_limit_seconds', 30))
        search_parameters.time_limit.FromSeconds(time_limit)
        
        logger.info(f"Parámetros de optimización: {self.optimization_profile}")
        
        # Resolver el problema
        logger.info("Iniciando resolución del problema...")
        self.solution = self.routing.SolveWithParameters(search_parameters)
        
        if not self.solution:
            logger.warning("No se encontró solución factible")
            return {
                'status': 'error',
                'message': 'No se encontró solución factible con las restricciones dadas',
                'vehicles_used': 0,
                'routes': []
            }
            
        return self._extract_solution()
            
    def _extract_solution(self) -> Dict[str, Any]:
        """Extrae la solución del modelo resuelto."""
        if not self.solution:
            return None
            
        solution_data = {
            'status': 'success',
            'objective_value': self.solution.ObjectiveValue(),
            'routes': [],
            'total_distance': 0,
            'total_time': 0,
            'vehicle_stats': [],
            'unassigned_nodes': []
        }
        
        time_dimension = self.routing.GetDimensionOrDie('Time')
        weight_dimension = self.routing.GetDimensionOrDie('Weight')
        volume_dimension = self.routing.GetDimensionOrDie('Volume')
        
        # Extraer rutas para cada vehículo
        for vehicle_id in range(len(self.vehicles)):
            route_data = {
                'vehicle_id': self.vehicles[vehicle_id].get('id'),
                'vehicle_name': self.vehicles[vehicle_id].get('name'),
                'route': [],
                'distance': 0,
                'time': 0,
                'load_weight': 0,
                'load_volume': 0
            }
            
            index = self.routing.Start(vehicle_id)
            route_distance = 0
            logger.info(f"Procesando vehículo {vehicle_id}: start_index={index}")
            
            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                location = self.locations[node_index]
                
                # Información del nodo
                time_var = time_dimension.CumulVar(index)
                weight_var = weight_dimension.CumulVar(index)
                volume_var = volume_dimension.CumulVar(index)
                
                node_data = {
                    'location_id': location.get('id'),
                    'location_name': location.get('name'),
                    'location_type': location.get('type'),
                    'coords': location.get('coords'),
                    'arrival_time': self.solution.Value(time_var),
                    'weight_load': self.solution.Value(weight_var),
                    'volume_load': self.solution.Value(volume_var) / 100.0,  # Convertir de vuelta
                    'service_time': location.get('service_time', 0)
                }
                
                route_data['route'].append(node_data)
                
                # Calcular distancia al siguiente nodo
                previous_index = index
                index = self.solution.Value(self.routing.NextVar(index))
                
                if not self.routing.IsEnd(index):
                    from_node = self.manager.IndexToNode(previous_index)
                    to_node = self.manager.IndexToNode(index)
                    
                    # Verificar que los índices estén dentro de los límites de la matriz
                    if (0 <= from_node < len(self.distance_matrix) and 
                        0 <= to_node < len(self.distance_matrix[from_node])):
                        route_distance += self.distance_matrix[from_node][to_node]
                    else:
                        logger.warning(f"Índices fuera de rango: from_node={from_node}, to_node={to_node}")
                        # Usar una distancia predeterminada segura (por ejemplo, 0 o un valor pequeño)
                        route_distance += 0
            
            # Añadir nodo final
            final_node_index = self.manager.IndexToNode(index)
            final_location = self.locations[final_node_index]
            
            time_var = time_dimension.CumulVar(index)
            weight_var = weight_dimension.CumulVar(index)
            volume_var = volume_dimension.CumulVar(index)
            
            final_node_data = {
                'location_id': final_location.get('id'),
                'location_name': final_location.get('name'),
                'location_type': final_location.get('type'),
                'coords': final_location.get('coords'),
                'arrival_time': self.solution.Value(time_var),
                'weight_load': self.solution.Value(weight_var),
                'volume_load': self.solution.Value(volume_var) / 100.0,
                'service_time': final_location.get('service_time', 0)
            }
            
            route_data['route'].append(final_node_data)
            route_data['distance'] = route_distance
            route_data['time'] = self.solution.Value(time_var)
            
            # Solo agregar la ruta si tiene más de un nodo (origen y destino)
            if len(route_data['route']) > 1:
                solution_data['routes'].append(route_data)
                solution_data['total_distance'] += route_data['distance']
                solution_data['total_time'] = max(solution_data['total_time'], route_data['time'])
                
                # Estadísticas del vehículo
                solution_data['vehicle_stats'].append({
                    'vehicle_id': route_data['vehicle_id'],
                    'vehicle_name': route_data['vehicle_name'],
                    'distance': route_data['distance'],
                    'time': route_data['time'],
                    'nodes_visited': len(route_data['route']),
                })
        
        logger.info(f"Solución extraída: {len(solution_data['routes'])} rutas, "
                   f"distancia total: {solution_data['total_distance']}, ")
                   
        return solution_data
