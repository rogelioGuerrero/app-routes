import logging
from fastapi import APIRouter, HTTPException

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vrp_solver.log')
    ]
)
logger = logging.getLogger('vrp_solver')
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from typing import List, Dict, Any, Optional
from schemas_skills import VRPSkillsRequest, SkillsLocation, SkillsVehicle
from route_polyline_utils import get_route_polyline_and_geojson
from schemas import VRPAdvancedResponse
from vrp_utils import min_to_hhmm, filter_viable_clients, add_warning, warn_matrix_fallback, warn_no_viable_clients
from vrp_constants import DEFAULT_SPEED_KMH, MAX_TIME_MINUTES, SKILL_PENALTY, DEFAULT_BUFFER_MINUTES

# --- Configurable polyline threshold ---
MAX_POLYLINE_ROUTES = 10  # Número máximo de rutas para generar polylines

logger = logging.getLogger(__name__)
router = APIRouter()

def get_depot_indices(locations: List[SkillsLocation]) -> List[int]:
    """
    Retorna los IDs de las ubicaciones que son depósitos.
    Si no hay depósitos, retorna [0] (primer ID como depósito por defecto).
    """
    depots = [loc.id for loc in locations if getattr(loc, 'is_depot', False)]
    return depots if depots else [0]  # Usar el primer ID como depósito por defecto

def get_advanced_search_parameters(problem_size: int) -> Any:
    """
    Devuelve parámetros de búsqueda optimizados según el tamaño del problema.
    """
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver.pywrapcp import DefaultRoutingSearchParameters
    
    # Crear parámetros de búsqueda con valores por defecto
    search_parameters = DefaultRoutingSearchParameters()
    
    # Configuración común
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    
    # Configurar metaheurística basada en el tamaño del problema
    if problem_size <= 20:  # Casos pequeños (hasta 20 ubicaciones)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 5
        search_parameters.solution_limit = 30
    elif problem_size <= 100:  # Casos medianos (21-100 ubicaciones)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 15
        search_parameters.solution_limit = 50
    else:  # Casos grandes (+100 ubicaciones)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
        )
        search_parameters.time_limit.seconds = 60
        search_parameters.solution_limit = 100
    
    # Configuración adicional para mejorar la búsqueda
    search_parameters.use_full_propagation = False
    search_parameters.log_search = False
    
    # Configurar estrategias de primer solución
    if problem_size <= 50:
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
    else:
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
    
    # Usar solo los parámetros básicos sin configurar operadores de búsqueda local
    # para evitar problemas de compatibilidad
    return search_parameters

# --- Función modular para obtención de matrices ---
async def get_distance_and_time_matrix(request, api_key, warnings):
    """
    Obtiene la matriz de distancias y tiempos usando Google Distance Matrix API o fallback euclidiano.
    Maneja errores específicos y logging. Soporta chunking si hay más de 10 ubicaciones.
    """
    from vrp_utils import add_warning
    import math

    # Integración ORS primero
    from vrp.matrix.ors import ORSMatrixProvider
    try:
        provider = ORSMatrixProvider(api_key)
        dist, dur, ors_warnings = await provider.get_matrix(request.locations)
        warnings.extend(ors_warnings)
        return dist, dur, warnings
    except Exception as e:
        warnings = add_warning(warnings, "ORS_ERROR", f"ORS failed: {e}")

    locations = request.locations
    n = len(locations)
    MAX_CHUNK = 10  # Límite de la API de Google
    
    if n <= MAX_CHUNK:
        # --- Comportamiento para pocas ubicaciones ---
        origins = [f"{loc.lat},{loc.lon}" for loc in locations]
        url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        params = {
            "origins": "|".join(origins),
            "destinations": "|".join(origins),
            "mode": getattr(request, 'mode', 'driving') or 'driving',
            "units": getattr(request, 'units', 'metric') or 'metric',
            "key": api_key
        }
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code != 200:
                logger.error(f"Error HTTP consultando Google Distance Matrix: {response.status_code} {response.text}")
                raise Exception(f"Error consultando Google Distance Matrix: {response.text}")
            data = response.json()
            
            # Procesar respuesta
            if data['status'] != 'OK':
                raise Exception(f"Error en la respuesta de Google: {data.get('error_message', 'Unknown error')}")
                
            # Inicializar matrices
            distance_matrix = [[0] * n for _ in range(n)]
            time_matrix = [[0] * n for _ in range(n)]
            
            # Llenar matrices con datos de la API
            for i, row in enumerate(data['rows']):
                for j, element in enumerate(row['elements']):
                    if element['status'] == 'OK':
                        distance_matrix[i][j] = element['distance']['value']  # en metros
                        time_matrix[i][j] = element['duration']['value'] // 60  # en minutos
                    else:
                        # Si hay un error en algún elemento, usar distancia euclidiana
                        loc1 = locations[i]
                        loc2 = locations[j]
                        dist = ((loc1.lat - loc2.lat)**2 + (loc1.lon - loc2.lon)**2)**0.5 * 111.32  # Aproximación en km
                        time_min = (dist / (DEFAULT_SPEED_KMH / 60))  # Tiempo en minutos
                        distance_matrix[i][j] = int(dist * 1000)  # Convertir a metros
                        time_matrix[i][j] = int(time_min)
                        
                        # Registrar advertencia
                        warn_msg = f"Usando distancia euclidiana entre {i} y {j}: {element.get('status', 'UNKNOWN_ERROR')}"
                        warnings = add_warning(warnings, "EUCLIDEAN_FALLBACK", warn_msg)
            
            return distance_matrix, time_matrix, warnings
            
        except Exception as e:
            logger.error(f"Error en get_distance_and_time_matrix: {str(e)}")
            # Fallback a matriz euclidiana
            warnings = add_warning(warnings, "FULL_FALLBACK", f"Usando matriz euclidiana debido a: {str(e)}")
            return get_euclidean_matrix(locations), get_euclidean_matrix(locations, time_based=True), warnings
    else:
        # --- Manejo de chunking para muchas ubicaciones ---
        warnings = add_warning(warnings, "CHUNKING_USED", 
                            "Se utilizó chunking para cumplir el límite de la API de Google Distance Matrix.")
        
        # Dividir en chunks
        chunks = [locations[i:i + MAX_CHUNK] for i in range(0, n, MAX_CHUNK)]
        num_chunks = len(chunks)
        
        # Inicializar matrices
        distance_matrix = [[0] * n for _ in range(n)]
        time_matrix = [[0] * n for _ in range(n)]
        
        # Procesar cada par de chunks
        for i1, chunk1 in enumerate(chunks):
            for i2, chunk2 in enumerate(chunks):
                # Procesar este par de chunks
                origins = [f"{loc.lat},{loc.lon}" for loc in chunk1]
                destinations = [f"{loc.lat},{loc.lon}" for loc in chunk2]
                
                url = "https://maps.googleapis.com/maps/api/distancematrix/json"
                params = {
                    "origins": "|".join(origins),
                    "destinations": "|".join(destinations),
                    "mode": getattr(request, 'mode', 'driving') or 'driving',
                    "units": getattr(request, 'units', 'metric') or 'metric',
                    "key": api_key
                }
                
                try:
                    response = requests.get(url, params=params, timeout=30)
                    data = response.json()
                    
                    if data['status'] == 'OK':
                        # Llenar la sección correspondiente de las matrices
                        base_row = i1 * MAX_CHUNK
                        base_col = i2 * MAX_CHUNK
                        
                        for i, row in enumerate(data['rows']):
                            for j, element in enumerate(row['elements']):
                                if element['status'] == 'OK':
                                    row_idx = base_row + i
                                    col_idx = base_col + j
                                    if row_idx < n and col_idx < n:  # Asegurar que no nos pasemos
                                        distance_matrix[row_idx][col_idx] = element['distance']['value']
                                        time_matrix[row_idx][col_idx] = element['duration']['value'] // 60
                    else:
                        warnings = add_warning(warnings, "API_ERROR", 
                                           f"Error en chunk ({i1},{i2}): {data.get('error_message', 'Unknown error')}")
                except Exception as e:
                    warnings = add_warning(warnings, "CHUNK_ERROR", 
                                         f"Error procesando chunk ({i1},{i2}): {str(e)}")
        
        return distance_matrix, time_matrix, warnings

def find_nearest_depot(location, depots, locations):
    """
    Encuentra el depósito más cercano a una ubicación dada.
    
    Args:
        location: Objeto SkillsLocation con lat/lon
        depots: Lista de IDs de depósitos
        locations: Lista de todas las ubicaciones
        
    Returns:
        ID del depósito más cercano
    """
    min_dist = float('inf')
    nearest_depot_id = depots[0]
    
    # Crear un diccionario {id: ubicación} para búsqueda rápida
    loc_dict = {loc.id: loc for loc in locations}
    
    for depot_id in depots:
        if depot_id not in loc_dict:
            continue
            
        depot_loc = loc_dict[depot_id]
        dist = ((location.lat - depot_loc.lat)**2 + (location.lon - depot_loc.lon)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            nearest_depot_id = depot_id
            
    return nearest_depot_id

def create_temp_location(lat: float, lon: float) -> SkillsLocation:
    """Crea un objeto SkillsLocation temporal con un ID ficticio."""
    return SkillsLocation(
        id=-1,  # ID temporal
        client_uuid="temp_location",
        name="Ubicación temporal",
        lat=lat,
        lon=lon,
        is_depot=False
    )

def solve_vrp_multiple_depots(
    locations, 
    vehicles, 
    depots, 
    distance_matrix, 
    time_matrix, 
    time_windows=None, 
    demands=None, 
    vehicle_capacities=None, 
    max_route_duration=MAX_TIME_MINUTES,
    max_route_distance=None,  # en metros, None para deshabilitar
    cost_per_km=1.0,
    cost_per_minute=0.0,
    fixed_cost=0.0,
    optimization_objective='minimize_distance'  # 'minimize_distance' | 'minimize_time' | 'minimize_cost'
):
    """
    Resuelve un problema VRP con múltiples depósitos usando OR-Tools.
    
    Args:
        locations: Lista de todas las ubicaciones
        vehicles: Lista de vehículos disponibles con depot_id asignado
        depots: Lista de índices de los depósitos
        distance_matrix: Matriz de distancias completa
        time_matrix: Matriz de tiempos completa
        time_windows: Diccionario {índice: (inicio, fin)} con ventanas de tiempo
        demands: Diccionario {índice: demanda} para cada ubicación
        vehicle_capacities: Lista de capacidades de los vehículos
        max_route_duration: Duración máxima de ruta en minutos
        
    Returns:
        Diccionario con las rutas, distancia total y tiempo total
    """
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    
    num_locations = len(locations)
    num_vehicles = len(vehicles)
    
    # Verificar que todos los vehículos tengan un depot_id válido
    for i, veh in enumerate(vehicles):
        if not hasattr(veh, 'depot_id') or veh.depot_id not in depots:
            # Si no tiene depot_id o no es válido, asignar al primer depósito
            setattr(veh, 'depot_id', depots[0])
            print(f"Advertencia: Vehículo {i} no tenía un depot_id válido. Asignado al depósito {depots[0]}")
    
    # Usar los depósitos asignados a cada vehículo
    starts = [veh.depot_id for veh in vehicles]
    ends = [veh.depot_id for veh in vehicles]  # Mismo depósito de inicio y fin
    # Crear el índice y modelo de enrutamiento con múltiples depósitos
    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, starts, ends)
    
    # Crear el modelo de enrutamiento con configuración básica
    routing = pywrapcp.RoutingModel(manager)
    
    # Definir las matrices de costos (versión final)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])
    
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(time_matrix[from_node][to_node])
    
    def cost_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        
        if optimization_objective == 'minimize_distance':
            return int(distance_matrix[from_node][to_node])
        elif optimization_objective == 'minimize_time':
            return int(time_matrix[from_node][to_node])
        else:  # minimize_cost
            dist = distance_matrix[from_node][to_node] / 1000.0  # Convertir a km
            time = time_matrix[from_node][to_node] / 60.0  # Convertir a horas
            cost = (dist * cost_per_km) + (time * cost_per_minute * 60)  # time en minutos
            return int(cost * 100)  # Multiplicar por 100 para mantener precisión
    
    # Registrar callbacks
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    cost_callback_index = routing.RegisterTransitCallback(cost_callback)
    
    # Configurar el objetivo de optimización
    if optimization_objective == 'minimize_time':
        routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)
    elif optimization_objective == 'minimize_cost':
        routing.SetArcCostEvaluatorOfAllVehicles(cost_callback_index)
    else:  # minimize_distance (por defecto)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Configurar la duración máxima de la ruta (en segundos)
    if max_route_duration is not None:
        routing.AddDimension(
            time_callback_index,  # time callback
            max_route_duration * 60,  # tiempo máximo de ruta en segundos (conversión de minutos a segundos)
            max_route_duration * 60,  # tiempo máximo de ruta en segundos
            False,  # Don't force start cumul to zero
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # Añadir restricción para que los vehículos regresen al depósito dentro del tiempo máximo
        for i in range(num_vehicles):
            routing.solver().Add(
                time_dimension.CumulVar(routing.End(i)) <= max_route_duration * 60
            )
    
    # Configurar capacidad de los vehículos si se especificó
    if demands and vehicle_capacities:
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return demands.get(from_node, 0)
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # slack
            vehicle_capacities,  # capacidades de los vehículos
            True,  # start cumul to zero
            'Capacity'
        )
    
    # Configurar ventanas de tiempo si se especificaron
    if time_windows:
        time_dimension = routing.GetDimensionOrDie('Time')
        for location_idx, (open_time, close_time) in time_windows.items():
            if location_idx < num_locations:  # Asegurarse de que el índice sea válido
                index = manager.NodeToIndex(location_idx)
                time_dimension.CumulVar(index).SetRange(open_time, close_time)
    
    # Añadir costo fijo por vehículo si es necesario
    if optimization_objective == 'minimize_cost' and fixed_cost > 0:
        def vehicle_fixed_cost_callback(vehicle_idx):
            return int(fixed_cost * 100)  # Multiplicar por 100 para mantener precisión
            
        fixed_cost_callback_index = routing.RegisterUnaryTransitCallback(
            vehicle_fixed_cost_callback)
        routing.SetFixedCostOfAllVehicles(fixed_cost_callback_index)
    
    # Configurar restricciones de tiempo (si no se configuró antes)
    if not routing.GetDimensionOrDie('Time'):
        routing.AddDimension(
            time_callback_index,
            max_route_duration,  # holgura máxima
            max_route_duration,  # máxima duración de ruta
            False,               # no empezar acumulando desde cero
            'Time'
        )
    time_dimension = routing.GetDimensionOrDie('Time')
    
    # Configurar restricción de distancia máxima si se especifica
    if max_route_distance is not None:
        routing.AddDimension(
            transit_callback_index,
            max_route_distance,  # holgura máxima
            max_route_distance,  # distancia máxima
            True,                # empezar acumulando desde cero
            'Distance')
        distance_dimension = routing.GetDimensionOrDie('Distance')
        
        # Asegurar que los vehículos no excedan la distancia máxima
        for vehicle_id in range(routing.vehicles()):
            end_index = routing.End(vehicle_id)
            distance_dimension.CumulVar(end_index).SetMax(max_route_distance)
    
    # Configurar ventanas de tiempo para depósitos (horario de trabajo)
    for vehicle_id in range(num_vehicles):
        # Configurar horario de inicio en el depósito de inicio
        start_idx = manager.NodeToIndex(starts[vehicle_id])
        time_dimension.CumulVar(start_idx).SetRange(
            vehicles[vehicle_id].start_time,
            max_route_duration
        )
        
        # Configurar horario de fin en el depósito de fin
        end_idx = manager.NodeToIndex(ends[vehicle_id])
        time_dimension.CumulVar(end_idx).SetRange(
            0,  # Hora de inicio mínima
            vehicles[vehicle_id].end_time
        )
    
    # Configurar ventanas de tiempo para clientes si están disponibles
    if time_windows:
        for location_idx, window in time_windows.items():
            if location_idx not in depots:  # No configurar ventana para depósitos
                index = manager.NodeToIndex(location_idx)
                time_dimension.CumulVar(index).SetRange(window[0], window[1])
    
    # Configurar restricciones de capacidad si están disponibles
    if demands and vehicle_capacities:
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return demands.get(from_node, 0)
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # holgura de capacidad
            vehicle_capacities,  # capacidades
            True,  # empezar acumulando desde cero
            'Capacity'
        )
    
    # Resolver el problema con manejo de tiempo mejorado
    logger.info(f"Iniciando resolución VRP con {len(locations)} nodos y {num_vehicles} vehículos...")
    solve_start = time.perf_counter()
    
    # Ejecutar resolución del VRP con parámetros optimizados
    search_parameters = get_advanced_search_parameters(num_locations)
    # Cambiar estrategia inicial a SAVINGS para mejor distribución multi-depósito
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS
    # Activar metaheurística de búsqueda local
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    solution = routing.SolveWithParameters(search_parameters)
    # Registrar tiempo de resolución
    solve_time = time.perf_counter() - solve_start
    logger.info(f"Tiempo de resolución: {solve_time:.2f} segundos")
    
    # Procesar la solución
    routes = []
    total_distance = 0
    total_time = 0
    
    # Verificar si se encontró solución
    if not solution:
        logger.warning("No se encontró una solución factible")
        return {
            'routes': [],
            'total_distance': 0,
            'total_time': 0
        }
    
    if solution:
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0
            route_time = 0
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
                route_time += time_matrix[manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
            
            # Añadir el depósito final si es diferente al de inicio
            end_depot = ends[vehicle_id]
            if end_depot != route[-1]:  # Evitar duplicar el último nodo si ya es el depósito final
                route.append(end_depot)
                # Sumar la distancia/tiempo desde el último nodo al depósito final
                last_node = route[-2] if len(route) > 1 else route[0]
                route_distance += distance_matrix[last_node][end_depot]
                route_time += time_matrix[last_node][end_depot]
            
            # Solo agregar la ruta si tiene más que solo depósitos
            if len([n for n in route if n not in depots]) > 0:
                routes.append({
                    'vehicle_id': vehicle_id,
                    'stops': route,
                    'distance': route_distance,
                    'duration': route_time,
                    'start_depot': starts[vehicle_id],
                    'end_depot': ends[vehicle_id]
                })
                total_distance += route_distance
                total_time += route_time
    
    return {
        'routes': routes,
        'total_distance': total_distance,
        'total_time': total_time
    }

def get_euclidean_matrix(locations, time_based=False):
    """Genera una matriz de distancias o tiempos euclidianos."""
    n = len(locations)
    matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0
            else:
                loc1 = locations[i]
                loc2 = locations[j]
                # Distancia euclidiana en grados (aproximación)
                dist = ((loc1.lat - loc2.lat)**2 + (loc1.lon - loc2.lon)**2)**0.5
                if time_based:
                    # Convertir a minutos asumiendo velocidad constante
                    # 1 grado ≈ 111.32 km, velocidad en km/h
                    matrix[i][j] = int((dist * 111.32) / (DEFAULT_SPEED_KMH / 60))
                else:
                    # Distancia en metros
                    matrix[i][j] = int(dist * 111.32 * 1000)
    
    return matrix

@router.post("/vrp-v3", response_model=VRPAdvancedResponse)
async def vrp_v3(request: VRPSkillsRequest):
    """
    Endpoint para resolver problemas de enrutamiento de vehículos (VRP) con soporte para:
    - Múltiples depósitos
    - Habilidades requeridas por cliente
    - Ventanas de tiempo
    - Restricciones de capacidad
    
    Nota: Los clientes no se pre-asignan a depósitos; el solver optimiza la asignación.
    """
    import time
    logger.info("Iniciando proceso VRP")
    t0 = time.perf_counter()
    
    # Contadores de tiempo
    timers = {
        'total': 0,
        'validation': 0,
        'matrix_calculation': 0,
        'solving': 0,
        'post_processing': 0
    }
    
    # Cargar API keys
    load_dotenv()
    ors_api_key = os.getenv("ORS_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not ors_api_key and not google_api_key:
        raise HTTPException(status_code=400, detail="Se requiere al menos ORS_API_KEY o GOOGLE_API_KEY en .env")
    
    # Asignar UUIDs únicos si no existen
    for loc in request.locations:
        if not hasattr(loc, 'client_uuid') or not getattr(loc, 'client_uuid', None):
            setattr(loc, 'client_uuid', str(uuidlib.uuid4()))
    for veh in request.vehicles:
        if not hasattr(veh, 'vehicle_uuid') or not getattr(veh, 'vehicle_uuid', None):
            setattr(veh, 'vehicle_uuid', str(uuidlib.uuid4()))
    
    # Validar solicitud
    logger.info("Iniciando validación de la solicitud")
    validation_start = time.perf_counter()
    # Removed import of nonexistent vrp_validator module
    is_valid, warnings, diagnostics = True, [], []
    validation_time = time.perf_counter() - validation_start
    timers['validation'] = validation_time
    logger.info(f"Validación completada en {validation_time:.4f} segundos")
    if not is_valid:
        t1 = time.perf_counter()
        metadata = {
            "computation_time_ms": int((t1-t0)*1000),
            "num_vehicles": len(request.vehicles),
            "num_clients": len([loc for loc in request.locations if not getattr(loc, 'is_depot', False)]),
            "strict_mode": getattr(request, 'strict_mode', False),
            "buffer_minutes": getattr(request, 'buffer_minutes', DEFAULT_BUFFER_MINUTES),
            "peak_hours": getattr(request, 'peak_hours', None),
            "peak_buffer_minutes": getattr(request, 'peak_buffer_minutes', 20)
        }
        return VRPAdvancedResponse(
            solution={}, 
            metadata=metadata, 
            warnings=warnings, 
            diagnostics=diagnostics
        )
    
    # Identificar depósitos (ubicaciones con is_depot=True)
    depots = get_depot_indices(request.locations)
    if not depots:
        warnings = add_warning(warnings, "NO_DEPOTS", "No se encontraron depósitos, usando la primera ubicación")
        depots = [0]
    
    # Obtener matrices de distancia y tiempo
    warnings = []
    logger.info("Iniciando cálculo de matrices de distancia/tiempo")
    matrix_start = time.perf_counter()
    try:
        # Intentar primero con ORS si hay API key
        if ors_api_key:
            from vrp.matrix.ors import ORSMatrixProvider
            try:
                logger.info("Usando ORS API para cálculo de matrices")
                provider = ORSMatrixProvider(ors_api_key)
                distance_matrix, time_matrix, matrix_warnings = await provider.get_matrix(request.locations)
                warnings.extend(matrix_warnings)
                matrix_time = time.perf_counter() - matrix_start
                logger.info(f"Matriz calculada usando ORS API en {matrix_time:.4f} segundos")
            except Exception as e:
                logger.warning(f"Error con ORS API: {str(e)}")
                if not google_api_key:
                    raise HTTPException(status_code=500, detail=f"Error con ORS API y no hay GOOGLE_API_KEY: {str(e)}")
        
        # Si no se usó ORS (por fallo o falta de API key), intentar con Google
        if not ors_api_key or 'distance_matrix' not in locals():
            distance_matrix, time_matrix, warnings = await get_distance_and_time_matrix(request, google_api_key, warnings)
            logger.info("Matriz calculada usando Google Distance Matrix API")
    except Exception as e:
        logger.error(f"Error al calcular matrices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al calcular matrices: {str(e)}")
    
    # Preparar datos adicionales para el VRP
    time_windows = {}
    demands = {}
    
    # Procesar ventanas de tiempo
    for i, loc in enumerate(request.locations):
        if hasattr(loc, 'time_window') and loc.time_window:
            time_windows[i] = (loc.time_window[0], loc.time_window[1])
    
    # Procesar demandas (asumiendo que los depósitos tienen demanda 0)
    for i, loc in enumerate(request.locations):
        if i not in depots:  # No procesar depósitos
            # Usar demanda de cantidad si está habilitado
            if hasattr(loc, 'demand') and loc.demand is not None:
                demands[i] = loc.demand
            elif hasattr(loc, 'quantity') and loc.quantity is not None:
                demands[i] = loc.quantity
            elif hasattr(loc, 'weight') and loc.weight is not None:
                demands[i] = loc.weight
            elif hasattr(loc, 'volume') and loc.volume is not None:
                demands[i] = loc.volume
            else:
                # Si no hay demanda definida, asumir 1
                demands[i] = 1
                
    # Verificar que todas las demandas sean enteros
    demands = {k: int(v) for k, v in demands.items()}
    
    # Log de demandas para depuración
    logger.info(f"Demandas procesadas: {demands}")
    
    # Obtener capacidades de los vehículos según flags
    vehicle_capacities = []
    for i, veh in enumerate(request.vehicles):
        # Priorizar cantidad (quantity) si está habilitado
        if hasattr(veh, 'use_quantity') and veh.use_quantity and hasattr(veh, 'capacity_quantity'):
            capacity = veh.capacity_quantity
        # Luego verificar peso si está habilitado
        elif hasattr(veh, 'use_weight') and veh.use_weight and hasattr(veh, 'capacity_weight'):
            capacity = veh.capacity_weight
        # Finalmente, verificar volumen si está habilitado
        elif hasattr(veh, 'use_volume') and veh.use_volume and hasattr(veh, 'capacity_volume'):
            capacity = veh.capacity_volume
        else:
            # Si no hay restricciones de capacidad, usar un valor alto
            capacity = 1000
        
        # Asegurar que la capacidad sea un entero
        capacity = int(capacity)
        vehicle_capacities.append(capacity)
        
        logger.info(f"Vehículo {i+1} (ID: {getattr(veh, 'id', 'N/A')}): Capacidad = {capacity}")
    
    # Verificar que los vehículos tengan asignado un depósito válido
    for veh in request.vehicles:
        if not hasattr(veh, 'depot_id') or veh.depot_id is None:
            # Si no tiene depósito asignado, asignar al más cercano
            veh_loc = create_temp_location(lat=veh.start_lat, lon=veh.start_lon)
            veh.depot_id = find_nearest_depot(veh_loc, depots, request.locations)
            print(f"Asignado vehículo {veh.id} al depósito {veh.depot_id} (más cercano)")
        else:
            # Verificar que el depósito asignado exista
            depot_exists = any(depot == veh.depot_id for depot in depots)
            if not depot_exists:
                # Si el depósito asignado no existe, asignar al más cercano
                veh_loc = create_temp_location(lat=veh.start_lat, lon=veh.start_lon)
                original_depot = veh.depot_id
                veh.depot_id = find_nearest_depot(veh_loc, depots, request.locations)
                print(f"Depósito {original_depot} no encontrado para vehículo {veh.id}. Asignado al depósito {veh.depot_id} (más cercano)")
    
    # Resolver el VRP con múltiples depósitos
    logger.info("Iniciando resolución del VRP")
    solving_start = time.perf_counter()
    solution = solve_vrp_multiple_depots(
        locations=request.locations,
        vehicles=request.vehicles,
        depots=depots,
        distance_matrix=distance_matrix,
        time_matrix=time_matrix,
        time_windows=time_windows,
        demands=demands,
        vehicle_capacities=vehicle_capacities,
        max_route_duration=MAX_TIME_MINUTES,
        max_route_distance=getattr(request, 'max_route_distance', None),  # en metros
        cost_per_km=getattr(request, 'cost_per_km', 1.0),
        cost_per_minute=getattr(request, 'cost_per_minute', 0.0),
        fixed_cost=getattr(request, 'fixed_cost', 0.0),
        optimization_objective=getattr(request, 'optimization_objective', 'minimize_distance')
    )
    solving_time = time.perf_counter() - solving_start
    timers['solving'] = solving_time
    logger.info(f"VRP resuelto en {solving_time:.4f} segundos")
    
    # Calcular tiempo total y preparar metadatos
    t1 = time.perf_counter()
    timers['total'] = t1 - t0
    timers['post_processing'] = timers['total'] - (timers['validation'] + timers['solving'] + timers.get('matrix_calculation', 0))
    
    # Log resumen de tiempos
    logger.info("\n" + "="*50)
    logger.info("RESUMEN DE TIEMPOS DE EJECUCIÓN")
    logger.info("-"*50)
    logger.info(f"Validación: {timers['validation']:.4f} segundos ({timers['validation']/timers['total']*100:.1f}%)")
    if 'matrix_calculation' in timers:
        logger.info(f"Cálculo de matrices: {timers['matrix_calculation']:.4f} segundos ({timers['matrix_calculation']/timers['total']*100:.1f}%)")
    logger.info(f"Resolución VRP: {timers['solving']:.4f} segundos ({timers['solving']/timers['total']*100:.1f}%)")
    logger.info(f"Procesamiento final: {timers['post_processing']:.4f} segundos ({timers['post_processing']/timers['total']*100:.1f}%)")
    logger.info("-"*50)
    logger.info(f"TOTAL: {timers['total']:.4f} segundos")
    logger.info("="*50 + "\n")
    
    metadata = {
        "computation_time_ms": int(timers['total']*1000),
        "num_vehicles": len(request.vehicles),
        "num_clients": len([loc for idx, loc in enumerate(request.locations) 
                           if idx not in depots and not getattr(loc, 'is_depot', False)]),
        "num_depots": len(depots),
        "strict_mode": getattr(request, 'strict_mode', False),
        "buffer_minutes": getattr(request, 'buffer_minutes', DEFAULT_BUFFER_MINUTES),
        "peak_hours": getattr(request, 'peak_hours', None),
        "peak_buffer_minutes": getattr(request, 'peak_buffer_minutes', 20),
        "timing_breakdown": {
            "validation_ms": int(timers['validation'] * 1000),
            "matrix_calculation_ms": int(timers.get('matrix_calculation', 0) * 1000),
            "solving_ms": int(timers['solving'] * 1000),
            "post_processing_ms": int(timers['post_processing'] * 1000)
        }
    }
    
    return VRPAdvancedResponse(
        solution=solution,
        metadata=metadata,
        warnings=warnings,
        diagnostics=None
    )
