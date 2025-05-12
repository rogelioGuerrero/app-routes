
from fastapi import APIRouter, HTTPException
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from typing import List
from schemas_skills import VRPSkillsRequest, SkillsLocation, SkillsVehicle
from route_polyline_utils import get_route_polyline_and_geojson
from schemas import VRPAdvancedResponse
from vrp_utils import min_to_hhmm, filter_viable_clients, add_warning, warn_matrix_fallback, warn_no_viable_clients, validate_full_request
import os
import requests
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import logging
from vrp_constants import DEFAULT_SPEED_KMH, MAX_TIME_MINUTES, SKILL_PENALTY, DEFAULT_BUFFER_MINUTES

# --- Configurable polyline threshold ---
MAX_POLYLINE_ROUTES = 10  # Cambia aquí si quieres más o menos rutas con polylines

logger = logging.getLogger(__name__)

router = APIRouter()

# --- Función modular para obtención de matrices ---
def get_distance_and_time_matrix(request, api_key, warnings):
    """
    Obtiene la matriz de distancias y tiempos usando Google Distance Matrix API o fallback euclidiano.
    Maneja errores específicos y logging. Soporta chunking si hay más de 10 ubicaciones.
    """
    from vrp_utils import add_warning
    import math
    locations = request.locations
    n = len(locations)
    MAX_CHUNK = 10
    if n <= MAX_CHUNK:
        # --- Comportamiento original ---
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
            status = data.get("status")
            if status != "OK":
                logger.warning(f"Respuesta inválida de Google Distance Matrix: {data}")
                if status == "OVER_QUERY_LIMIT":
                    raise HTTPException(status_code=429, detail="Límite de cuota de Google alcanzado")
                elif status == "INVALID_REQUEST":
                    raise HTTPException(status_code=400, detail="Solicitud inválida a Google Distance Matrix")
                elif status == "REQUEST_DENIED":
                    raise HTTPException(status_code=403, detail="Acceso denegado a Google Distance Matrix")
                else:
                    raise Exception(f"Respuesta inválida de Google Distance Matrix: {data}")
            distance_matrix = []
            time_matrix = []
            for row in data["rows"]:
                distance_row = [el["distance"]["value"]/1000 if el.get("distance") else 1e6 for el in row["elements"]]
                time_row = [el["duration"]["value"]//60 if el.get("duration") else 1e6 for el in row["elements"]]
                distance_matrix.append(distance_row)
                time_matrix.append(time_row)
            return distance_matrix, time_matrix
        except HTTPException as he:
            logger.warning(f"HTTPException en consulta de matriz: {he.detail}")
            raise
        except Exception as e:
            logger.warning(f"Fallo la API de Google o error inesperado, se usa fallback euclidiano: {e}")
            warn_matrix_fallback(warnings, str(e))
            n = len(locations)
            distance_matrix = []
            time_matrix = []
            for i in range(n):
                d_row = []
                t_row = []
                for j in range(n):
                    if i == j:
                        d_row.append(0)
                        t_row.append(0)
                    else:
                        lat1, lon1 = locations[i].lat, locations[i].lon
                        lat2, lon2 = locations[j].lat, locations[j].lon
                        dist = ((lat1-lat2)**2 + (lon1-lon2)**2)**0.5 * 111  # Aprox km
                        d_row.append(dist)
                        t_row.append(int(dist/DEFAULT_SPEED_KMH*60))  # velocidad configurable
                distance_matrix.append(d_row)
                time_matrix.append(t_row)
            # --- Ajuste de congestión por horas pico ---
            peak_hours = getattr(request, 'peak_hours', None)
            peak_multiplier = getattr(request, 'peak_multiplier', 1.3)
            if peak_hours:
                for i in range(n):
                    for j in range(n):
                        time_matrix[i][j] = int(time_matrix[i][j] * peak_multiplier)
            return distance_matrix, time_matrix
    else:
        # --- Chunking ---
        add_warning(warnings, "CHUNKING_USED", "Se utilizó chunking para cumplir el límite de la API de Google Distance Matrix.")
        origins_list = [f"{loc.lat},{loc.lon}" for loc in locations]
        distance_matrix = [[None for _ in range(n)] for _ in range(n)]
        time_matrix = [[None for _ in range(n)] for _ in range(n)]
        num_chunks = math.ceil(n / MAX_CHUNK)
        try:
            for i in range(num_chunks):
                orig_start = i * MAX_CHUNK
                orig_end = min((i+1)*MAX_CHUNK, n)
                chunk_origins = origins_list[orig_start:orig_end]
                for j in range(num_chunks):
                    dest_start = j * MAX_CHUNK
                    dest_end = min((j+1)*MAX_CHUNK, n)
                    chunk_destinations = origins_list[dest_start:dest_end]
                    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
                    params = {
                        "origins": "|".join(chunk_origins),
                        "destinations": "|".join(chunk_destinations),
                        "mode": getattr(request, 'mode', 'driving') or 'driving',
                        "units": getattr(request, 'units', 'metric') or 'metric',
                        "key": api_key
                    }
                    response = requests.get(url, params=params, timeout=30)
                    if response.status_code != 200:
                        logger.error(f"Error HTTP consultando Google Distance Matrix: {response.status_code} {response.text}")
                        raise Exception(f"Error consultando Google Distance Matrix: {response.text}")
                    data = response.json()
                    status = data.get("status")
                    if status != "OK":
                        logger.warning(f"Respuesta inválida de Google Distance Matrix: {data}")
                        if status == "OVER_QUERY_LIMIT":
                            raise HTTPException(status_code=429, detail="Límite de cuota de Google alcanzado")
                        elif status == "INVALID_REQUEST":
                            raise HTTPException(status_code=400, detail="Solicitud inválida a Google Distance Matrix")
                        elif status == "REQUEST_DENIED":
                            raise HTTPException(status_code=403, detail="Acceso denegado a Google Distance Matrix")
                        else:
                            raise Exception(f"Respuesta inválida de Google Distance Matrix: {data}")
                    # Llenar la submatriz en la matriz final
                    for oi, row in enumerate(data["rows"]):
                        for di, el in enumerate(row["elements"]):
                            global_i = orig_start + oi
                            global_j = dest_start + di
                            if global_i < n and global_j < n:
                                distance_matrix[global_i][global_j] = el["distance"]["value"]/1000 if el.get("distance") else 1e6
                                time_matrix[global_i][global_j] = el["duration"]["value"]//60 if el.get("duration") else 1e6
            # Reemplaza None por 1e6 si alguna celda quedó vacía
            for i in range(n):
                for j in range(n):
                    if distance_matrix[i][j] is None:
                        distance_matrix[i][j] = 1e6
                    if time_matrix[i][j] is None:
                        time_matrix[i][j] = 1e6
            return distance_matrix, time_matrix
        except HTTPException as he:
            logger.warning(f"HTTPException en consulta de matriz: {he.detail}")
            raise
        except Exception as e:
            logger.warning(f"Fallo la API de Google o error inesperado durante chunking, se usa fallback euclidiano: {e}")
            warn_matrix_fallback(warnings, str(e))
            distance_matrix = []
            time_matrix = []
            for i in range(n):
                d_row = []
                t_row = []
                for j in range(n):
                    if i == j:
                        d_row.append(0)
                        t_row.append(0)
                    else:
                        lat1, lon1 = locations[i].lat, locations[i].lon
                        lat2, lon2 = locations[j].lat, locations[j].lon
                        dist = ((lat1-lat2)**2 + (lon1-lon2)**2)**0.5 * 111  # Aprox km
                        d_row.append(dist)
                        t_row.append(int(dist/DEFAULT_SPEED_KMH*60))
                distance_matrix.append(d_row)
                time_matrix.append(t_row)
            # --- Ajuste de congestión por horas pico ---
            peak_hours = getattr(request, 'peak_hours', None)
            peak_multiplier = getattr(request, 'peak_multiplier', 1.3)
            if peak_hours:
                for i in range(n):
                    for j in range(n):
                        time_matrix[i][j] = int(time_matrix[i][j] * peak_multiplier)
            return distance_matrix, time_matrix


@router.post("/vrp-v2", response_model=VRPAdvancedResponse)
async def vrp_v2(request: VRPSkillsRequest):
    print("[DEBUG] INICIO vrp_v2")
    """
    Nuevo endpoint VRP vitaminado (v2): modularización del filtrado de clientes, base para mejoras futuras.
    """
    import uuid as uuidlib
    t0 = time.perf_counter()
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="GOOGLE_API_KEY no encontrada en .env")

    # --- GENERA vehicle_uuid PARA CADA VEHÍCULO Y client_uuid PARA CADA CLIENTE SI NO LO TIENEN ---
    for loc in request.locations:
        if not hasattr(loc, 'client_uuid') or not getattr(loc, 'client_uuid', None):
            setattr(loc, 'client_uuid', str(uuidlib.uuid4()))
    for veh in request.vehicles:
        if not hasattr(veh, 'vehicle_uuid') or not getattr(veh, 'vehicle_uuid', None):
            setattr(veh, 'vehicle_uuid', str(uuidlib.uuid4()))
    print("[DEBUG] UUIDs asignados")

    # --- VALIDACIÓN CENTRALIZADA ---
    is_valid, warnings, diagnostics = validate_full_request(request)
    print(f"[DEBUG] Resultado validación: is_valid={is_valid}, warnings={warnings}, diagnostics={diagnostics}")
    if not is_valid:
        t1 = time.perf_counter()
        metadata = {
            "computation_time_ms": int((t1-t0)*1000),
            "num_vehicles": request.num_vehicles if hasattr(request, 'num_vehicles') else len(request.vehicles),
            "num_clients": len(request.locations) - 1 if hasattr(request, 'locations') else None,
            "strict_mode": getattr(request, 'strict_mode', False),
            "buffer_minutes": getattr(request, 'buffer_minutes', DEFAULT_BUFFER_MINUTES),
            "peak_hours": getattr(request, 'peak_hours', None),
            "peak_buffer_minutes": getattr(request, 'peak_buffer_minutes', 20)
        }
        return VRPAdvancedResponse(solution={}, metadata=metadata, warnings=warnings, diagnostics=diagnostics)

    warnings = warnings or []

    # --- FILTRO PREVIO DE CLIENTES INVIABLES (skills y capacidades) ---
    depot_index = request.depot if hasattr(request, 'depot') else 0
    viable_locations, excluded_clients = filter_viable_clients(request.locations, request.vehicles, depot_index)
    print(f"[DEBUG] len(viable_locations): {len(viable_locations)}, type: {[type(x) for x in viable_locations]}")
    print(f"[DEBUG] len(excluded_clients): {len(excluded_clients)}, type: {[type(x) for x in excluded_clients]}")
    if len(viable_locations) <= 1:
        warn_no_viable_clients(warnings, excluded_clients)
        t1 = time.perf_counter()
        metadata = {
            "computation_time_ms": int((t1-t0)*1000),
            "num_vehicles": request.num_vehicles if hasattr(request, 'num_vehicles') else len(request.vehicles),
            "num_clients": len(request.locations) - 1 if hasattr(request, 'locations') else None,
            "strict_mode": getattr(request, 'strict_mode', False),
            "buffer_minutes": getattr(request, 'buffer_minutes', DEFAULT_BUFFER_MINUTES),
            "peak_hours": getattr(request, 'peak_hours', None),
            "peak_buffer_minutes": getattr(request, 'peak_buffer_minutes', 20)
        }
        return VRPAdvancedResponse(solution={}, metadata=metadata, warnings=warnings, excluded_clients=excluded_clients)
    request.locations = viable_locations

    # --- MATRIZ DE DISTANCIAS ---
    print("[DEBUG] Antes de get_distance_and_time_matrix")
    distance_matrix, time_matrix = get_distance_and_time_matrix(request, api_key, warnings)
    print("[DEBUG] Matrices obtenidas")
    print(f"[ASSERT] distance_matrix: {distance_matrix}")
    print(f"[ASSERT] time_matrix: {time_matrix}")
    if distance_matrix is None or time_matrix is None:
        raise HTTPException(status_code=500, detail="Error crítico: Las matrices de distancia o tiempo son None. Verifica la API KEY o el fallback.")
    if len(distance_matrix) == 0 or len(distance_matrix[0]) == 0:
        raise HTTPException(status_code=500, detail="Error crítico: La matriz de distancias está vacía.")
    if len(time_matrix) == 0 or len(time_matrix[0]) == 0:
        raise HTTPException(status_code=500, detail="Error crítico: La matriz de tiempos está vacía.")

    # LOG de depuración previo al solver


    # --- RESPUESTA SOLO MATRICES SI detail_level=minimal ---
    if getattr(request, 'detail_level', 'full') == 'minimal':
        t1 = time.perf_counter()
        return VRPAdvancedResponse(
            solution={
                "distance_matrix": distance_matrix,
                "time_matrix": time_matrix,
            },
            metadata={"computation_time_ms": int((t1-t0)*1000)},
            warnings=warnings if warnings else None
        )

    # --- LIMPIEZA DE DATOS PARA EL SOLVER ---
    def clean_location(loc):
        return {
            'id': loc.id,
            'name': getattr(loc, 'name', None),
            'lat': loc.lat,
            'lon': loc.lon,
            'demand': getattr(loc, 'demand', 0),
            'weight': getattr(loc, 'weight', 0.0),
            'volume': getattr(loc, 'volume', 0.0),
            'time_window': getattr(loc, 'time_window', [0, 1440]),
            'service_time': getattr(loc, 'service_time', 0),
            'required_skills': getattr(loc, 'required_skills', [])
        }
    def clean_vehicle(veh):
        return {
            'id': veh.id,
            'start_lat': veh.start_lat,
            'start_lon': veh.start_lon,
            'end_lat': getattr(veh, 'end_lat', veh.start_lat),
            'end_lon': getattr(veh, 'end_lon', veh.start_lon),
            'provided_skills': getattr(veh, 'provided_skills', []),
            'capacity_weight': getattr(veh, 'capacity_weight', 0.0),
            'capacity_volume': getattr(veh, 'capacity_volume', 0.0),
            'capacity_quantity': getattr(veh, 'capacity_quantity', 0),
            'start_time': getattr(veh, 'start_time', 0),
            'end_time': getattr(veh, 'end_time', 1440),
            'use_quantity': getattr(veh, 'use_quantity', False),
            'use_weight': getattr(veh, 'use_weight', False),
            'use_volume': getattr(veh, 'use_volume', False)
        }
    clean_locations = [clean_location(loc) for loc in request.locations]
    clean_vehicles = [clean_vehicle(veh) for veh in request.vehicles]
    print(f"[DEBUG] clean_locations: {clean_locations}")
    print(f"[DEBUG] clean_vehicles: {clean_vehicles}")

    # --- SOLVER VRP ---
    n = len(clean_locations)
    manager = pywrapcp.RoutingIndexManager(n, request.num_vehicles, request.depot)
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # --- DIMENSIONES DE CAPACIDAD AVANZADAS (mejora) ---
    from vrp_utils import add_capacity_dimensions
    add_capacity_dimensions(routing, manager, request)

    # --- SKILLS: RESTRICCIÓN ESTRICTA O PENALIZACIÓN (configurable) ---
    vehicle_skills = [set(getattr(v, 'provided_skills', []) or set(getattr(v, 'skills', []) or [])) for v in request.vehicles]
    location_skills = [set(getattr(l, 'required_skills', []) or set(getattr(l, 'skills', []) or [])) for l in request.locations]

    skills_penalty_mode = getattr(request, 'skills_penalty_mode', 'strict')
    if skills_penalty_mode == 'penalty':
        from vrp_utils import apply_skills_penalty, VRPConstants
        apply_skills_penalty(routing, manager, request, vehicle_skills, location_skills, penalty=VRPConstants.SKILL_PENALTY)
    else:
        for node_idx, req_skills in enumerate(location_skills):
            if not req_skills:
                continue
            for vehicle_idx, prov_skills in enumerate(vehicle_skills):
                if not req_skills.issubset(prov_skills):
                    routing.VehicleVar(manager.NodeToIndex(node_idx)).RemoveValue(vehicle_idx)

    # --- TIME WINDOWS Y SERVICE TIME ---
    time_windows = []
    service_times = []
    for idx, loc in enumerate(request.locations):
        tw = loc.time_window if loc.time_window and len(loc.time_window) == 2 else [420, 1080]
        time_windows.append(tw)
        service_times.append(getattr(loc, 'service_time', 5) if idx != request.depot else 0)

    # --- DIMENSIÓN DE TIEMPO (AddDimension) ---
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(time_matrix[from_node][to_node] + service_times[from_node])
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        1440,  # slack máximo permitido
        1440,  # tiempo máximo por ruta
        False,
        'Time')
    time_dimension = routing.GetDimensionOrDie('Time')
    time_dimension.SetSlackCostCoefficientForAllVehicles(100)
    adjust_depot_window = getattr(request, 'adjust_depot_window', False)
    for vehicle_id, vehicle in enumerate(request.vehicles):
        depot_index = routing.Start(vehicle_id)
        start_time_orig = getattr(vehicle, 'start_time', 420)
        end_time = getattr(vehicle, 'end_time', 1080)
        if adjust_depot_window:
            # Para cada cliente (excepto depósito), calcula la hora más temprana de salida posible
            min_departure = start_time_orig
            for idx, tw in enumerate(time_windows):
                if idx == request.depot:
                    continue
                travel_time = time_matrix[request.depot][idx]
                # No puede salir antes del horario laboral ni antes de lo necesario para llegar justo a tiempo
                candidate_departure = max(start_time_orig, tw[0] - travel_time)
                min_departure = max(min_departure, candidate_departure)
            start_time = min_departure
        else:
            start_time = start_time_orig
        time_dimension.CumulVar(depot_index).SetRange(start_time, end_time)
    # Resto de nodos normales
    for idx, window in enumerate(time_windows):
        if idx == request.depot:
            continue
        index = manager.NodeToIndex(idx)
        time_dimension.CumulVar(index).SetRange(window[0], window[1])
    for idx, st in enumerate(service_times):
        index = manager.NodeToIndex(idx)
        time_dimension.SlackVar(index).SetValue(st)

    # --- PARÁMETROS DEL SOLVER ---
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = getattr(request, 'solver_timeout', 30)
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)

    # --- DIAGNÓSTICO SI NO HAY SOLUCIÓN ---
    if not solution:
        out_of_time = []
        for i, loc in enumerate(request.locations):
            if i == request.depot:
                continue
            tw = loc.time_window if loc.time_window else [0, 1440]
            service_time = service_times[i]
            compatible = False
            for v in request.vehicles:
                if not (v.start_time <= tw[1] and v.end_time >= tw[0]):
                    continue
                latest_arrival = min(tw[1], v.end_time - service_time)
                if latest_arrival >= max(tw[0], v.start_time):
                    compatible = True
                    break
            if not compatible:
                out_of_time.append(loc.name)
        add_warning(warnings, code="NO_FEASIBLE_SOLUTION", message="No se pudo encontrar una solución factible.", context={"out_of_time": out_of_time} if out_of_time else None)
        t1 = time.perf_counter()
        metadata = {
            "computation_time_ms": int((t1-t0)*1000),
            "num_vehicles": getattr(request, 'num_vehicles', len(getattr(request, 'vehicles', []))),
            "num_clients": len(getattr(request, 'locations', [])) - 1 if hasattr(request, 'locations') else None,
            "strict_mode": getattr(request, 'strict_mode', False),
            "buffer_minutes": getattr(request, 'buffer_minutes', 10),
            "peak_hours": getattr(request, 'peak_hours', None),
            "peak_buffer_minutes": getattr(request, 'peak_buffer_minutes', 20)
        }
        return VRPAdvancedResponse(solution={}, metadata=metadata, warnings=warnings)

    from vrp_utils import min_to_hhmm

    # --- Construcción de solución avanzada (rutas, polylines, detalles) ---
    t_poly_start = time.perf_counter()
    num_vehicles = getattr(request, 'num_vehicles', len(getattr(request, 'vehicles', [])))
    include_polylines = getattr(request, 'include_polylines', True)
    polyline_warning = None
    diagnostics = None
    solution_data = None

    if include_polylines and num_vehicles > MAX_POLYLINE_ROUTES:
        polyline_warning = {
            "code": "POLYLINE_LIMIT_EXCEEDED",
            "message": f"Haz superado el límite de rutas ({MAX_POLYLINE_ROUTES}) con polylines. ¡Contáctanos para ampliar tu servicio!",
            "context": {"num_vehicles": num_vehicles, "polyline_limit": MAX_POLYLINE_ROUTES}
        }
        include_polylines = False
        # diagnostics puede ser útil para frontend avanzado
        diagnostics = {"polyline_generation": "skipped", "reason": polyline_warning["message"]}

    # Si no se generan polylines, llama a build_vrp_solution con un flag especial
    from vrp_utils import build_vrp_solution
    if not include_polylines:
        # Patch: build_vrp_solution espera polylines por defecto, así que las forzamos a None
        def build_vrp_solution_no_poly(solution, routing, manager, request, time_dimension, time_windows, service_times, distance_matrix, time_matrix):
            data = build_vrp_solution(solution, routing, manager, request, time_dimension, time_windows, service_times, distance_matrix, time_matrix)
            data["route_polylines"] = None
            return data
        solution_data = build_vrp_solution_no_poly(solution, routing, manager, request, time_dimension, time_windows, service_times, distance_matrix, time_matrix)
    else:
        # Polyline por defecto: build_vrp_solution ya llama a get_route_polyline_and_geojson
        # pero lo hace secuencial, así que lo paralelizamos aquí si hay varias rutas
        # Primero obtenemos la estructura base
        solution_data = build_vrp_solution(solution, routing, manager, request, time_dimension, time_windows, service_times, distance_matrix, time_matrix)
        # Si hay rutas y polylines, recalculamos en paralelo
        if solution_data and solution_data.get("routes"):
            routes = [r["route"] for r in solution_data["routes"]]
            locations = request.locations
            async def compute_polylines(routes, locations):
                def get_polyline(route):
                    latlons = [(locations[idx].lat, locations[idx].lon) for idx in route]
                    try:
                        polyline, _ = get_route_polyline_and_geojson(latlons)
                    except Exception:
                        polyline = None
                    return polyline
                with ThreadPoolExecutor() as executor:
                    loop = asyncio.get_event_loop()
                    tasks = [loop.run_in_executor(executor, get_polyline, route) for route in routes]
                    return await asyncio.gather(*tasks)
            polylines = await compute_polylines(routes, locations)
            solution_data["route_polylines"] = polylines
    t_poly_end = time.perf_counter()

    # --- Metadata enriquecida ---
    total_time = int((t_poly_end - t0) * 1000)
    poly_time = int((t_poly_end - t_poly_start) * 1000)
    metadata = {
        "computation_time_ms": total_time,
        "polyline_time_ms": poly_time,
        "num_vehicles": num_vehicles,
        "num_clients": len(getattr(request, 'locations', [])) - 1 if hasattr(request, 'locations') else None,
        "strict_mode": getattr(request, 'strict_mode', False),
        "buffer_minutes": getattr(request, 'buffer_minutes', 10),
        "peak_hours": getattr(request, 'peak_hours', None),
        "peak_buffer_minutes": getattr(request, 'peak_buffer_minutes', 20)
    }

    # --- Warnings y diagnostics ---
    if polyline_warning:
        if warnings is None:
            warnings = []
        warnings.append(polyline_warning)

    return VRPAdvancedResponse(
        solution=solution_data,
        metadata=metadata,
        warnings=warnings if warnings else None,
        diagnostics=diagnostics
    )