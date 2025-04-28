import logging
from fastapi import HTTPException
from typing import List
import requests
from math import sqrt

# --- Constantes ---
class VRPConstants:
    WORKDAY_START_MIN = 420  # 07:00
    WORKDAY_END_MIN = 1080   # 18:00
    DEFAULT_SERVICE_TIME_MIN = 5
    INF_DISTANCE = 1_000_000
    MAX_TIME_MIN = 1440
    DEFAULT_SOLVER_TIMEOUT_SEC = 30
    MAX_LOCATIONS = 100   # Puedes ajustar según tu servidor
    MAX_VEHICLES = 10

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Excepciones personalizadas ---
class GoogleAPIError(Exception):
    pass

class SolverError(Exception):
    pass

# --- Utilidades ---
def min_to_hhmm(m: int) -> str:
    h = int(m) // 60
    mm = int(m) % 60
    return f"{h:02d}:{mm:02d}"


def validate_request_coords(request) -> None:
    for loc in request.locations:
        if not hasattr(loc, 'lat') or not hasattr(loc, 'lon'):
            raise HTTPException(status_code=400, detail=f"Ubicación {loc.name} no tiene coordenadas")


def validate_full_request(request):
    # Límite de ubicaciones y vehículos
    if len(request.locations) > VRPConstants.MAX_LOCATIONS:
        raise HTTPException(status_code=400, detail=f"Demasiadas ubicaciones (máximo {VRPConstants.MAX_LOCATIONS})")
    if len(request.vehicles) > VRPConstants.MAX_VEHICLES:
        raise HTTPException(status_code=400, detail=f"Demasiados vehículos (máximo {VRPConstants.MAX_VEHICLES})")
    # Depot válido
    if request.depot < 0 or request.depot >= len(request.locations):
        raise HTTPException(status_code=400, detail="Índice de depósito inválido")
    # Ventanas de tiempo válidas
    for loc in request.locations:
        if loc.time_window:
            if len(loc.time_window) != 2 or loc.time_window[0] > loc.time_window[1] or loc.time_window[0] < 0:
                raise HTTPException(status_code=400, detail=f"Ventana de tiempo inválida en {getattr(loc, 'name', loc.id)}")
    # Coherencia vehículos
    if request.num_vehicles != len(request.vehicles):
        raise HTTPException(status_code=400, detail="num_vehicles no coincide con la lista de vehículos")
    # Al menos un vehículo
    if not request.vehicles or len(request.vehicles) == 0:
        raise HTTPException(status_code=400, detail="No se proporcionaron vehículos")
    # Caso solo depósito
    if len(request.locations) <= 1:
        return False, ["Solo se proporcionó el depósito"], {"routes": [], "total_distance": 0}
    # Validación estricta de skills/capacidades si strict_mode está activo
    strict_mode = getattr(request, 'strict_mode', False)
    if strict_mode:
        # Skills: cada ubicación debe ser cubrible por al menos un vehículo
        for idx, loc in enumerate(request.locations):
            req_skills = set(getattr(loc, 'required_skills', []) or [])
            if not req_skills:
                continue
            cubre_alguien = any(req_skills.issubset(set(getattr(v, 'provided_skills', []) or [])) for v in request.vehicles)
            if not cubre_alguien:
                raise HTTPException(status_code=400, detail=f"Ubicación {getattr(loc, 'name', idx)} requiere skills no cubiertos por ningún vehículo")
        # Capacidades: cada ubicación debe caber en algún vehículo
        for idx, loc in enumerate(request.locations):
            fits = False
            for v in request.vehicles:
                fits = True
                if hasattr(loc, 'weight') and hasattr(v, 'capacity_weight'):
                    fits = fits and (getattr(loc, 'weight', 0) <= getattr(v, 'capacity_weight', 0))
                if hasattr(loc, 'volume') and hasattr(v, 'capacity_volume'):
                    fits = fits and (getattr(loc, 'volume', 0) <= getattr(v, 'capacity_volume', 0))
                if hasattr(loc, 'demand') and hasattr(v, 'capacity_quantity'):
                    fits = fits and (getattr(loc, 'demand', 0) <= getattr(v, 'capacity_quantity', 0))
                if fits:
                    break
            if not fits:
                raise HTTPException(status_code=400, detail=f"Ubicación {getattr(loc, 'name', idx)} excede capacidades de todos los vehículos")
    return True, [], None

def validate_skills_and_capacities(request) -> List[str]:
    warnings = []
    vehicle_skills = [set(v.provided_skills or []) for v in request.vehicles]
    location_skills = [set(l.required_skills or []) for l in request.locations]
    uncovered = []
    for idx, req_skills in enumerate(location_skills):
        if req_skills and not any(req_skills.issubset(vs) for vs in vehicle_skills):
            uncovered.append((idx, req_skills))
    if uncovered:
        all_vehicle_skills = set().union(*vehicle_skills)
        for idx, req_skills in uncovered:
            missing = req_skills - all_vehicle_skills
            warnings.append(f"{request.locations[idx].name} (falta: {', '.join(missing)})")
    # Capacidad
    over_capacity = []
    for loc in request.locations:
        can_serve = any(
            (getattr(loc, 'weight', 0) <= v.capacity_weight) and
            (getattr(loc, 'volume', 0) <= v.capacity_volume) and
            (getattr(loc, 'demand', 0) <= v.capacity_quantity)
            for v in request.vehicles
        )
        if not can_serve:
            over_capacity.append(loc.name)
    if over_capacity:
        warnings.append(f"Ubicaciones sobre capacidad: {', '.join(over_capacity)}")
    return warnings


from datetime import datetime, timedelta

def build_vrp_solution(solution, routing, manager, request, time_dimension, time_windows, service_times, distance_matrix, time_matrix):
    """
    Devuelve la estructura avanzada con rutas, polylines, puntos y detalles por parada según el formato solicitado por el usuario.
    """
    from route_polyline_utils import get_route_polyline_and_geojson
    if solution is None:
        return {
            "routes": [],
            "total_distance": 0,
            "route_polylines": [],
            "route_points": [],
            "details": []
        }
    num_vehicles = request.num_vehicles
    depot = request.depot
    n = len(request.locations)
    routes = []
    route_polylines = []
    route_points = []
    details = []
    total_distance = 0
    locations = request.locations
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        times = []
        espera = []
        service_starts = []
        service_ends = []
        route_distance = 0
        stops = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            # Tiempos
            arrival = solution.Value(time_dimension.CumulVar(index))
            times.append(arrival)
            # Espera
            wait = max(0, arrival - time_windows[node][0])
            espera.append(wait)
            # Servicio
            service_start = arrival
            service_end = arrival + service_times[node]
            service_starts.append(service_start)
            service_ends.append(service_end)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(index):
                route_distance += distance_matrix[node][manager.IndexToNode(index)]
        # Añadir regreso al depósito
        node = manager.IndexToNode(index)
        route.append(node)
        arrival = solution.Value(time_dimension.CumulVar(index))
        times.append(arrival)
        espera.append(0)
        service_starts.append(arrival)
        service_ends.append(arrival)
        # Construir stops detallados
        for stop_idx, node in enumerate(route[1:], 1):
            # Cálculo correcto del tiempo de espera: solo si el vehículo llega antes de la ventana de servicio
            ventana_inicio = time_windows[node][0]
            arrival = times[stop_idx]
            wait_time = max(0, ventana_inicio - arrival) if arrival < ventana_inicio else 0
            stop = {
                "stop_index": stop_idx,
                "location_id": node,
                "arrival_time": arrival,
                "arrival_time_hhmm": min_to_hhmm(arrival),
                "wait_time": wait_time,
                "service_time": service_times[node],
                "service_start": service_starts[stop_idx],
                "service_start_hhmm": min_to_hhmm(service_starts[stop_idx]),
                "service_end": service_ends[stop_idx],
                "service_end_hhmm": min_to_hhmm(service_ends[stop_idx])
            }
            stops.append(stop)
        # Rutas y detalles
        routes.append({
            "vehicle_id": vehicle_id,
            "route": route,
            "route_distance": round(route_distance, 2)
        })
        total_distance += route_distance
        # Polyline y puntos
        latlons = [(locations[nodo].lat, locations[nodo].lon) for nodo in route]
        try:
            polyline, _ = get_route_polyline_and_geojson(latlons)
        except Exception:
            polyline = None
        route_polylines.append(polyline)
        route_points.append([[float(locations[nodo].lat), float(locations[nodo].lon)] for nodo in route])
        details.append({
            "vehicle_id": vehicle_id,
            "stops": stops
        })
    return {
        "routes": routes,
        "total_distance": round(total_distance, 2),
        "route_polylines": route_polylines,
        "route_points": route_points,
        "details": details
    }


def parse_time_str(s):
    # '07:00' -> minutos desde medianoche
    h, m = map(int, s.split(':'))
    return h*60 + m

def is_in_peak(hour_minute, peak_hours):
    # hour_minute: minutos desde medianoche
    for period in (peak_hours or []):
        if isinstance(period, str):
            if '-' in period:
                start, end = period.split('-')
            else:
                continue  # formato no válido
        else:
            start, end = period
        start_min = parse_time_str(start)
        end_min = parse_time_str(end)
        if start_min <= hour_minute < end_min:
            return True
    return False

def build_distance_and_time_matrices(request, api_key: str):
    buffer = getattr(request, 'buffer_minutes', 10) or 10
    peak_buffer = getattr(request, 'peak_buffer_minutes', 20) or 20
    peak_hours = getattr(request, 'peak_hours', None)
    origins = [f"{loc.lat},{loc.lon}" for loc in request.locations]
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": "|".join(origins),
        "destinations": "|".join(origins),
        "key": api_key,
        "mode": getattr(request, 'mode', 'driving'),
        "units": getattr(request, 'units', 'metric'),
    }
    import requests
    resp = requests.get(url, params=params)
    data = resp.json()
    if data.get("status") != "OK":
        raise GoogleAPIError(f"Respuesta inválida: {data}")
    n = len(request.locations)
    distance_matrix = [[el["distance"]["value"]/1000 if el.get("distance") else VRPConstants.INF_DISTANCE for el in row["elements"]] for row in data["rows"]]
    # Estimación simple: todos los trayectos inician a las 07:00 (puedes mejorar esto luego)
    base_time = parse_time_str("07:00")
    time_matrix = []
    for i, row in enumerate(data["rows"]):
        time_row = []
        for j, el in enumerate(row["elements"]):
            t = el["duration"]["value"]//60 if el.get("duration") else VRPConstants.INF_DISTANCE
            # Aplica buffer estándar
            t_total = t + buffer
            # Si inicia en hora pico, suma el extra
            if is_in_peak(base_time, peak_hours):
                t_total += peak_buffer
            time_row.append(t_total)
        time_matrix.append(time_row)
    return distance_matrix, time_matrix

    origins = [f"{loc.lat},{loc.lon}" for loc in request.locations]
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": "|".join(origins),
        "destinations": "|".join(origins),
        "mode": request.mode or "driving",
        "units": request.units or "metric",
        "key": api_key
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Error consultando Google API: {str(e)}")
        raise GoogleAPIError(f"Error de red al consultar Google Distance Matrix: {str(e)}")
    data = response.json()
    if data.get("status") != "OK":
        logger.error(f"Respuesta inválida de Google: {data}")
        raise GoogleAPIError(f"Respuesta inválida: {data}")
    n = len(request.locations)
    distance_matrix = [[el["distance"]["value"]/1000 if el.get("distance") else VRPConstants.INF_DISTANCE for el in row["elements"]] for row in data["rows"]]
    # Suma el buffer a cada trayecto
    time_matrix = [[(el["duration"]["value"]//60 + buffer) if el.get("duration") else VRPConstants.INF_DISTANCE for el in row["elements"]] for row in data["rows"]]
    return distance_matrix, time_matrix

# --- API Key segura ---
def load_api_key():
    import os
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or not api_key.startswith("AIza"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY inválida o no configurada")
    return api_key

# --- Dimensiones de capacidad para OR-Tools ---
def add_capacity_dimensions(routing, manager, request):
    """
    Añade dimensiones de capacidad (weight, volume, demand) al solver.
    Solo se añade la dimensión si hay datos relevantes (>0) en las ubicaciones/vehículos.
    """
    from ortools.constraint_solver import routing_enums_pb2
    capacities = [
        ("weight", "capacity_weight"),
        ("volume", "capacity_volume"),
        ("demand", "capacity_quantity")
    ]
    for cap_name, veh_attr in capacities:
        # ¿Hay alguna demanda/capacidad relevante?
        has_data = any(getattr(loc, cap_name, 0) for loc in request.locations) and any(getattr(v, veh_attr, 0) for v in request.vehicles)
        if not has_data:
            continue
        def make_callback(attr):
            def callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                return int(getattr(request.locations[from_node], attr, 0) or 0)
            return callback
        callback = make_callback(cap_name)
        cb_index = routing.RegisterTransitCallback(callback)
        max_cap = max(int(getattr(v, veh_attr, 0) or 0) for v in request.vehicles)
        routing.AddDimension(
            cb_index,
            0,  # No slack
            int(max_cap),
            True,  # start cumul to zero
            cap_name.capitalize()
        )
        dim = routing.GetDimensionOrDie(cap_name.capitalize())
        # Limita la capacidad máxima por vehículo
        for vehicle_id, v in enumerate(request.vehicles):
            dim.CumulVar(routing.Start(vehicle_id)).SetMax(int(getattr(v, veh_attr, 0) or 0))

# --- Penalizaciones para skills y ventanas de tiempo ---
def apply_skills_penalty(routing, manager, request, vehicle_skills, location_skills, penalty=100_000):
    """
    Permite penalizar la no visita de ubicaciones si ningún vehículo cubre los skills requeridos.
    """
    for node_idx, req_skills in enumerate(location_skills):
        if not req_skills:
            continue
        allowed_vehicles = [v_idx for v_idx, prov_skills in enumerate(vehicle_skills) if req_skills.issubset(prov_skills)]
        if allowed_vehicles:
            routing.SetAllowedVehiclesForIndex(allowed_vehicles, manager.NodeToIndex(node_idx))
        else:
            # Penaliza si no se visita (disyunción)
            routing.AddDisjunction([manager.NodeToIndex(node_idx)], penalty)

# Penalización de ventanas de tiempo (opcional, aquí solo hook)
def apply_time_window_penalty(routing, manager, time_dimension, penalty=100_000):
    # Puedes usar AddDisjunction para permitir saltar nodos fuera de ventana con penalización
    pass  # Hook para extender si quieres penalizar violaciones de ventana

# --- Fallback: matriz euclidiana ---
def euclidean_distance(loc1, loc2):
    return sqrt((loc1.lat - loc2.lat)**2 + (loc1.lon - loc2.lon)**2) * 111  # Aproximación km

def build_fallback_matrix(locations, buffer=10, peak_hours=None, peak_buffer=20):
    n = len(locations)
    distance_matrix = [[0.0] * n for _ in range(n)]
    time_matrix = [[0.0] * n for _ in range(n)]
    base_time = parse_time_str("07:00")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = euclidean_distance(locations[i], locations[j])
            distance_matrix[i][j] = dist
            t = dist / 40 * 60 + buffer
            if is_in_peak(base_time, peak_hours):
                t += peak_buffer
            time_matrix[i][j] = t
    return distance_matrix, time_matrix
