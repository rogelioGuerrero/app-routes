import logging
from fastapi import HTTPException
from typing import List
import requests
from math import sqrt

# --- Constantes ---
class VRPConstants:
    """
    Constantes globales para la configuración del VRP.
    """
    WORKDAY_START_MIN = 420  # 07:00
    WORKDAY_END_MIN = 1080   # 18:00
    DEFAULT_SERVICE_TIME_MIN = 5
    INF_DISTANCE = 1_000_000
    MAX_TIME_MIN = 1440
    DEFAULT_SOLVER_TIMEOUT_SEC = 30
    MAX_LOCATIONS = 100   # Puedes ajustar según tu servidor
    MAX_VEHICLES = 10
    SKILL_PENALTY = 100_000
    TIME_WINDOW_PENALTY = 100_000

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Excepciones personalizadas ---
class GoogleAPIError(Exception):
    """Excepción para errores relacionados con la API de Google."""
    pass

class SolverError(Exception):
    """Excepción para errores del solver de rutas."""
    pass

# --- Utilidades para advertencias y sugerencias ---
def add_warning(warnings_list, code, message, context=None):
    """
    Agrega una advertencia estandarizada a la lista.
    """
    warnings_list.append({
        "type": "warning",
        "code": code,
        "message": message,
        "context": context or {}
    })

def add_suggestion(suggestions_list, code, message, context=None):
    """
    Agrega una sugerencia estandarizada a la lista.
    """
    suggestions_list.append({
        "type": "suggestion",
        "code": code,
        "message": message,
        "context": context or {}
    })

# Helpers para advertencias comunes

def warn_matrix_fallback(warnings_list, error_msg):
    add_warning(
        warnings_list,
        code="MATRIX_FALLBACK",
        message="Se usó matriz euclidiana por error en API externa",
        context={"error": str(error_msg)}
    )

def warn_no_viable_clients(warnings_list, excluded_clients):
    add_warning(
        warnings_list,
        code="NO_VIABLE_CLIENTS",
        message="No hay clientes viables para ruteo (excluidos por skills o capacidades)",
        context={"excluded_clients": excluded_clients}
    )

# --- Utilidades ---
def filter_viable_clients(locations, vehicles, depot_idx=0):
    """
    Filtra ubicaciones que pueden ser atendidas por algún vehículo (skills y capacidades).
    Retorna (viables, excluidos).
    """
    viables = []
    excluidos = []
    for idx, loc in enumerate(locations):
        if idx == depot_idx:
            viables.append(loc)
            continue
        skill_ok = any(
            all(skill in v.provided_skills for skill in getattr(loc, 'required_skills', []))
            for v in vehicles
        ) if getattr(loc, 'required_skills', None) else True
        capacity_ok = any(
            (getattr(loc, 'weight', 0) <= getattr(v, 'capacity_weight', 1e9)) and
            (getattr(loc, 'volume', 0) <= getattr(v, 'capacity_volume', 1e9)) and
            (getattr(loc, 'demand', 0) <= getattr(v, 'capacity_quantity', 1e9))
            for v in vehicles
        )
        if skill_ok and capacity_ok:
            viables.append(loc)
        else:
            excluidos.append(getattr(loc, 'name', getattr(loc, 'id', idx)))
    return viables, excluidos

def min_to_hhmm(m: int) -> str:
    """
    Convierte minutos a formato HH:MM.
    """
    h = int(m) // 60
    mm = int(m) % 60
    return f"{h:02d}:{mm:02d}"


def validate_request_coords(request) -> None:
    """
    Valida que todas las ubicaciones tengan coordenadas.
    """
    for loc in request.locations:
        if not hasattr(loc, 'lat') or not hasattr(loc, 'lon'):
            logger.error(f"Ubicación {loc.name} no tiene coordenadas")
            raise HTTPException(status_code=400, detail=f"Ubicación {loc.name} no tiene coordenadas")


def validate_time_windows(locations):
    """
    Valida que las ventanas de tiempo sean correctas.
    """
    for loc in locations:
        if loc.time_window:
            if len(loc.time_window) != 2 or loc.time_window[0] > loc.time_window[1] or loc.time_window[0] < 0:
                logger.error(f"Ventana de tiempo inválida en {getattr(loc, 'name', loc.id)}")
                raise HTTPException(status_code=400, detail=f"Ventana de tiempo inválida en {getattr(loc, 'name', loc.id)}")


def validate_vehicle_consistency(request):
    """
    Valida la coherencia entre num_vehicles y la lista de vehículos.
    """
    if request.num_vehicles != len(request.vehicles):
        logger.error("num_vehicles no coincide con la lista de vehículos")
        raise HTTPException(status_code=400, detail="num_vehicles no coincide con la lista de vehículos")
    if not request.vehicles or len(request.vehicles) == 0:
        logger.error("No se proporcionaron vehículos")
        raise HTTPException(status_code=400, detail="No se proporcionaron vehículos")


def validate_full_request(request):
    """
    Valida la petición completa y devuelve advertencias estructuradas y diagnóstico si algo falla.
    Retorna (is_valid, warnings, diagnostics)
    """
    warnings = []
    diagnostics = {}
    # Limites de ubicaciones y vehículos
    if len(request.locations) > VRPConstants.MAX_LOCATIONS:
        add_warning(warnings, code="TOO_MANY_LOCATIONS", message=f"Demasiadas ubicaciones (máximo {VRPConstants.MAX_LOCATIONS})")
        diagnostics["max_locations"] = VRPConstants.MAX_LOCATIONS
        return False, warnings, diagnostics
    if len(request.vehicles) > VRPConstants.MAX_VEHICLES:
        add_warning(warnings, code="TOO_MANY_VEHICLES", message=f"Demasiados vehículos (máximo {VRPConstants.MAX_VEHICLES})")
        diagnostics["max_vehicles"] = VRPConstants.MAX_VEHICLES
        return False, warnings, diagnostics
    if request.depot < 0 or request.depot >= len(request.locations):
        add_warning(warnings, code="INVALID_DEPOT_INDEX", message="Índice de depósito inválido")
        diagnostics["depot_index"] = request.depot
        return False, warnings, diagnostics
    try:
        validate_time_windows(request.locations)
        validate_vehicle_consistency(request)
    except HTTPException as e:
        add_warning(warnings, code="VALIDATION_ERROR", message=str(e.detail))
        diagnostics["validation_error"] = str(e.detail)
        return False, warnings, diagnostics
    if len(request.locations) <= 1:
        add_warning(warnings, code="ONLY_DEPOT", message="Solo se proporcionó el depósito")
        diagnostics["routes"] = []
        diagnostics["total_distance"] = 0
        return False, warnings, diagnostics
    strict_mode = getattr(request, 'strict_mode', False)
    if strict_mode:
        try:
            validate_skills_coverage(request)
        except HTTPException as e:
            add_warning(warnings, code="SKILLS_NOT_COVERED", message=str(e.detail))
            diagnostics["skills"] = str(e.detail)
            return False, warnings, diagnostics
        try:
            validate_capacity_coverage(request)
        except HTTPException as e:
            add_warning(warnings, code="CAPACITY_NOT_COVERED", message=str(e.detail))
            diagnostics["capacities"] = str(e.detail)
            return False, warnings, diagnostics
    return True, warnings, diagnostics

def validate_skills_coverage(request):
    """
    Valida que cada ubicación pueda ser cubierta por los skills de algún vehículo.
    """
    for idx, loc in enumerate(request.locations):
        req_skills = set(getattr(loc, 'required_skills', []) or [])
        if not req_skills:
            continue
        cubre_alguien = any(req_skills.issubset(set(getattr(v, 'provided_skills', []) or [])) for v in request.vehicles)
        if not cubre_alguien:
            logger.error(f"Ubicación {getattr(loc, 'name', idx)} requiere skills no cubiertos por ningún vehículo")
            raise HTTPException(status_code=400, detail=f"Ubicación {getattr(loc, 'name', idx)} requiere skills no cubiertos por ningún vehículo")

def validate_capacity_coverage(request):
    """
    Valida que cada ubicación pueda ser atendida por al menos un vehículo en cuanto a capacidades.
    """
    for idx, loc in enumerate(request.locations):
        fits = False
        for v in request.vehicles:
            fits = True
            if hasattr(loc, 'weight') and hasattr(v, 'capacity_weight'):
                fits = fits and (getattr(loc, 'weight', 0) <= getattr(v, 'capacity_weight', 0) or 0)
            if hasattr(loc, 'volume') and hasattr(v, 'capacity_volume'):
                fits = fits and (getattr(loc, 'volume', 0) <= getattr(v, 'capacity_volume', 0) or 0)
            if hasattr(loc, 'demand') and hasattr(v, 'capacity_quantity'):
                fits = fits and (getattr(loc, 'demand', 0) <= getattr(v, 'capacity_quantity', 0) or 0)
            if fits:
                break
        if not fits:
            logger.error(f"Ubicación {getattr(loc, 'name', idx)} excede capacidades de todos los vehículos")
            raise HTTPException(status_code=400, detail=f"Ubicación {getattr(loc, 'name', idx)} excede capacidades de todos los vehículos")

def validate_skills_and_capacities(request) -> List[str]:
    """
    Devuelve advertencias detalladas si hay ubicaciones con skills no cubiertos por ningún vehículo,
    o si hay capacidades excedidas, con sugerencias accionables para el usuario.
    """
    warnings = []
    vehicle_skills = [set(getattr(v, 'provided_skills', []) or []) for v in request.vehicles]
    location_skills = [set(getattr(l, 'required_skills', []) or []) for l in request.locations]
    uncovered = []
    for idx, req_skills in enumerate(location_skills):
        if req_skills and not any(req_skills.issubset(vs) for vs in vehicle_skills):
            missing = req_skills - set().union(*vehicle_skills)
            name = getattr(request.locations[idx], 'name', f"ID {idx}")
            uncovered.append((name, missing, req_skills))
    if uncovered:
        msg = "Ubicaciones con habilidades no cubiertas: "
        for name, missing, req_skills in uncovered:
            msg += f"{name} (falta: {', '.join(missing)}; requeridas: {', '.join(req_skills)}), "
        msg += "Agregue estas habilidades a algún vehículo para cubrir todas las ubicaciones."
        warnings.append(msg)
    # Advertencias de capacidades excedidas
    for idx, loc in enumerate(request.locations):
        if idx == getattr(request, 'depot', 0):
            continue
        fits = False
        reasons = []
        for v in request.vehicles:
            fits_vehicle = True
            if hasattr(loc, 'weight') and hasattr(v, 'capacity_weight'):
                if getattr(loc, 'weight', 0) > getattr(v, 'capacity_weight', 0) or 0:
                    fits_vehicle = False
                    reasons.append(f"peso {getattr(loc, 'weight', 0)} > capacidad {getattr(v, 'capacity_weight', 0) or 0}")
            if hasattr(loc, 'volume') and hasattr(v, 'capacity_volume'):
                if getattr(loc, 'volume', 0) > getattr(v, 'capacity_volume', 0) or 0:
                    fits_vehicle = False
                    reasons.append(f"volumen {getattr(loc, 'volume', 0)} > capacidad {getattr(v, 'capacity_volume', 0) or 0}")
            if hasattr(loc, 'demand') and hasattr(v, 'capacity_quantity'):
                if getattr(loc, 'demand', 0) > getattr(v, 'capacity_quantity', 0) or 0:
                    fits_vehicle = False
                    reasons.append(f"demanda {getattr(loc, 'demand', 0)} > capacidad {getattr(v, 'capacity_quantity', 0) or 0}")
            if fits_vehicle:
                fits = True
                break
        if not fits and reasons:
            name = getattr(loc, 'name', f"ID {idx}")
            msg = f"Ubicación '{name}' excede capacidades de todos los vehículos: {', '.join(reasons)}. Considere aumentar la capacidad de algún vehículo."
            warnings.append(msg)
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
        # Agregar el depósito como primer stop
        depot_node = route[0]
        depot_stop = {
            "stop_index": 0,
            "location_id": depot_node,
            "client_uuid": getattr(locations[depot_node], 'client_uuid', None),
            "arrival_time": times[0],
            "arrival_time_hhmm": min_to_hhmm(times[0]),
            "wait_time": 0,
            "service_time": service_times[depot_node],
            "service_start": service_starts[0],
            "service_start_hhmm": min_to_hhmm(service_starts[0]),
            "service_end": service_ends[0],
            "service_end_hhmm": min_to_hhmm(service_ends[0])
        }
        stops = [depot_stop]
        for stop_idx, node in enumerate(route[1:], 1):
            # Cálculo correcto del tiempo de espera: solo si el vehículo llega antes de la ventana de servicio
            ventana_inicio = time_windows[node][0]
            arrival = times[stop_idx]
            wait_time = max(0, ventana_inicio - arrival) if arrival < ventana_inicio else 0
            stop = {
                "stop_index": stop_idx,
                "location_id": node,
                "client_uuid": getattr(locations[node], 'client_uuid', None),
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
        # Tiempos efectivos del vehículo
        start_time = service_ends[0]  # Hora de salida del depósito
        end_time = service_ends[-1]   # Hora de regreso al depósito
        route_travel_time = end_time - start_time
        routes.append({
            "vehicle_id": vehicle_id,
            "vehicle_uuid": getattr(request.vehicles[vehicle_id], 'vehicle_uuid', None),
            "plate_number": getattr(request.vehicles[vehicle_id], 'plate_number', None),
            "provided_skills": getattr(request.vehicles[vehicle_id], 'provided_skills', []),
            "capacity_weight": getattr(request.vehicles[vehicle_id], 'capacity_weight', None),
            "capacity_volume": getattr(request.vehicles[vehicle_id], 'capacity_volume', None),
            "capacity_quantity": getattr(request.vehicles[vehicle_id], 'capacity_quantity', None),
            "route": route,
            "route_distance": round(route_distance, 2),
            "start_time": start_time,
            "start_time_hhmm": min_to_hhmm(start_time),
            "end_time": end_time,
            "end_time_hhmm": min_to_hhmm(end_time),
            "route_travel_time": route_travel_time,
            "route_travel_time_hhmm": f"{route_travel_time//60:02}:{route_travel_time%60:02}"
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
            "vehicle_uuid": getattr(request.vehicles[vehicle_id], 'vehicle_uuid', None),
            "plate_number": getattr(request.vehicles[vehicle_id], 'plate_number', None),
            "provided_skills": getattr(request.vehicles[vehicle_id], 'provided_skills', []),
            "capacity_weight": getattr(request.vehicles[vehicle_id], 'capacity_weight', None),
            "capacity_volume": getattr(request.vehicles[vehicle_id], 'capacity_volume', None),
            "capacity_quantity": getattr(request.vehicles[vehicle_id], 'capacity_quantity', None),
            "stops": stops
        })
    # --- Nueva estructura: client_points ---
    client_points = []
    for idx, loc in enumerate(locations):
        client_dict = {
            "client_uuid": getattr(loc, 'client_uuid', None),
            "location_id": idx,
            "lat": float(getattr(loc, 'lat', 0)),
            "lon": float(getattr(loc, 'lon', 0)),
        }
        if hasattr(loc, 'name') and getattr(loc, 'name', None):
            client_dict["name"] = getattr(loc, 'name')
        if hasattr(loc, 'required_skills') and getattr(loc, 'required_skills', None):
            client_dict["required_skills"] = getattr(loc, 'required_skills')
        if hasattr(loc, 'demand') and getattr(loc, 'demand', None):
            client_dict["demand"] = getattr(loc, 'demand')
        # Ventana horaria
        if time_windows and idx < len(time_windows):
            tw = time_windows[idx]
            client_dict["time_window"] = list(tw)
            client_dict["time_window_hhmm"] = [min_to_hhmm(tw[0]), min_to_hhmm(tw[1])]
        client_points.append(client_dict)
    return {
        "routes": routes,
        "total_distance": round(total_distance, 2),
        "route_polylines": route_polylines,
        "client_points": client_points,
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
    """
    Construye las matrices de distancia y tiempo usando la API de Google Distance Matrix.
    Incluye manejo robusto de errores y buffers para hora pico.
    """
    buffer = getattr(request, 'buffer_minutes', 10) or 10
    peak_buffer = getattr(request, 'peak_buffer_minutes', 20) or 20
    peak_hours = getattr(request, 'peak_hours', None)
    origins = [f"{getattr(loc, 'lat', 0)},{getattr(loc, 'lon', 0)}" for loc in request.locations]
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": "|".join(origins),
        "destinations": "|".join(origins),
        "key": api_key,
        "mode": getattr(request, 'mode', 'driving'),
        "units": getattr(request, 'units', 'metric'),
    }
    import requests
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.exception(f"Error consultando Google API: {str(e)}")
        raise GoogleAPIError(f"Error de red al consultar Google Distance Matrix: {str(e)}")
    data = resp.json()
    if data.get("status") != "OK":
        logger.error(f"Respuesta inválida de Google: {data}")
        raise GoogleAPIError(f"Respuesta inválida: {data}")
    n = len(request.locations)
    distance_matrix = [[el.get("distance", {}).get("value", VRPConstants.INF_DISTANCE) / 1000 for el in row["elements"]] for row in data["rows"]]
    base_time = parse_time_str("07:00")
    time_matrix = []
    for i, row in enumerate(data["rows"]):
        time_row = []
        for j, el in enumerate(row["elements"]):
            t = el.get("duration", {}).get("value", VRPConstants.INF_DISTANCE) // 60
            t_total = t + buffer if t != VRPConstants.INF_DISTANCE else VRPConstants.INF_DISTANCE
            if is_in_peak(base_time, peak_hours):
                t_total += peak_buffer
            time_row.append(t_total)
        time_matrix.append(time_row)
    return distance_matrix, time_matrix

# --- API Key segura ---
def load_api_key():
    """
    Carga y valida la API key de Google Maps desde variables de entorno.
    Lanza un error si la clave es inválida o no está configurada.
    """
    from dotenv import load_dotenv
    load_dotenv()
    import os
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or not api_key.startswith("AIza"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY inválida o no configurada")
    return api_key

# --- Dimensiones de capacidad para OR-Tools ---
def add_capacity_dimensions(routing, manager, request):
    """
    Añade dimensiones de capacidad (weight, volume, demand) al solver de OR-Tools.
    Cada dimensión es independiente y se modela como en la dimensión de tiempo.
    Si alguna dimensión no se cumple, el solver penalizará la solución.
    """
    for cap_name, attr_loc, attr_veh, dim_name in [
        ("weight", "weight", "capacity_weight", "Weight"),
        ("volume", "volume", "capacity_volume", "Volume"),
        ("demand", "demand", "capacity_quantity", "Demand")
    ]:
        if any(getattr(loc, attr_loc, 0) > 0 for loc in request.locations) and any(getattr(v, attr_veh, 0) > 0 for v in request.vehicles):
            def capacity_callback(from_index, to_index, attr=attr_loc):
                from_node = manager.IndexToNode(from_index)
                return int(getattr(request.locations[from_node], attr, 0) or 0)
            callback_idx = routing.RegisterTransitCallback(capacity_callback)
            max_cap = max(getattr(v, attr_veh, 0) or 0 for v in request.vehicles)
            routing.AddDimension(
                callback_idx,
                0,  # sin holgura
                int(max_cap),
                True,  # acumulación desde cero (debe ser bool)
                dim_name
            )
            dimension = routing.GetDimensionOrDie(dim_name)
            # Limitar la capacidad máxima solo en el depósito de cada vehículo
            for vehicle_id, v in enumerate(request.vehicles):
                cap = getattr(v, attr_veh, 0) or 0
                dimension.CumulVar(routing.Start(vehicle_id)).SetMax(int(cap))
                # Opcional: también puedes limitar el final si lo deseas
                # dimension.CumulVar(routing.End(vehicle_id)).SetMax(int(cap))

# --- Penalizaciones para skills y ventanas de tiempo ---
def apply_skills_penalty(routing, manager, request, vehicle_skills, location_skills, penalty=100_000):
    """
    Penaliza la no visita de ubicaciones si ningún vehículo cubre los skills requeridos,
    y restringe los vehículos permitidos por ubicación según skills.
    """
    for node_idx, req_skills in enumerate(location_skills):
        if not req_skills:
            continue
        allowed_vehicles = [
            v_idx for v_idx, prov_skills in enumerate(vehicle_skills)
            if req_skills.issubset(prov_skills)
        ]
        if allowed_vehicles:
            routing.SetAllowedVehiclesForIndex(allowed_vehicles, manager.NodeToIndex(node_idx))
        else:
            # Penaliza si no hay ningún vehículo que pueda cubrir los skills
            routing.AddDisjunction([manager.NodeToIndex(node_idx)], penalty=penalty)

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
