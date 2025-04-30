from fastapi import APIRouter, HTTPException
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from typing import List
from schemas_skills import VRPSkillsRequest, SkillsLocation, SkillsVehicle
from route_polyline_utils import get_route_polyline_and_geojson
from schemas import VRPAdvancedResponse
from vrp_utils import min_to_hhmm
import os
import requests
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

router = APIRouter()

@router.post("/vrp-v1", response_model=VRPAdvancedResponse)
async def vrp_v1(request: VRPSkillsRequest):
    """
    Nuevo endpoint VRP vitaminado: robusto, validaciones claras, base para mejoras.
    Mejora sobre vrp-skills-check. Aquí se integrarán capacidades avanzadas y fallback.
    """
    t0 = time.perf_counter()
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="GOOGLE_API_KEY no encontrada en .env")

    warnings = []

    # --- VALIDACIÓN: Coordenadas ---
    for loc in request.locations:
        if not hasattr(loc, 'lat') or not hasattr(loc, 'lon'):
            raise HTTPException(status_code=400, detail=f"Ubicación {loc.name} no tiene coordenadas")
        # Validación de ventana de tiempo
        if hasattr(loc, 'time_window') and loc.time_window:
            tw = loc.time_window
            if len(tw) == 2 and tw[1] <= tw[0]:
                raise HTTPException(status_code=400, detail=f"Ubicación {getattr(loc, 'name', loc.id)} tiene una ventana de tiempo inválida: fin <= inicio")

    # --- FILTRO PREVIO DE CLIENTES INVIABLES (skills y capacidades) ---
    vehicle_skills = [set(getattr(v, 'provided_skills', []) or []) for v in request.vehicles]
    viable_locations = []
    excluded_clients = []
    for idx, loc in enumerate(request.locations):
        req_skills = set(getattr(loc, 'required_skills', []) or [])
        skills_ok = not req_skills or any(req_skills.issubset(vs) for vs in vehicle_skills)
        capacity_ok = any(
            (getattr(loc, 'weight', 0) <= (getattr(v, 'capacity_weight', 0) or 0)) and
            (getattr(loc, 'volume', 0) <= (getattr(v, 'capacity_volume', 0) or 0)) and
            (getattr(loc, 'demand', 0) <= (getattr(v, 'capacity_quantity', 0) or 0))
            for v in request.vehicles
        )
        if idx == getattr(request, 'depot', 0):
            viable_locations.append(loc)
        elif not skills_ok:
            excluded_clients.append({"name": getattr(loc, 'name', f"ID {idx}"), "reason": "skills no cubiertos"})
        elif not capacity_ok:
            excluded_clients.append({"name": getattr(loc, 'name', f"ID {idx}"), "reason": "excede capacidades"})
        else:
            viable_locations.append(loc)
    if len(viable_locations) <= 1:
        msg = "No hay clientes viables para ruteo (todos excluidos por skills o capacidades)."
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
        return VRPAdvancedResponse(solution={}, metadata=metadata, warnings=[msg], excluded_clients=excluded_clients)
    # Reemplaza las ubicaciones en el request por solo las viables
    request.locations = viable_locations

    # --- MATRIZ DE DISTANCIAS ---
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
        if response.status_code != 200:
            raise Exception(f"Error consultando Google Distance Matrix: {response.text}")
        data = response.json()
        if data.get("status") != "OK":
            raise Exception(f"Respuesta inválida de Google Distance Matrix: {data}")
        n = len(request.locations)
        distance_matrix = []
        time_matrix = []
        for row in data["rows"]:
            distance_row = [el["distance"]["value"]/1000 if el.get("distance") else 1e6 for el in row["elements"]]
            time_row = [el["duration"]["value"]//60 if el.get("duration") else 1e6 for el in row["elements"]]
            distance_matrix.append(distance_row)
            time_matrix.append(time_row)
        matrix_warning = None
    except Exception as e:
        # --- Fallback euclidiano (mejora) ---
        matrix_warning = f"Se usó matriz euclidiana por error en API externa: {str(e)}"
        n = len(request.locations)
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
                    lat1, lon1 = request.locations[i].lat, request.locations[i].lon
                    lat2, lon2 = request.locations[j].lat, request.locations[j].lon
                    dist = ((lat1-lat2)**2 + (lon1-lon2)**2)**0.5 * 111  # Aprox km
                    d_row.append(dist)
                    t_row.append(int(dist/40*60))  # 40km/h promedio
            distance_matrix.append(d_row)
            time_matrix.append(t_row)
        # --- Ajuste de congestión por horas pico ---
        peak_hours = getattr(request, 'peak_hours', None)  # Ejemplo: [450, 570] para 07:30 a 09:30
        peak_multiplier = getattr(request, 'peak_multiplier', 1.3)  # 30% más lento
        if peak_hours:
            for i in range(n):
                for j in range(n):
                    # Solución sencilla: aplica a toda la matriz
                    time_matrix[i][j] = int(time_matrix[i][j] * peak_multiplier)

    # --- RESPUESTA SOLO MATRICES SI detail_level=minimal ---
    if getattr(request, 'detail_level', 'full') == 'minimal':
        t1 = time.perf_counter()
        return VRPAdvancedResponse(
            solution={
                "distance_matrix": distance_matrix,
                "time_matrix": time_matrix,
            },
            metadata={"computation_time_ms": int((t1-t0)*1000)},
            warnings=[matrix_warning] if matrix_warning else None
        )

    # --- SOLVER VRP ---
    manager = pywrapcp.RoutingIndexManager(n, request.num_vehicles, request.depot)
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # --- SKILLS: RESTRICCIÓN ESTRICTA O PENALIZACIÓN (configurable) ---
    # Nuevo parámetro: skills_penalty_mode = 'strict' (default) o 'penalty'
    # --- Definir skills de vehículos y ubicaciones ---
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

    # --- DIMENSIONES DE CAPACIDAD AVANZADAS (mejora) ---
    from vrp_utils import add_capacity_dimensions
    add_capacity_dimensions(routing, manager, request)

    # --- TIME WINDOWS Y SERVICE TIME ---
    time_windows = []
    service_times = []
    for idx, loc in enumerate(request.locations):
        tw = loc.time_window if loc.time_window and len(loc.time_window) == 2 else [420, 1080]
        time_windows.append(tw)
        service_times.append(getattr(loc, 'service_time', 5) if idx != request.depot else 0)
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
    # --- Ajuste dinámico de la ventana del depósito (mejora real) ---
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
        msg = "No se pudo encontrar una solución factible. "
        if out_of_time:
            msg += f"Ubicaciones fuera de ventana: {', '.join(out_of_time)}. "
        msg += "Verifica ventanas de tiempo, capacidades y skills."
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
        return VRPAdvancedResponse(solution={}, metadata=metadata, warnings=[msg, matrix_warning] if matrix_warning else [msg])

    # --- PROCESAMIENTO DE RESULTADOS ---
    routes = []
    total_distance = 0
    suggestions = []
    for vehicle_id in range(request.num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        route_distance = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        route.append(manager.IndexToNode(index))
        routes.append({
            "vehicle_id": vehicle_id,
            "route": route,
            "route_distance": route_distance
        })
        total_distance += route_distance
        # --- SUGERENCIAS DE HORA DE SALIDA SI HAY MUCHA ESPERA EN EL PRIMER CLIENTE ---
        if len(route) > 2:
            # Primer cliente real (no depósito)
            depot_idx = route[0]
            first_client_idx = route[1]
            travel_time = time_matrix[depot_idx][first_client_idx]
            cliente = request.locations[first_client_idx]
            ventana_inicio = cliente.time_window[0] if hasattr(cliente, 'time_window') and cliente.time_window else 0
            hora_salida_sugerida = ventana_inicio - travel_time
            # Usa el horario laboral si es más restrictivo
            veh = request.vehicles[vehicle_id]
            hora_salida_sugerida = max(hora_salida_sugerida, getattr(veh, 'start_time', 420))
            # Calcula el wait_time real en la solución
            # (El detalle de paradas se arma después, así que lo calculamos aquí)
            actual_departure = getattr(veh, 'start_time', 420)
            actual_arrival = actual_departure + travel_time
            wait_time = max(0, ventana_inicio - actual_arrival)
            if wait_time > 30:  # Si hay más de 30 min de espera, sugiere
                msg = (
                    f"Sugerimos salir del depósito a las {min_to_hhmm(hora_salida_sugerida)} para evitar esperar {wait_time} minutos en el primer cliente ({getattr(cliente, 'name', 'cliente')})."
                )
                suggestions.append(msg)

    # --- POLYLINES Y DETALLE DE PARADAS ---
    route_polylines = []
    route_points = []
    details = []
    for idx_r, r in enumerate(routes):
        latlons = [(request.locations[idx].lat, request.locations[idx].lon) for idx in r["route"]]
        try:
            polyline, _ = get_route_polyline_and_geojson(latlons, api_key=api_key)
        except Exception:
            polyline = None
        route_polylines.append(polyline)
        route_points.append(latlons)
        # Detalle de paradas
        stops_detail = []
        veh = request.vehicles[r["vehicle_id"]]
        current_time = getattr(veh, 'start_time', 420)
        for i in range(1, len(r["route"])):
            prev_idx = r["route"][i-1]
            idx = r["route"][i]
            travel_time = time_matrix[prev_idx][idx]
            current_time += travel_time
            window_start = time_windows[idx][0] if idx < len(time_windows) else 420
            real_arrival = current_time
            wait = max(0, window_start - current_time)
            start_service = real_arrival + wait
            service_duration = service_times[idx] if idx < len(service_times) else 0
            end_service = start_service + service_duration
            # SUGERENCIA: Si es el primer cliente real, revisa espera real y agrega sugerencia si corresponde
            if i == 1 and idx != request.depot:
                cliente = request.locations[idx]
                if wait > 30:
                    hora_salida_sugerida = window_start - travel_time
                    hora_salida_sugerida = max(hora_salida_sugerida, getattr(veh, 'start_time', 420))
                    msg = (
                        f"Sugerimos salir del depósito a las {min_to_hhmm(hora_salida_sugerida)} para evitar esperar {wait} minutos en el primer cliente ({getattr(cliente, 'name', 'cliente')})."
                    )
                    suggestions.append(msg)
            stops_detail.append({
                "stop_index": i,
                "location_id": idx,
                "arrival_time": real_arrival,
                "arrival_time_hhmm": min_to_hhmm(real_arrival),
                "wait_time": wait,
                "service_time": service_duration,
                "service_start": start_service,
                "service_start_hhmm": min_to_hhmm(start_service),
                "service_end": end_service,
                "service_end_hhmm": min_to_hhmm(end_service)
            })
            current_time = end_service
        details.append({
            "vehicle_id": r["vehicle_id"],
            "stops": stops_detail
        })

    # --- Unifica warnings y elimina duplicados ---
    # --- SUGERENCIAS (por ruta) y ADVERTENCIAS (por solver) ---
    unique_suggestions = list(dict.fromkeys(suggestions))
    solution_obj = {
        "routes": routes,
        "total_distance": total_distance,
        "route_polylines": route_polylines,
        "route_points": route_points,
        "details": details,
        "suggestions": unique_suggestions
    }
    # Advertencias del solver (ej: problemas de matriz)
    warnings_root = []
    if 'matrix_warning' in locals() and matrix_warning:
        warnings_root.append(matrix_warning)

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
    if warnings_root:
        return {
            "solution": solution_obj,
            "metadata": metadata,
            "warnings": warnings_root
        }
    else:
        return {
            "solution": solution_obj,
            "metadata": metadata
        }
