from fastapi import APIRouter, HTTPException
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from typing import List
from schemas_skills import VRPSkillsRequest, SkillsLocation, SkillsVehicle
from route_polyline_utils import get_route_polyline_and_geojson
from schemas import VRPAdvancedResponse
import os
import requests
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()

import time
from schemas import VRPAdvancedResponse, RouteDetail

@router.post("/vrp-skills-check", response_model=VRPAdvancedResponse)
async def vrp_skills_check(request: VRPSkillsRequest):
    """
    Endpoint VRP con skills, chequeos previos y diagnósticos amigables.
    Devuelve rutas, detalles, tiempos, polylines, geojson y advertencias legibles.
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

    # --- PRECHEQUEO: Skills ---
    vehicle_skills = [set(v.provided_skills or []) for v in request.vehicles]
    location_skills = [set(l.required_skills or []) for l in request.locations]
    uncovered = []
    for idx, req_skills in enumerate(location_skills):
        if req_skills and not any(req_skills.issubset(vs) for vs in vehicle_skills):
            uncovered.append((idx, req_skills))
    if uncovered:
        msg = "Ubicaciones con habilidades no cubiertas: "
        all_vehicle_skills = set().union(*vehicle_skills)
        for idx, req_skills in uncovered:
            missing = req_skills - all_vehicle_skills
            msg += f"{request.locations[idx].name} (falta: {', '.join(missing)}), "
        msg += "Añada estas habilidades a algún vehículo."
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
        return VRPAdvancedResponse(solution=None, metadata=metadata, warnings=[msg])

    # --- PRECHEQUEO: Capacidades por ubicación y vehículo ---
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
        msg = (
            f"Las siguientes ubicaciones superan la capacidad de todos los vehículos: {', '.join(over_capacity)}. "
            "Ajuste la capacidad de los vehículos o reduzca las demandas de estas ubicaciones."
        )
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
        return VRPAdvancedResponse(solution=None, metadata=metadata, warnings=[msg])

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
    response = requests.get(url, params=params, timeout=30)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Error consultando Google Distance Matrix: {response.text}")
    data = response.json()
    if data.get("status") != "OK":
        raise HTTPException(status_code=400, detail=f"Respuesta inválida de Google Distance Matrix: {data}")
    n = len(request.locations)
    distance_matrix = []
    time_matrix = []
    for row in data["rows"]:
        distance_row = [el["distance"]["value"]/1000 if el.get("distance") else 1e6 for el in row["elements"]]
        time_row = [el["duration"]["value"]//60 if el.get("duration") else 1e6 for el in row["elements"]]
        distance_matrix.append(distance_row)
        time_matrix.append(time_row)

    # Respeta detail_level: si minimal, solo calcula matrices y retorna metadata básica
    if getattr(request, 'detail_level', 'full') == 'minimal':
        t1 = time.perf_counter()
        return VRPAdvancedResponse(
            solution={
                "distance_matrix": distance_matrix,
                "time_matrix": time_matrix,
            },
            metadata={"computation_time_ms": int((t1 - t0)*1000)},
            warnings=None
        )

    # --- SOLVER ---
    manager = pywrapcp.RoutingIndexManager(n, request.num_vehicles, request.depot)
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # Skills logic: only allow assignment if vehicle provides all required_skills for the location
    for node_idx, req_skills in enumerate(location_skills):
        if not req_skills:
            continue
        for vehicle_idx, prov_skills in enumerate(vehicle_skills):
            if not req_skills.issubset(prov_skills):
                routing.VehicleVar(manager.NodeToIndex(node_idx)).RemoveValue(vehicle_idx)
    # Time windows y service time
    time_windows = []
    service_times = []
    for idx, loc in enumerate(request.locations):
        tw = loc.time_window if loc.time_window and len(loc.time_window) == 2 else [420, 1080]
        time_windows.append(tw)
        # Servicio en el depósito es 0
        if idx == request.depot:
            service_times.append(0)
        else:
            service_times.append(getattr(loc, 'service_time', 5))
    # Callback de tiempo incluye servicio del nodo de origen
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(time_matrix[from_node][to_node] + service_times[from_node])
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        1440,
        1440,
        False,
        'Time')
    time_dimension = routing.GetDimensionOrDie('Time')
    time_dimension.SetSlackCostCoefficientForAllVehicles(100)
    for idx, window in enumerate(time_windows):
        index = manager.NodeToIndex(idx)
        time_dimension.CumulVar(index).SetRange(window[0], window[1])
    for idx, st in enumerate(service_times):
        index = manager.NodeToIndex(idx)
        time_dimension.SlackVar(index).SetValue(st)
    # --- Configuración flexible del solver ---
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # Permitir timeout configurable desde el request, si existe
    search_parameters.time_limit.seconds = getattr(request, 'solver_timeout', 30)
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)

    # --- DIAGNÓSTICO SI NO HAY SOLUCIÓN: Ventanas de tiempo mejoradas ---
    if not solution:
        out_of_time = []
        for i, loc in enumerate(request.locations):
            if i == request.depot:
                continue
            tw = loc.time_window if loc.time_window else [0, 1440]
            service_time = service_times[i]
            compatible = False
            for v in request.vehicles:
                # Verificar superposición de ventanas
                if not (v.start_time <= tw[1] and v.end_time >= tw[0]):
                    continue
                # Calcular si el servicio cabe en el horario del vehículo
                latest_arrival = min(tw[1], v.end_time - service_time)
                if latest_arrival >= max(tw[0], v.start_time):
                    compatible = True
                    break
            if not compatible:
                out_of_time.append(loc.name)
        if out_of_time:
            msg = (
                f"Las siguientes ubicaciones tienen ventanas de tiempo incompatibles con los vehículos: {', '.join(out_of_time)}. "
                "Ajuste los horarios de los vehículos o las ventanas de tiempo de estas ubicaciones."
            )
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
        return VRPAdvancedResponse(solution=None, metadata=metadata, warnings=[msg])
        msg = "No se encontró solución factible para las restricciones dadas. Revise: habilidades requeridas y ofrecidas, capacidades de los vehículos y ventanas de tiempo."
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
        return VRPAdvancedResponse(solution=None, metadata=metadata, warnings=[msg])

    # --- CONSTRUIR RESPUESTA NORMAL (optimización de tiempos de llegada ya mejorada) ---
    routes = []
    total_distance = 0
    details = []
    arrival_times = []
    warnings = []
    route_details = []
    def min_to_hhmm(m):
        h = int(m) // 60
        mm = int(m) % 60
        return f"{h:02d}:{mm:02d}"
    workday_start = 420  # 07:00
    workday_end = 1080   # 18:00
    for vehicle_id in range(request.num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        route_distance = 0
        times = []
        service_starts = []
        service_ends = []
        current_time = workday_start
        times.append(current_time)
        stop = 0
        stops_detail = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            travel_time = time_matrix[manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
            current_time += travel_time
            if not routing.IsEnd(index):
                window_start = time_windows[index][0]
                real_arrival = current_time
                wait = max(0, window_start - current_time)
                start_service = real_arrival + wait
                service_duration = service_times[index] if index < len(service_times) else 0
                end_service = start_service + service_duration
                times.append(real_arrival)
                service_starts.append(start_service)
                service_ends.append(end_service)
                current_time = end_service
                stops_detail.append({
                    "stop_index": stop+1,
                    "location_id": node,
                    "arrival_time": real_arrival,
                    "arrival_time_hhmm": min_to_hhmm(real_arrival),
                    "wait_time": wait,
                    "service_time": service_duration,
                    "service_start": start_service,
                    "service_start_hhmm": min_to_hhmm(start_service),
                    "service_end": end_service,
                    "service_end_hhmm": min_to_hhmm(end_service)
                })
            route_distance += distance_matrix[manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
            stop += 1
        route.append(manager.IndexToNode(index))
        routes.append(route)
        total_distance += route_distance
        route_details.append(RouteDetail(
            vehicle_id=vehicle_id+1,
            route=route,
            distance_km=route_distance,
            stops=stops_detail
        ))
        arrival_times.append(service_starts)
        for i, t in enumerate(service_ends):
            if t > workday_end:
                warnings.append(f"Vehículo {vehicle_id+1}: Entrega en parada {i+1} fuera de jornada laboral ({min_to_hhmm(t)} > {min_to_hhmm(workday_end)})")
    arrival_times_formatted = [[min_to_hhmm(t) for t in times] for times in arrival_times]
    # Polylines solo si include_polylines
    route_polylines = None
    route_points = None
    if getattr(request, 'include_polylines', True):
        route_polylines = await compute_polylines(routes, request.locations)
        route_points = [[(request.locations[idx].lat, request.locations[idx].lon) for idx in route] for route in routes]
    t1 = time.perf_counter()
    # Estructura la respuesta
    solution = {
        "routes": routes,
        "total_distance": round(total_distance, 2),
        "details": [rd.dict() for rd in route_details],
        "arrival_times": arrival_times,
        "arrival_times_formatted": arrival_times_formatted,
    }
    if route_polylines is not None:
        solution["route_polylines"] = route_polylines
    if route_points is not None:
        solution["route_points"] = route_points
    return VRPAdvancedResponse(
        solution=solution,
        metadata={"computation_time_ms": int((t1-t0)*1000)},
        warnings=warnings if warnings else None
    )

# --- Paralelización de cálculo de polylines ---
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
