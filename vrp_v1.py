from fastapi import APIRouter, HTTPException
from schemas_skills import VRPSkillsRequest
from schemas import VRPAdvancedResponse, RouteDetail
from vrp_utils import (
    VRPConstants, min_to_hhmm, validate_request_coords, validate_skills_and_capacities,
    build_distance_and_time_matrices, GoogleAPIError, validate_full_request, build_fallback_matrix
)
from route_polyline_utils import get_route_polyline_and_geojson
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import time
from dotenv import load_dotenv

router = APIRouter()

@router.post("/vrp-v1", response_model=VRPAdvancedResponse)
async def vrp_v1(request: VRPSkillsRequest):
    t0 = time.perf_counter()
    load_dotenv()
    from vrp_utils import load_api_key
    api_key = load_api_key()

    # Validación robusta de entrada y casos límite
    valid, early_warnings, early_solution = validate_full_request(request)
    if not valid:
        return VRPAdvancedResponse(solution={}, metadata={}, warnings=early_warnings)
    # Si strict_mode está activo, el validador ya lanza error si hay ubicaciones imposibles
    # Si no, se permite penalización y solución parcial (lógica ya implementada en el solver)

    warnings = validate_skills_and_capacities(request)
    if warnings:
        return VRPAdvancedResponse(solution={}, metadata={}, warnings=warnings)

    # Construcción de matrices con fallback
    try:
        distance_matrix, time_matrix = build_distance_and_time_matrices(request, api_key)
        matrix_warning = None
    except GoogleAPIError:
        from vrp_utils import logger
        logger.warning("Usando matriz euclidiana como fallback")
        buffer = getattr(request, 'buffer_minutes', 10) or 10
        peak_hours = getattr(request, 'peak_hours', None)
        peak_buffer = getattr(request, 'peak_buffer_minutes', 20) or 20
        distance_matrix, time_matrix = build_fallback_matrix(request.locations, buffer=buffer, peak_hours=peak_hours, peak_buffer=peak_buffer)
        matrix_warning = "Se usó matriz de distancias euclidianas por error en API externa."

    # Si detail_level es minimal, solo matrices
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

    # --- Solver OR-Tools ---
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    n = len(request.locations)
    manager = pywrapcp.RoutingIndexManager(n, request.num_vehicles, request.depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Skills y penalizaciones
    vehicle_skills = [set(v.provided_skills or []) for v in request.vehicles]
    location_skills = [set(l.required_skills or []) for l in request.locations]
    from vrp_utils import apply_skills_penalty
    apply_skills_penalty(routing, manager, request, vehicle_skills, location_skills, penalty=100_000)

    # Dimensiones de capacidad (weight, volume, demand)
    from vrp_utils import add_capacity_dimensions
    add_capacity_dimensions(routing, manager, request)

    # Time windows y service time
    time_windows = []
    service_times = []
    for idx, loc in enumerate(request.locations):
        tw = loc.time_window if loc.time_window and len(loc.time_window) == 2 else [VRPConstants.WORKDAY_START_MIN, VRPConstants.WORKDAY_END_MIN]
        time_windows.append(tw)
        service_times.append(getattr(loc, 'service_time', VRPConstants.DEFAULT_SERVICE_TIME_MIN) if idx != request.depot else 0)

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(time_matrix[from_node][to_node] + service_times[from_node])
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        VRPConstants.MAX_TIME_MIN,
        VRPConstants.MAX_TIME_MIN,
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

    # (Opcional) penalización por saltar ventanas de tiempo: hook para extender
    # from vrp_utils import apply_time_window_penalty
    # apply_time_window_penalty(routing, manager, time_dimension, penalty=100_000)

    # Configuración flexible del solver
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = getattr(request, 'solver_timeout', VRPConstants.DEFAULT_SOLVER_TIMEOUT_SEC)
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)

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

    # --- Warning enriquecido: ubicaciones no atendidas (incluso si no hay solución) ---
    not_assigned = []
    for idx, loc in enumerate(request.locations):
        if idx == request.depot:
            continue
        name = getattr(loc, 'name', f"ID {idx}")
        why = []
        if hasattr(loc, 'required_skills') and loc.required_skills:
            why.append(f"skills requeridos: {loc.required_skills}")
        if hasattr(loc, 'weight') and hasattr(request.vehicles[0], 'capacity_weight'):
            why.append(f"peso: {getattr(loc, 'weight', 0)}")
        if hasattr(loc, 'volume') and hasattr(request.vehicles[0], 'capacity_volume'):
            why.append(f"volumen: {getattr(loc, 'volume', 0)}")
        if hasattr(loc, 'demand') and hasattr(request.vehicles[0], 'capacity_quantity'):
            why.append(f"demand: {getattr(loc, 'demand', 0)}")
        if hasattr(loc, 'time_window'):
            why.append(f"ventana: {getattr(loc, 'time_window', [])}")
        not_assigned.append(f"{name} (" + ", ".join(why) + ")")
    warnings_out = [matrix_warning] if matrix_warning else []
    if not solution:
        if not_assigned:
            warnings_out.append(f"No se pudo asignar: {', '.join(not_assigned)}")
        else:
            warnings_out.append("No se encontró solución factible para las restricciones dadas. Revise: habilidades requeridas y ofrecidas, capacidades de los vehículos y ventanas de tiempo.")
        return VRPAdvancedResponse(solution={}, metadata=metadata, warnings=warnings_out)

    # --- Procesamiento de resultados ---
    routes = []
    total_distance = 0
    assigned_nodes = set()
    for vehicle_id in range(request.num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        route_distance = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            assigned_nodes.add(node_index)
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

    # --- Warning enriquecido: ubicaciones no atendidas ---
    not_assigned = []
    for idx, loc in enumerate(request.locations):
        if idx == request.depot:
            continue
        if idx not in assigned_nodes:
            name = getattr(loc, 'name', f"ID {idx}")
            why = []
            if hasattr(loc, 'required_skills') and loc.required_skills:
                why.append(f"skills requeridos: {loc.required_skills}")
            if hasattr(loc, 'weight') and hasattr(request.vehicles[0], 'capacity_weight'):
                why.append(f"peso: {getattr(loc, 'weight', 0)}")
            if hasattr(loc, 'volume') and hasattr(request.vehicles[0], 'capacity_volume'):
                why.append(f"volumen: {getattr(loc, 'volume', 0)}")
            if hasattr(loc, 'demand') and hasattr(request.vehicles[0], 'capacity_quantity'):
                why.append(f"demand: {getattr(loc, 'demand', 0)}")
            if hasattr(loc, 'time_window'):
                why.append(f"ventana: {getattr(loc, 'time_window', [])}")
            not_assigned.append(f"{name} (" + ", ".join(why) + ")")
    warnings_out = [matrix_warning] if matrix_warning else []
    if not_assigned:
        warnings_out.append(f"No se pudo asignar: {', '.join(not_assigned)}")

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

    route_details = []
    arrival_times = []
    for vehicle_id in range(request.num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        route_distance = 0
        times = []
        service_starts = []
        service_ends = []
        current_time = VRPConstants.WORKDAY_START_MIN
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
        return route, route_distance, stops_detail, service_starts, service_ends

    for vehicle_id in range(request.num_vehicles):
        route, route_distance, stops_detail, service_starts, service_ends = build_route(vehicle_id)
        routes.append(route)
        total_distance += route_distance
        route_details.append(RouteDetail(
            vehicle_id=vehicle_id+1,
            route=route,
            distance_km=route_distance,
            stops=stops_detail
        ))
        arrival_times.append(service_starts)

    arrival_times_formatted = [[min_to_hhmm(t) for t in times] for times in arrival_times]
    # Polylines solo si include_polylines
    route_polylines = None
    route_points = None
    if getattr(request, 'include_polylines', True):
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
        route_polylines = await compute_polylines(routes, request.locations)
        route_points = [[(request.locations[idx].lat, request.locations[idx].lon) for idx in route] for route in routes]
    t1 = time.perf_counter()
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
        warnings=None
    )
