from fastapi import APIRouter, HTTPException
from schemas_skills import VRPSkillsRequest
from schemas import VRPAdvancedResponse, RouteDetail
from vrp_utils import (
    VRPConstants, min_to_hhmm, validate_request_coords, validate_skills_and_capacities,
    build_distance_and_time_matrices, GoogleAPIError, validate_full_request, build_fallback_matrix,
    build_vrp_solution
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
        return VRPAdvancedResponse(
            solution={},
            metadata={},
            warnings=early_warnings)
    # Si strict_mode está activo, el validador ya lanza error si hay ubicaciones imposibles
    # Si no, se permite penalización y solución parcial (lógica ya implementada en el solver)

    warnings = validate_skills_and_capacities(request)
    if warnings:
        return VRPAdvancedResponse(
            solution={},
            metadata={
                "sugerencia_salida_deposito": sugerencia_salida,
                "mensaje_sugerencia": mensaje_sugerencia
            },
            warnings=warnings)

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

    # --- Ciclo externo para minimizar la espera en el primer cliente reconstruyendo el modelo ---
    max_wait_minutes = 10
    max_iters = 5
    iter_count = 0
    best_solution = None
    best_wait = None
    best_earliest_start = None
    earliest_start = None

    while iter_count < max_iters:
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
            VRPConstants.MAX_TIME_MIN,  # slack máximo permitido
            VRPConstants.MAX_TIME_MIN,  # tiempo máximo por ruta
            False,
            'Time')
        time_dimension = routing.GetDimensionOrDie('Time')
        penalizacion_espera = 1000
        time_dimension.SetSlackCostCoefficientForAllVehicles(penalizacion_espera)
        # Ajuste dinámico de la ventana del depósito para evitar salidas demasiado tempranas
        for vehicle_id, vehicle in enumerate(request.vehicles):
            depot_index = routing.Start(vehicle_id)
            start_time_orig = getattr(vehicle, 'start_time', VRPConstants.WORKDAY_START_MIN)
            end_time = getattr(vehicle, 'end_time', VRPConstants.WORKDAY_END_MIN)
            earliest_client_start = min([tw[0] for i, tw in enumerate(time_windows) if i != request.depot])
            min_travel_time = min([time_matrix[request.depot][i] for i in range(n) if i != request.depot])
            # En la segunda iteración en adelante, earliest_start se ajusta
            if earliest_start is not None:
                start_time = earliest_start
            else:
                start_time = max(start_time_orig, earliest_client_start - min_travel_time)
            time_dimension.CumulVar(depot_index).SetRange(start_time, end_time)
        for idx, window in enumerate(time_windows):
            if idx == request.depot:
                continue
            index = manager.NodeToIndex(idx)
            time_dimension.CumulVar(index).SetRange(window[0], window[1])
        for idx, st in enumerate(service_times):
            index = manager.NodeToIndex(idx)
            time_dimension.SlackVar(index).SetValue(st)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = getattr(request, 'solver_timeout', VRPConstants.DEFAULT_SOLVER_TIMEOUT_SEC)
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        solution = routing.SolveWithParameters(search_parameters)
        if solution is None:
            break
        # Extrae la espera en el primer cliente de la ruta del primer vehículo
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(index)
            index = solution.Value(routing.NextVar(index))
        if len(route) > 1:
            first_customer_idx = manager.IndexToNode(route[1])
            first_customer_index = route[1]
            arrival = solution.Value(time_dimension.CumulVar(first_customer_index))
            wait = arrival - time_windows[first_customer_idx][0]
            if wait < 0:
                wait = 0
        else:
            wait = 0
        if best_wait is None or wait < best_wait:
            best_wait = wait
            best_solution = solution
            best_earliest_start = start_time
        if wait <= max_wait_minutes:
            break
        # Ajusta earliest_start para la siguiente iteración
        depot_index = routing.Start(0)
        current_start = solution.Value(time_dimension.CumulVar(depot_index))
        earliest_start = current_start + (wait - max_wait_minutes)
        iter_count += 1
    solution = best_solution

    # --- Sugerencia de salida del depósito si la espera en el primer cliente es excesiva ---
    sugerencia_salida = None
    mensaje_sugerencia = None
    # Umbral de espera prolongada configurable
    # (Eliminado: sugerencia de salida y mensaje, ya no se usan)

    t1 = time.perf_counter()
    # Calcular hora de salida real del depósito para el primer cliente (solo si hay ruta)
    hora_salida_deposito = None
    if solution is not None:
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(index)
            index = solution.Value(routing.NextVar(index))
        if len(route) > 1:
            first_customer_idx = manager.IndexToNode(route[1])
            arrival = solution.Value(time_dimension.CumulVar(route[1]))
            travel_time = time_matrix[request.depot][first_customer_idx]
            hora_salida_min = arrival - travel_time
            hora_salida_deposito = {
                "minutos": hora_salida_min,
                "hhmm": min_to_hhmm(hora_salida_min),
                "primer_cliente": first_customer_idx
            }

    metadata = {
        "computation_time_ms": int((t1-t0)*1000),
        "num_vehicles": getattr(request, 'num_vehicles', len(getattr(request, 'vehicles', []))),
        "num_clients": len(getattr(request, 'locations', [])) - 1 if hasattr(request, 'locations') else None,
        "strict_mode": getattr(request, 'strict_mode', False),
        "buffer_minutes": getattr(request, 'buffer_minutes', 10),
        "peak_hours": getattr(request, 'peak_hours', None),
        "peak_buffer_minutes": getattr(request, 'peak_buffer_minutes', 20),
        "hora_salida_deposito": hora_salida_deposito
    }

    # --- Return FINAL asegurando metadata correcta ---
    return VRPAdvancedResponse(
        solution=build_vrp_solution(solution, routing, manager, request, time_dimension, time_windows, service_times, distance_matrix, time_matrix),
        metadata=metadata,
        warnings=[matrix_warning] if 'matrix_warning' in locals() and matrix_warning else None
    )

    # --- Ciclo externo para minimizar la espera en el primer cliente reconstruyendo el modelo ---
    max_wait_minutes = 10
    max_iters = 5
    iter_count = 0
    best_solution = None
    best_wait = None
    best_earliest_start = None
    earliest_start = None

    while iter_count < max_iters:
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
            VRPConstants.MAX_TIME_MIN,  # slack máximo permitido
            VRPConstants.MAX_TIME_MIN,  # tiempo máximo por ruta
            False,
            'Time')
        time_dimension = routing.GetDimensionOrDie('Time')
        penalizacion_espera = 1000
        time_dimension.SetSlackCostCoefficientForAllVehicles(penalizacion_espera)
        # Ajuste dinámico de la ventana del depósito para evitar salidas demasiado tempranas
        for vehicle_id, vehicle in enumerate(request.vehicles):
            depot_index = routing.Start(vehicle_id)
            start_time_orig = getattr(vehicle, 'start_time', VRPConstants.WORKDAY_START_MIN)
            end_time = getattr(vehicle, 'end_time', VRPConstants.WORKDAY_END_MIN)
            earliest_client_start = min([tw[0] for i, tw in enumerate(time_windows) if i != request.depot])
            min_travel_time = min([time_matrix[request.depot][i] for i in range(n) if i != request.depot])
            # En la segunda iteración en adelante, earliest_start se ajusta
            if earliest_start is not None:
                start_time = earliest_start
            else:
                start_time = max(start_time_orig, earliest_client_start - min_travel_time)
            time_dimension.CumulVar(depot_index).SetRange(start_time, end_time)
        for idx, window in enumerate(time_windows):
            if idx == request.depot:
                continue
            index = manager.NodeToIndex(idx)
            time_dimension.CumulVar(index).SetRange(window[0], window[1])
        for idx, st in enumerate(service_times):
            index = manager.NodeToIndex(idx)
            time_dimension.SlackVar(index).SetValue(st)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = getattr(request, 'solver_timeout', VRPConstants.DEFAULT_SOLVER_TIMEOUT_SEC)
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        solution = routing.SolveWithParameters(search_parameters)
        if solution is None:
            break
        # Extrae la espera en el primer cliente de la ruta del primer vehículo
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(index)
            index = solution.Value(routing.NextVar(index))
        if len(route) > 1:
            first_customer_idx = manager.IndexToNode(route[1])
            first_customer_index = route[1]
            arrival = solution.Value(time_dimension.CumulVar(first_customer_index))
            wait = arrival - time_windows[first_customer_idx][0]
            if wait < 0:
                wait = 0
        else:
            wait = 0
        if best_wait is None or wait < best_wait:
            best_wait = wait
            best_solution = solution
            best_earliest_start = start_time
        if wait <= max_wait_minutes:
            break
        # Ajusta earliest_start para la siguiente iteración
        depot_index = routing.Start(0)
        current_start = solution.Value(time_dimension.CumulVar(depot_index))
        earliest_start = current_start + (wait - max_wait_minutes)
        iter_count += 1
    solution = best_solution

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

    # Si no hay solución, retorna respuesta vacía pero válida
    if not solution:
        return VRPAdvancedResponse(solution={}, metadata=metadata, warnings=warnings_out)

    # --- Armado del objeto solution ---
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

        # Detalle de paradas para cada vehículo
        stops_detail = []
        current_time = VRPConstants.WORKDAY_START_MIN
        times = [current_time]
        service_starts = []
        service_ends = []
        stop = 0
        for i in range(1, len(r["route"])):
            prev_idx = r["route"][i-1]
            idx = r["route"][i]
            travel_time = time_matrix[prev_idx][idx]
            current_time += travel_time
            window_start = time_windows[idx][0] if idx < len(time_windows) else VRPConstants.WORKDAY_START_MIN
            real_arrival = current_time
            wait = max(0, window_start - current_time)
            start_service = real_arrival + wait
            service_duration = service_times[idx] if idx < len(service_times) else 0
            end_service = start_service + service_duration
            times.append(real_arrival)
            service_starts.append(start_service)
            service_ends.append(end_service)
            current_time = end_service
            stops_detail.append({
                "stop_index": stop+1,
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
            stop += 1
        details.append({
            "vehicle_id": r["vehicle_id"],
            "stops": stops_detail
        })

    solution = {
        "routes": routes,
        "total_distance": total_distance,
        "route_polylines": route_polylines,
        "route_points": route_points,
        "details": details
    }
    return VRPAdvancedResponse(
        solution=solution,
        metadata=metadata,
        warnings=warnings_out if warnings_out else None
    )

    # --- Retorno de seguridad: nunca debería llegar aquí, pero previene errores 500 si algo falla ---
    return VRPAdvancedResponse(solution={}, metadata=metadata, warnings=["Error interno: no se generó respuesta."])

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
