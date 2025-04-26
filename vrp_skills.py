from fastapi import APIRouter, HTTPException
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from typing import List
from schemas_skills import VRPSkillsRequest, SkillsLocation, SkillsVehicle
from route_polyline_utils import get_route_polyline_and_geojson
from schemas import VRPAdvancedResponse

router = APIRouter()

@router.post("/vrp-skills")
def vrp_skills(request: VRPSkillsRequest):
    """
    Endpoint avanzado de VRP con skills, ventanas de tiempo y geometría comprimida.
    Devuelve la misma estructura enriquecida que /vrp-capacity (rutas, detalles, tiempos, polylines, geojson).
    """
    import os
    import requests
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="GOOGLE_API_KEY no encontrada en .env")
    origins = [f"{loc.lat},{loc.lon}" for loc in request.locations]
    destinations = origins
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": "|".join(origins),
        "destinations": "|".join(destinations),
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
    manager = pywrapcp.RoutingIndexManager(n, request.num_vehicles, request.depot)
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # Skills logic: only allow assignment if vehicle provides all required_skills for the location
    vehicle_skills = [set(v.provided_skills or []) for v in request.vehicles]
    location_skills = [set(l.required_skills or []) for l in request.locations]
    for node_idx, req_skills in enumerate(location_skills):
        if not req_skills:
            continue
        for vehicle_idx, prov_skills in enumerate(vehicle_skills):
            if not req_skills.issubset(prov_skills):
                routing.VehicleVar(manager.NodeToIndex(node_idx)).RemoveValue(vehicle_idx)
    # Time windows and service time
    time_windows = []
    service_times = []
    for loc in request.locations:
        tw = loc.time_window if loc.time_window and len(loc.time_window) == 2 else [420, 1080]
        time_windows.append(tw)
        service_times.append(getattr(loc, 'service_time', 5))
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(time_matrix[from_node][to_node])
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        1440,
        1440,
        False,
        'Time')
    time_dimension = routing.GetDimensionOrDie('Time')
    for idx, window in enumerate(time_windows):
        index = manager.NodeToIndex(idx)
        time_dimension.CumulVar(index).SetRange(window[0], window[1])
    # Add service time at each stop
    for idx, st in enumerate(service_times):
        index = manager.NodeToIndex(idx)
        time_dimension.SlackVar(index).SetValue(st)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = 10
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        raise HTTPException(status_code=400, detail="No se encontró solución factible para las restricciones dadas.")
    routes = []
    total_distance = 0
    details = []
    arrival_times = []
    warnings = []
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
            route_distance += distance_matrix[manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
            stop += 1
        route.append(manager.IndexToNode(index))
        routes.append(route)
        total_distance += route_distance
        dets = []
        for i in range(1, len(route)-1):
            arr = times[i]
            serv = service_starts[i-1] if i-1 < len(service_starts) else arr
            end = service_ends[i-1] if i-1 < len(service_ends) else serv
            wait = max(0, serv - arr)
            stime = service_times[route[i]] if route[i] < len(service_times) else 0
            dets.append(f"Stop {i}: Nodo {route[i]} llegada {min_to_hhmm(arr)} espera {wait}min servicio {stime}min inicia {min_to_hhmm(serv)} sale {min_to_hhmm(end)}")
        details.append(f"Vehículo {vehicle_id+1}: {route} | Distancia: {route_distance:.2f} km\n" + "\n".join(dets))
        arrival_times.append(service_starts)
        for i, t in enumerate(service_ends):
            if t > workday_end:
                warnings.append(f"Vehículo {vehicle_id+1}: Entrega en parada {i+1} fuera de jornada laboral ({min_to_hhmm(t)} > {min_to_hhmm(workday_end)})")
    arrival_times_formatted = [[min_to_hhmm(t) for t in times] for times in arrival_times]
    route_polylines = []
    route_geojson = []
    for route in routes:
        latlons = [(request.locations[idx].lat, request.locations[idx].lon) for idx in route]
        try:
            polyline, geojson = get_route_polyline_and_geojson(latlons)
        except Exception:
            polyline, geojson = None, None
        route_polylines.append(polyline)
        route_geojson.append(geojson)
    return VRPAdvancedResponse(
        routes=routes,
        total_distance=round(total_distance, 2),
        details=details,
        arrival_times=arrival_times,
        arrival_times_formatted=arrival_times_formatted,
        warnings=warnings,
        route_polylines=route_polylines,
        route_geojson=route_geojson
    )
