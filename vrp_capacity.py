from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from dotenv import load_dotenv

from schemas import AdvancedLocation, VRPCapacityRequest, VRPAdvancedResponse
from route_polyline_utils import get_route_polyline_and_geojson

load_dotenv()

router = APIRouter()

@router.post("/vrp-capacity", response_model=VRPAdvancedResponse)
def vrp_capacity(request: VRPCapacityRequest):
    # Construcción de matriz de distancias real usando Google Distance Matrix API
    import requests
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
    # Procesar las matrices de distancia y tiempo
    n = len(request.locations)
    distance_matrix = []
    time_matrix = []
    for row in data["rows"]:
        distance_row = [el["distance"]["value"]/1000 if el.get("distance") else 1e6 for el in row["elements"]]  # km
        time_row = [el["duration"]["value"]//60 if el.get("duration") else 1e6 for el in row["elements"]]  # minutos
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

    # Restricción de cantidad de paquetes
    if request.use_quantity and request.vehicle_capacities_quantity:
        demands = [loc.demand or 0 for loc in request.locations]
        def demand_callback(from_index):
            return demands[manager.IndexToNode(from_index)]
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,
            request.vehicle_capacities_quantity,
            True,
            'Quantity')
    # Restricción de peso
    if request.use_weight and request.vehicle_capacities_weight:
        weights = [getattr(loc, 'weight', 0.0) or 0.0 for loc in request.locations]
        def weight_callback(from_index):
            return int(weights[manager.IndexToNode(from_index)])
        weight_callback_index = routing.RegisterUnaryTransitCallback(weight_callback)
        routing.AddDimensionWithVehicleCapacity(
            weight_callback_index,
            0,
            [int(w) for w in request.vehicle_capacities_weight],
            True,
            'Weight')
    # Restricción de volumen
    if request.use_volume and request.vehicle_capacities_volume:
        volumes = [getattr(loc, 'volume', 0.0) or 0.0 for loc in request.locations]
        def volume_callback(from_index):
            return int(volumes[manager.IndexToNode(from_index)] * 1000)  # Escala a enteros si es necesario
        volume_callback_index = routing.RegisterUnaryTransitCallback(volume_callback)
        routing.AddDimensionWithVehicleCapacity(
            volume_callback_index,
            0,
            [int(v * 1000) for v in request.vehicle_capacities_volume],
            True,
            'Volume')
    # Ventanas de tiempo (opcional)
    if request.use_time_windows:
        # Garantiza que cada locación tiene time_window como par [start, end]
        time_windows = []
        for loc in request.locations:
            tw = getattr(loc, 'time_window', None)
            if not tw or not isinstance(tw, list) or len(tw) != 2:
                time_windows.append([0, 1440])
            else:
                time_windows.append([tw[0], tw[1]])
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
    def min_to_hhmm(m):
        h = int(m) // 60
        mm = int(m) % 60
        return f"{h:02d}:{mm:02d}"

    # Leer jornada laboral del request o usar valores por defecto
    workday_start = request.workday_start if request.workday_start is not None else 420  # 07:00
    workday_end = request.workday_end if request.workday_end is not None else 1080      # 18:00

    warnings = []

    for vehicle_id in range(request.num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        route_distance = 0
        times = []  # Hora de llegada física
        service_starts = []  # Hora de inicio de servicio (considerando espera)
        service_ends = []    # Hora de salida (inicio_servicio + service_time)
        current_time = workday_start
        times.append(current_time)
        service_time_list = [getattr(loc, 'service_time', 0) for loc in request.locations]
        # time_windows ya está garantizado como par [start, end]
        for_stop_windows = time_windows if request.use_time_windows else [[0, 1440] for _ in request.locations]
        stop = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            travel_time = time_matrix[manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
            current_time += travel_time
            # Espera si llega antes de la ventana
            if not routing.IsEnd(index):
                window_start = for_stop_windows[index] [0]
                real_arrival = current_time
                wait = max(0, window_start - current_time)
                start_service = real_arrival + wait
                service_duration = service_time_list[index] if index < len(service_time_list) else 0
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
        # Detalle por parada
        dets = []
        for i in range(1, len(route)-1):  # Saltar el depósito inicial/final
            arr = times[i]
            serv = service_starts[i-1] if i-1 < len(service_starts) else arr
            end = service_ends[i-1] if i-1 < len(service_ends) else serv
            wait = max(0, serv - arr)
            stime = service_time_list[route[i]] if route[i] < len(service_time_list) else 0
            dets.append(f"Parada {i}: Punto {route[i]} | Hora de llegada {min_to_hhmm(arr)} | Tiempo de espera: {wait}min | Tiempo de servicio {stime}min | Inicia servicio: {min_to_hhmm(serv)} - Fin de servicio: {min_to_hhmm(end)}")
        details.append(f"Vehículo {vehicle_id+1}: {route} | Distancia: {route_distance:.2f} km\n" + "\n".join(dets))
        arrival_times.append(service_starts)
        # Validar si alguna entrega supera el fin de jornada
        for i, t in enumerate(service_ends):
            if t > workday_end:
                warnings.append(f"Vehículo {vehicle_id+1}: Entrega en parada {i+1} fuera de jornada laboral ({min_to_hhmm(t)} > {min_to_hhmm(workday_end)})")

    # Formatear los tiempos de llegada después de calcularlos
    arrival_times_formatted = [[min_to_hhmm(t) for t in times] for times in arrival_times]

    # Obtener polylines y geojson para cada ruta optimizada
    route_polylines = []
    route_geojson = []
    for route in routes:
        # Construir lista de (lat, lon) en orden de visita
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
