from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# Modelos para entrada de restricciones
class AdvancedLocation(BaseModel):
    id: int
    name: Optional[str] = None
    lat: float
    lon: float
    time_window: Optional[List[int]] = None  # [start_min, end_min] desde inicio del día
    demand: Optional[int] = 0

class VRPAdvancedRequest(BaseModel):
    locations: List[AdvancedLocation]
    num_vehicles: int = 1
    depot: int = 0
    vehicle_capacities: Optional[List[int]] = None  # Si hay restricciones de capacidad
    mode: Optional[str] = "driving"
    units: Optional[str] = "metric"
    use_time_windows: Optional[bool] = False
    use_demands: Optional[bool] = False

class VRPAdvancedResponse(BaseModel):
    routes: List[List[int]]
    total_distance: float
    total_duration: Optional[float] = None
    details: Optional[List[str]] = None

@router.post("/vrp-advanced", response_model=VRPAdvancedResponse)
def vrp_advanced(request: VRPAdvancedRequest):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="GOOGLE_API_KEY no encontrada en .env")

    # 1. Obtener matriz de distancias y duraciones
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
    data = response.json()
    if data.get("status") != "OK":
        raise HTTPException(status_code=400, detail=f"Google Distance Matrix error: {data.get('status')} - {data.get('error_message')}")
    distance_matrix = [[el["distance"]["value"] for el in row["elements"]] for row in data["rows"]]  # metros
    duration_matrix = [[el["duration"]["value"] for el in row["elements"]] for row in data["rows"]]  # segundos

    # 2. Configuración de OR-Tools
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), request.num_vehicles, request.depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Restricción: capacidad de vehículos
    if request.use_demands and request.vehicle_capacities:
        demands = [loc.demand or 0 for loc in request.locations]
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return demands[from_node]
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            request.vehicle_capacities,  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')

    # Restricción: ventanas de tiempo
    if request.use_time_windows:
        time_windows = [loc.time_window if loc.time_window else [0, 24*60] for loc in request.locations]
        def duration_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return duration_matrix[from_node][to_node]
        duration_callback_index = routing.RegisterTransitCallback(duration_callback)
        routing.AddDimension(
            duration_callback_index,
            30*60,  # allow waiting time (en segundos)
            24*60*60,  # maximum route duration (en segundos)
            False,  # Don't force start cumul to zero
            'Time')
        time_dimension = routing.GetDimensionOrDie('Time')
        for idx, window in enumerate(time_windows):
            index = manager.NodeToIndex(idx)
            time_dimension.CumulVar(index).SetRange(window[0]*60, window[1]*60)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = 20

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        raise HTTPException(status_code=400, detail="No se encontró solución con OR-Tools (avanzado).")

    routes = []
    details = []
    total_distance = 0
    total_duration = 0
    for vehicle_id in range(request.num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        route_distance = 0
        route_duration = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            if request.use_time_windows:
                time_dimension = routing.GetDimensionOrDie('Time')
                arrival = solution.Min(time_dimension.CumulVar(previous_index))
                details.append(f"Nodo {node}: llegada a {arrival//60} min")
            if request.use_demands:
                details.append(f"Nodo {node}: demanda {demands[node]}")
        route.append(manager.IndexToNode(index))
        routes.append(route)
        total_distance += route_distance
    return VRPAdvancedResponse(
        routes=routes,
        total_distance=total_distance/1000,  # metros a km
        details=details
    )
