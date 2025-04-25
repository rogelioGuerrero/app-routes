from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

router = APIRouter()

class VRPRequest(BaseModel):
    distance_matrix: List[List[float]]
    num_vehicles: int = 1
    depot: int = 0  # nodo de inicio/fin

class VRPResponse(BaseModel):
    routes: List[List[int]]
    total_distance: float

@router.post("/solve-vrp", response_model=VRPResponse)
def solve_vrp(request: VRPRequest):
    # OR-Tools espera enteros: convertimos km a metros y redondeamos
    distance_matrix = [[int(d * 1000) for d in row] for row in request.distance_matrix]
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), request.num_vehicles, request.depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        raise HTTPException(status_code=400, detail="No se encontró solución con OR-Tools.")

    routes = []
    total_distance = 0
    for vehicle_id in range(request.num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        route_distance = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        route.append(manager.IndexToNode(index))  # Añadir el nodo final
        routes.append(route)
        total_distance += route_distance
    # Convertir distancia total a km
    return VRPResponse(routes=routes, total_distance=total_distance / 1000)
