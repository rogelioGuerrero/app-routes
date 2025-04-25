from fastapi import APIRouter, HTTPException
from schemas import RouteOptimizationRequest, RouteOptimizationResponse
from utils import optimize_route_google
from typing import Any
from fastapi import Body

router = APIRouter()

@router.post("/optimize/route", response_model=RouteOptimizationResponse)
def optimize_route(
    request: RouteOptimizationRequest = Body(...)
) -> Any:
    """
    Endpoint para optimizar rutas usando Google Directions.
    - Por defecto, optimiza el orden de los waypoints.
    - Si optimize_waypoints=False en el request, respeta el orden dado (ideal para rutas de VRP avanzado).
    Ejemplo de request:
    {
      "locations": [...],
      "vehicles": [...],
      "optimize_waypoints": false
    }
    """
    try:
        if len(request.locations) < 2:
            raise ValueError("Se requieren al menos origen y destino en locations.")
        origin = f"{request.locations[0].lat},{request.locations[0].lon}"
        destination = f"{request.locations[-1].lat},{request.locations[-1].lon}"
        waypoints = [f"{loc.lat},{loc.lon}" for loc in request.locations[1:-1]]
        google_response = optimize_route_google(origin, destination, waypoints, optimize_waypoints=request.optimize_waypoints)
        print("Respuesta Google Directions:", google_response)  # Para depuración automática
        # Procesar la respuesta de Google
        if not google_response.get("routes"):
            error_msg = google_response.get("status", "Sin status")
            error_detail = google_response.get("error_message", str(google_response))
            raise ValueError(f"La API de Google no retornó rutas. Status: {error_msg}. Detalle: {error_detail}")
        route_data = google_response["routes"][0]
        legs = route_data["legs"]
        total_distance = sum(leg["distance"]["value"] for leg in legs) / 1000  # km
        total_duration = sum(leg["duration"]["value"] for leg in legs) / 60    # min
        # Obtener el orden óptimo de los waypoints
        order = google_response["routes"][0].get("waypoint_order", list(range(len(waypoints))))
        # Reconstruir el orden de IDs
        ordered_ids = [request.locations[0].id] + [request.locations[i+1].id for i in order] + [request.locations[-1].id]
        route = [ordered_ids]
        details = "\n".join([
            f"Leg {i+1}: {leg['start_address']} -> {leg['end_address']} | Distancia: {leg['distance']['text']} | Duración: {leg['duration']['text']}"
            for i, leg in enumerate(legs)
        ])
        return RouteOptimizationResponse(
            route=route,
            total_distance=total_distance,
            geojson=route_data.get("overview_polyline"),
            total_duration=total_duration,
            details=details
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en optimización: {str(e)}")
