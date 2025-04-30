from pydantic import BaseModel, Field, validator
from typing import List, Optional

class Location(BaseModel):
    id: int
    name: Optional[str] = None
    lat: float
    lon: float
    weight: Optional[float] = None
    volume: Optional[float] = None
    time_window_start: Optional[int] = None  # minutos desde medianoche
    time_window_end: Optional[int] = None
    priority: Optional[int] = None

class Vehicle(BaseModel):
    id: int
    start_lat: float
    start_lon: float
    end_lat: Optional[float] = None
    end_lon: Optional[float] = None
    capacity_weight: Optional[float] = None
    capacity_volume: Optional[float] = None
    capacity_quantity: Optional[int] = None
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    use_quantity: Optional[bool] = False
    use_weight: Optional[bool] = False
    use_volume: Optional[bool] = False

class RouteOptimizationRequest(BaseModel):
    locations: List[Location]
    vehicles: Optional[List[Vehicle]] = None
    optimize_waypoints: bool = True  # Si es True, Google optimiza el orden; si False, respeta el orden dado.
    max_weight: Optional[float] = None
    max_volume: Optional[float] = None
    max_quantity: Optional[int] = None
    use_time_windows: Optional[bool] = False
    use_quantity: Optional[bool] = False
    use_weight: Optional[bool] = False
    use_volume: Optional[bool] = False

class RouteOptimizationResponse(BaseModel):
    route: List[List[int]]  # IDs de locations por vehículo
    total_distance: float  # en kilómetros
    geojson: Optional[dict] = None
    total_duration: Optional[float] = None  # en minutos u opcional
    details: Optional[str] = None

# Modelos para el endpoint vrp-capacity
class AdvancedLocation(BaseModel):
    id: int
    name: Optional[str] = None
    lat: float
    lon: float
    demand: Optional[int] = 0
    weight: Optional[float] = 0.0
    volume: Optional[float] = 0.0
    time_window: Optional[List[int]] = None  # [start, end] en minutos desde medianoche
    service_time: int = 5  # Tiempo de servicio en minutos (default=5)

    @validator('time_window')
    def validate_time_window(cls, v):
        if not v or not isinstance(v, list) or len(v) != 2:
            return [420, 1080]  # 07:00 a 18:00
        return v

class VRPCapacityRequest(BaseModel):
    locations: List[AdvancedLocation]  # Cada locación debe tener time_window=[start, end] y puede tener service_time (min)
    # Jornada laboral en minutos desde medianoche (ej: 420 = 07:00, 1080 = 18:00). Si no se envía, se usan los valores por defecto 07:00-18:00.
    workday_start: Optional[int] = None
    workday_end: Optional[int] = None
    num_vehicles: int = 1
    depot: int = 0
    vehicle_capacities_quantity: Optional[List[int]] = None
    vehicle_capacities_weight: Optional[List[float]] = None
    vehicle_capacities_volume: Optional[List[float]] = None
    mode: Optional[str] = "driving"
    units: Optional[str] = "metric"
    use_time_windows: Optional[bool] = False
    use_quantity: Optional[bool] = False
    use_weight: Optional[bool] = False
    use_volume: Optional[bool] = False

class RouteDetail(BaseModel):
    vehicle_id: int
    route: List[int]
    distance_km: float
    stops: List[dict]  # Cada dict: llegada, servicio, espera, etc.

class VRPAdvancedResponse(BaseModel):
    solution: dict  # Rutas, distancias, detalles, arrival_times, polylines...
    metadata: dict  # Info adicional (ej: tiempo de cómputo)
    warnings: Optional[List[dict]] = None
    excluded_clients: Optional[list] = None
    diagnostics: Optional[dict] = None
