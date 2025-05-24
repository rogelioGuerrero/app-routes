from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

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

    @field_validator('time_window', mode='before')
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

class Location(BaseModel):
    """Ubicación con coordenadas y metadatos."""
    id: int
    lat: float
    lon: float
    name: str = ""
    address: str = ""
    demand: float = 0.0
    weight: float = 0.0
    volume: float = 0.0
    time_window: List[int] = [0, 1440]  # Ventana de tiempo por defecto: todo el día
    time_window_start: Optional[int] = None  # Para compatibilidad
    time_window_end: Optional[int] = None    # Para compatibilidad
    service_time: int = 0
    priority: int = 1    # prioridad de la ubicación (mayor = más importante)
    is_depot: bool = False
    required_skills: List[str] = []
    
    class Config:
        extra = 'allow'  # Permite campos adicionales
        
    def __init__(self, **data):
        # Asegurar que los campos numéricos tengan valores predeterminados
        for field in ['demand', 'weight', 'volume']:
            if field in data and data[field] is None:
                data[field] = 0.0
        super().__init__(**data)


class VehicleRoute(BaseModel):
    """Ruta asignada a un vehículo."""
    vehicle_id: int
    vehicle_name: Optional[str] = None
    locations: List[Location] = []
    distance: float = 0.0  # en km
    time: float = 0.0      # en minutos
    time_str: Optional[str] = None  # tiempo formateado como HH:MM
    load: float = 0.0      # carga total de la ruta
    
    class Config:
        extra = 'allow'  # Permite campos adicionales


class VRPAdvancedResponse(BaseModel):
    """Respuesta del servicio VRP avanzado."""
    routes: List[VehicleRoute] = []
    total_distance: float = 0.0
    total_time: float = 0.0
    unassigned_locations: List[int] = []
    warnings: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    
    class Config:
        extra = 'allow'  # Permite campos adicionales para compatibilidad


# Modelos para compatibilidad con versiones anteriores
class LegacyVRPResponse(BaseModel):
    """Modelo de respuesta para compatibilidad con versiones anteriores."""
    solution: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    warnings: Optional[List[Dict[str, Any]]] = None
    excluded_clients: Optional[List[int]] = None
    diagnostics: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = 'allow'  # Permite campos adicionales


class VRPSolutionStatus(str, Enum):
    """Estado de la solución del VRP."""
    SUCCESS = "success"
    PARTIAL = "partial"  # Algunas ubicaciones no pudieron ser asignadas
    FAILED = "failed"    # No se pudo encontrar una solución factible
    ERROR = "error"      # Ocurrió un error durante el procesamiento


class VRPSolverType(str, Enum):
    """Tipos de solvers disponibles para el VRP."""
    OR_TOOLS = "or_tools"
    HEURISTIC = "heuristic"
    EXACT = "exact"
    
    
class RouteOptimizationMetrics(BaseModel):
    """Métricas de optimización de rutas."""
    total_distance: float  # en kilómetros
    total_time: float      # en minutos
    total_stops: int       # número total de paradas
    vehicles_used: int     # número de vehículos utilizados
    utilization_rate: float  # tasa de utilización de vehículos (0-1)
    
    class Config:
        extra = 'allow'  # Permite campos adicionales


class RouteOptimizationResult(BaseModel):
    """Resultado de la optimización de rutas."""
    status: VRPSolutionStatus
    routes: List[VehicleRoute]
    metrics: RouteOptimizationMetrics
    unassigned: List[Location] = []
    warnings: List[Dict[str, Any]] = []
    execution_time: float  # en segundos
    
    class Config:
        extra = 'allow'  # Permite campos adicionales
