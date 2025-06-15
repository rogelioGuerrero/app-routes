"""Modelos Pydantic para la API del solucionador VRP."""

from typing import List, Dict, Any, Optional, Literal, Tuple
from enum import Enum
from pydantic import BaseModel, Field, computed_field, validator

class VRPSolutionStatus(str, Enum):
    """Enum para los estados de la solución VRP."""
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    NO_SOLUTION_FOUND = "NO_SOLUTION_FOUND"
    INVALID_INPUT = "INVALID_INPUT"
    ERROR_SOLVING_PROBLEM = "ERROR_SOLVING_PROBLEM"

# Tipos de proveedores de matrices
MatrixProviderType = Literal['ors', 'google', 'euclidean']

class BaseLocation(BaseModel):
    """Ubicación base con identificador y geometría. La geometría se especifica únicamente a través de 'coords'."""
    id: str = Field(..., description="Identificador único de la ubicación")
    coords: List[float] = Field(..., description="Coordenadas como [longitud, latitud]", min_length=2, max_length=2)
    service_time: int = Field(0, ge=0, description="Tiempo de servicio en segundos")

    @computed_field
    @property
    def lat(self) -> float:
        """Latitud derivada de 'coords'."""
        return self.coords[1]

    @computed_field
    @property
    def lng(self) -> float:
        """Longitud derivada de 'coords'."""
        return self.coords[0]

class Location(BaseLocation):
    """Ubicación para un problema CVRP simple, con demanda no negativa."""
    demand: float = Field(0, ge=0, description="Demanda de la ubicación (debe ser 0 para depósitos, no negativa para clientes)")

class Vehicle(BaseModel):
    """Vehículo con capacidad y costos."""
    id: str = Field(..., description="Identificador único del vehículo")
    capacity: float = Field(..., gt=0, description="Capacidad del vehículo")
    fixed_cost: float = Field(0, ge=0, description="Costo fijo por usar este vehículo")
    cost_per_km: float = Field(0, ge=0, description="Costo por kilómetro recorrido")

class MatrixProviderConfig(BaseModel):
    """Configuración para un proveedor de matrices."""
    name: MatrixProviderType = Field(..., description="Nombre del proveedor")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parámetros específicos del proveedor")

class ProfileParameter(BaseModel):
    """Parámetro de perfil."""
    name: str
    type: str
    default: Any = None
    required: bool = True
    options: Optional[List[Any]] = None
    description: str

class ProfileDefinition(BaseModel):
    """Definición de perfil."""
    name: str
    description: str
    parameters: List[ProfileParameter]

class OptimizationProfile(BaseModel):
    """Perfil de optimización."""
    name: Literal['cost_saving', 'punctuality', 'balanced'] = Field(..., description="Perfil de optimización a usar")
    override_params: Dict[str, Any] = Field(default_factory=dict, description="Ajustes específicos para el perfil")

class CVRPRequest(BaseModel):
    """Solicitud para resolver un problema CVRP."""
    api_version: str = Field("1.0", description="Versión de la API")
    optimization_profile: OptimizationProfile = Field(..., description="Perfil de optimización y parámetros opcionales")
    locations: List[Location] = Field(..., min_items=2, description="Lista de ubicaciones (el primer elemento es el depósito)")
    vehicles: List[Vehicle] = Field(..., min_items=1, description="Lista de vehículos disponibles")
    providers: Optional[List[MatrixProviderConfig]] = Field(
        None,
        description="Proveedores de matrices a usar, en orden de preferencia. Si es None, se usarán ORS, Google y Euclidean"
    )
    force_refresh: bool = Field(False, description="Ignorar caché y forzar obtención de nuevas matrices")
    time_limit_seconds: int = Field(30, gt=0, le=300, description="Tiempo máximo de resolución en segundos")
    use_duration_matrix: bool = Field(True, description="Si es True, usa la matriz de duraciones para restricciones de tiempo")
    
    @validator('locations')
    def validate_locations(cls, v):
        # Verificar que al menos un depósito (demand=0) existe
        if not any(loc.demand == 0 for loc in v):
            raise ValueError("Al menos una ubicación debe tener demanda 0 (depósito)")
        return v

class RouteStop(BaseModel):
    """Parada en una ruta."""
    location_id: str
    coords: Optional[Tuple[float, float]] = Field(None, description="Coordenadas como (longitud, latitud)")
    arrival_time: Optional[float] = None
    departure_time: Optional[float] = None
    load: float
    distance_from_previous: float

class Route(BaseModel):
    """Ruta de un vehículo."""
    vehicle_id: str
    stops: List[RouteStop]
    total_distance: float
    total_load: float
    total_time: Optional[float] = None
    polyline_ors: Optional[str] = Field(None, description="Polyline codificado de OpenRouteService para la ruta")

class CVRPSolution(BaseModel):
    """Solución de un problema CVRP."""
    status: str
    routes: List[Route]
    total_distance: float
    total_load: float
    total_vehicles_used: int
    metadata: Dict[str, Any] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "OPTIMAL",
                "total_distance": 1000.5,
                "total_load": 150.0,
                "total_vehicles_used": 2,
                "routes": [
                    {
                        "vehicle_id": "veh1",
                        "total_distance": 600.5,
                        "total_load": 80.0,
                        "stops": [
                            {
                                "location_id": "depot",
                                "arrival_time": 0,
                                "departure_time": 0,
                                "load": 0,
                                "distance_from_previous": 0
                            },
                            # ... más paradas
                        ]
                    }
                ],
                "metadata": {
                    "matrix_provider": "ors",
                    "solver_time_seconds": 2.5,
                    "num_locations": 10,
                    "num_vehicles": 2
                }
            }
        }

# --- Modelos para optimización avanzada (VRPTW, PdD, peso/volumen) ---
class VRPTWLocation(BaseLocation):
    """Ubicación para VRPTW con pickup/delivery, ventanas y demandas de peso/volumen."""
    demand: float = Field(0, description="Demanda principal/general de la ubicación (positivo para entregas, negativo para recogidas, 0 para depósitos)")
    weight_demand: float = Field(0, description="Demanda de peso (positivo para entregas, negativo para recogidas)")
    volume_demand: float = Field(0, description="Demanda de volumen (positivo para entregas, negativo para recogidas)")
    time_window_start: int = Field(0, ge=0, description="Inicio de la ventana de tiempo en segundos desde la medianoche")
    time_window_end: int = Field(24*3600, ge=0, description="Fin de la ventana de tiempo en segundos desde la medianoche")
    pickup_id: Optional[str] = Field(None, description="ID de pickup para PdD")
    delivery_id: Optional[str] = Field(None, description="ID de delivery para PdD")
    required_skills: Optional[List[str]] = Field(default=None, description="Lista de habilidades requeridas en esta ubicación")

class ExtendedVehicle(Vehicle):
    """Vehículo con capacidades de peso y volumen extendidas y ubicación de inicio/fin."""
    weight_capacity: float = Field(..., gt=0, description="Capacidad de peso del vehículo")
    volume_capacity: float = Field(..., gt=0, description="Capacidad de volumen del vehículo")
    skills: Optional[List[str]] = Field(default=None, description="Lista de habilidades que posee el vehículo")
    start_location_id: Optional[str] = Field(None, description="ID de la ubicación de inicio del vehículo. Si es None, se usa el primer depósito.")
    end_location_id: Optional[str] = Field(None, description="ID de la ubicación de fin del vehículo. Si es None, se usa la ubicación de inicio.")
    ors_profile: Optional[str] = Field("driving-car", description="Perfil de OpenRouteService para generación de polylines (ej: 'driving-car', 'cycling-road')")

class VRPTWRequest(BaseModel):
    """Solicitud para resolver VRP con ventanas, peso/volumen y PdD."""
    api_version: str = Field("1.0", description="Versión de la API")
    optimization_profile: OptimizationProfile = Field(..., description="Perfil de optimización y parámetros opcionales")
    locations: List[VRPTWLocation] = Field(..., min_items=2, description="Lista de ubicaciones con ventanas y PdD")
    vehicles: List[ExtendedVehicle] = Field(..., min_items=1, description="Lista de vehículos con capacidades adicionales")
    providers: Optional[List[MatrixProviderConfig]] = Field(None, description="Proveedores de matrices")
    force_refresh: bool = Field(False, description="Ignorar caché y forzar nuevas matrices")
    time_limit_seconds: int = Field(30, gt=0, le=300, description="Tiempo máximo de resolución (s)")
    use_duration_matrix: bool = Field(True, description="Usar matriz de duraciones para restricciones de tiempo")

class VRPTWSolution(BaseModel):
    """Solución para VRP con ventanas, peso/volumen y PdD."""
    status: str
    routes: List[Route]
    total_distance: float
    total_load: float
    total_vehicles_used: int
    total_weight: float
    total_volume: float
    metadata: Dict[str, Any] = {}

# Nuevo modelo unificado
class UnifiedVRPRequest(BaseModel):
    """Solicitud unificada para resolver cualquier variante de VRP."""
    api_version: str = Field("1.0", description="Versión de la API")
    optimization_profile: OptimizationProfile = Field(..., description="Perfil de optimización y parámetros opcionales")
    locations: List[VRPTWLocation] = Field(..., min_items=2, description="Lista de ubicaciones con coordenadas, demanda, ventanas y PdD")
    vehicles: List[ExtendedVehicle] = Field(..., min_items=1, description="Lista de vehículos con capacidades de peso/volumen y costos")
    depots: Optional[List[int]] = Field(None, description="Índices de depósitos para MDVRP")

    @validator('depots', pre=True, always=True)
    def set_default_depot(cls, v, values):
        """Establece el depósito por defecto en el índice 0 si no se proporciona ninguno."""
        if v is None:
            return [0]
        return v
    starts_ends: Optional[List[List[int]]] = Field(None, description="Pares [start,end] para cada vehículo")
    pickups_deliveries: Optional[List[List[str]]] = Field(None, description="Pares de IDs de ubicación para tareas de recogida y entrega")
    allow_skipping_nodes: bool = Field(False, description="Permitir nodos opcionales con penalización")
    penalties: Optional[List[int]] = Field(None, description="Costo de penalización por no visitar cada nodo")
    max_route_duration: Optional[int] = Field(None, description="Duración máxima permitida para cada ruta")
    force_refresh: bool = Field(False, description="Ignorar caché de matrices")
    time_limit_seconds: int = Field(30, gt=0, le=300, description="Tiempo máximo de resolución en segundos")
    solver_params: Optional[Dict[str, Any]] = Field(None, description="Parámetros específicos del solver de OR-Tools")

class UnifiedVRPSolution(BaseModel):
    """Solución unificada para cualquier variante de VRP."""
    status: str
    routes: Optional[List[Route]] = None
    total_distance: float
    total_load: float
    total_vehicles_used: int
    total_weight: Optional[float] = None
    total_volume: Optional[float] = None
    metadata: Dict[str, Any]
