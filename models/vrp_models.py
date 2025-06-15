"""Modelos de datos para el problema VRP."""

from enum import Enum
from typing import List, Optional, Dict, Any, Union, Literal, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator, HttpUrl

# Enum para el estado de la solución
class VRPSolutionStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    NO_SOLUTION_FOUND = "NO_SOLUTION_FOUND"
    ERROR_SOLVING_PROBLEM = "ERROR_SOLVING_PROBLEM"
    INVALID_INPUT = "INVALID_INPUT"


class Location(BaseModel):
    """Ubicación con coordenadas y demanda."""
    id: Union[int, str] = Field(..., description="Identificador único de la ubicación")
    coords: Optional[List[float]] = Field(default=None, description="Coordenadas como [longitud, latitud]")
    lat: Optional[float] = Field(default=None, required=False, description="Latitud en grados decimales")
    lng: Optional[float] = Field(default=None, required=False, description="Longitud en grados decimales")
    demand: float = Field(0, description="Demanda de la ubicación (positivo para entregas, negativo para recogidas, 0 para depósitos)")
    weight_demand: float = Field(0, description="Demanda de peso (positivo para entregas, negativo para recogidas)")
    volume_demand: float = Field(0, description="Demanda de volumen (positivo para entregas, negativo para recogidas)")
    service_time: int = Field(0, ge=0, description="Tiempo de servicio en segundos")
    time_window_start: Optional[int] = Field(0, ge=0, description="Inicio de la ventana de tiempo en segundos desde la medianoche")
    time_window_end: Optional[int] = Field(24*3600, ge=0, description="Fin de la ventana de tiempo en segundos desde la medianoche")
    pickup_id: Optional[str] = Field(None, description="ID de recogida asociado (para problemas de recogida y entrega)")
    delivery_id: Optional[str] = Field(None, description="ID de entrega asociado (para problemas de recogida y entrega)")
    required_skills: Optional[List[str]] = None
    
    @validator('coords')
    def validate_coords_length(cls, v):
        if v is not None and len(v) != 2:
            raise ValueError('coords debe ser una lista de dos floats: [longitud, latitud]')
        return v

    @validator('id')
    def convert_id_to_str(cls, v):
        return str(v)
    
    @validator('demand', 'service_time', 'volume_demand', 'weight_demand')
    def convert_to_float(cls, v):
        return float(v)
    
    @validator('time_window_end')
    def validate_time_window(cls, v, values):
        if 'time_window_start' in values and v is not None and values['time_window_start'] is not None:
            if v <= values['time_window_start']:
                raise ValueError('time_window_end debe ser mayor que time_window_start')
        return v

    @root_validator(pre=True)
    def populate_lat_lng_from_coords_or_ensure_lat_lng_present(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        print(f"DEBUG VRP_MODELS: root_validator received values: {values}")
        coords = values.get('coords')
        # lat and lng are fetched later after potential modification by coords


        if coords is not None:
            if isinstance(coords, list) and len(coords) == 2:
                # If coords are provided, they populate lat/lng
                values['lng'] = coords[0]
                values['lat'] = coords[1]
                # lat and lng fields are now guaranteed to be populated if coords was valid
            else:
                # If coords is present but malformed, the specific 'coords' field validator will catch it.
                # We don't raise here to allow that validator to provide a more specific error.
                # However, lat/lng might still be missing, leading to the check below.
                pass
        
        # After attempting to populate from coords, check if lat/lng are now effectively present.
        # This covers cases where coords were not provided, or were malformed and didn't populate lat/lng.
        current_lat = values.get('lat')
        current_lng = values.get('lng')

        if current_lat is None or current_lng is None:
            print(f"DEBUG VRP_MODELS: Raising ValueError. current_lat: {current_lat}, current_lng: {current_lng}, original_coords_in_values: {values.get('coords')}")
            # This error means that neither 'coords' successfully populated lat/lng, 
            # nor were 'lat' and 'lng' provided directly in the input.
            raise ValueError("La ubicación debe tener 'coords' válidas o ambos 'lat' y 'lng' explícitos.")
        
        print(f"DEBUG VRP_MODELS: root_validator returning values: {values}")
        return values

class Vehicle(BaseModel):
    """Vehículo con capacidad y costos."""
    id: Union[int, str] = Field(..., description="Identificador único del vehículo")
    capacity: float = Field(..., gt=0, description="Capacidad del vehículo")
    start_location_id: Optional[Union[int, str]] = Field(None, description="ID de la ubicación de inicio (si es diferente del depósito)")
    end_location_id: Optional[Union[int, str]] = Field(None, description="ID de la ubicación de fin (si es diferente del depósito)")
    fixed_cost: float = Field(0, ge=0, description="Costo fijo por usar este vehículo")
    cost_per_km: float = Field(0, ge=0, description="Costo por kilómetro recorrido")
    max_travel_time: Optional[int] = Field(None, ge=0, description="Tiempo máximo de viaje en segundos")
    skills: Optional[List[str]] = None
    
    @validator('id', 'start_location_id', 'end_location_id')
    def convert_ids_to_str(cls, v):
        if v is not None:
            return str(v)
        return None
    
    @validator('capacity', 'fixed_cost', 'cost_per_km')
    def convert_to_float(cls, v):
        return float(v)

class RouteStop(BaseModel):
    """Parada en una ruta."""
    location_id: str = Field(..., description="ID de la ubicación")
    coords: Optional[Tuple[float, float]] = Field(None, description="Coordenadas como (longitud, latitud)")
    arrival_time: int = Field(..., description="Hora de llegada en segundos")
    departure_time: int = Field(..., description="Hora de salida en segundos")
    load: float = Field(..., description="Carga del vehículo después de la visita")
    weight: Optional[float] = Field(0, description="Peso del vehículo después de la visita")
    volume: Optional[float] = Field(0, description="Volumen del vehículo después de la visita")
    distance_from_previous: int = Field(..., description="Distancia desde la parada anterior en metros")
    travel_time_to_stop: Optional[int] = Field(0, description="Tiempo de viaje desde la parada anterior")
    service_time_at_stop: Optional[int] = Field(0, description="Tiempo de servicio en la parada")
    wait_time: Optional[float] = Field(None, description="Tiempo de espera en segundos (si hay ventanas de tiempo)")

class Route(BaseModel):
    """Ruta de un vehículo."""
    vehicle_id: str = Field(..., description="ID del vehículo asignado a esta ruta")
    stops: List[RouteStop] = Field(..., description="Lista de paradas en la ruta")
    total_distance: int = Field(..., description="Distancia total de la ruta en metros")
    total_load: float = Field(..., description="Carga total transportada")
    total_weight: Optional[float] = Field(0, description="Peso total transportado")
    total_volume: Optional[float] = Field(0, description="Volumen total transportado")
    total_time: int = Field(..., description="Tiempo total de la ruta en segundos")
    total_cost: Optional[float] = Field(None, description="Costo total de la ruta")

class VRPSolution(BaseModel):
    """Solución de un problema VRP."""
    status: VRPSolutionStatus = Field(..., description="Estado de la solución")
    routes: List[Route] = Field(..., description="Lista de rutas asignadas a vehículos")
    total_distance: float = Field(..., description="Distancia total de todas las rutas en metros")
    total_load: float = Field(..., description="Carga total transportada")
    total_vehicles_used: int = Field(..., description="Número total de vehículos utilizados")
    total_cost: Optional[float] = Field(None, description="Costo total de la solución")
    execution_time: Optional[float] = Field(None, description="Tiempo de ejecución en segundos")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadatos adicionales sobre la solución"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "OPTIMAL",
                "routes": [
                    {
                        "vehicle_id": "veh1",
                        "stops": [
                            {
                                "location_id": "depot",
                                "arrival_time": 0,
                                "departure_time": 0,
                                "load": 0,
                                "distance_from_previous": 0,
                                "cumulative_distance": 0
                            },
                            {
                                "location_id": "loc1",
                                "arrival_time": 300,
                                "departure_time": 360,
                                "load": 10,
                                "distance_from_previous": 5000,
                                "cumulative_distance": 5000
                            },
                            {
                                "location_id": "depot",
                                "arrival_time": 600,
                                "departure_time": 600,
                                "load": 0,
                                "distance_from_previous": 5000,
                                "cumulative_distance": 10000
                            }
                        ],
                        "total_distance": 10000,
                        "total_load": 10,
                        "total_time": 600,
                        "total_cost": 150.5
                    }
                ],
                "total_distance": 10000,
                "total_load": 10,
                "total_vehicles_used": 1,
                "total_cost": 150.5,
                "execution_time": 2.5,
                "metadata": {
                    "solver_version": "1.0.0",
                    "solved_at": "2023-10-01T12:00:00Z",
                    "constraints_violated": []
                }
            }
        }


class SolverConfig(BaseModel):
    """Configuración para el solver."""
    allow_skipping_nodes: bool = Field(False, description="Permitir que el solver descarte nodos (ubicaciones) si es necesario, aplicando una penalización.")
    default_penalty_for_dropping_nodes: int = Field(1_000_000, description="Penalización por defecto si se descarta un nodo y no tiene una penalización específica.")
    time_limit_seconds: int = Field(30, ge=1, description="Límite de tiempo para la ejecución del solver en segundos.")
    first_solution_strategy: Optional[str] = Field("AUTOMATIC", description="Estrategia para encontrar la primera solución (ej. PATH_CHEAPEST_ARC, SAVINGS, SWEEP).")
    local_search_metaheuristic: Optional[str] = Field("AUTOMATIC", description="Metaheurística para la búsqueda local (ej. GUIDED_LOCAL_SEARCH, TABU_SEARCH).")
    log_search: bool = Field(False, description="Habilitar logs detallados de la búsqueda del solver OR-Tools.")

class VrpProblemData(BaseModel):
    """Datos completos para un problema VRP, incluyendo ubicaciones, vehículos y configuración del solver."""
    locations: List[Location] = Field(..., description="Lista de todas las ubicaciones, incluyendo depósitos y clientes.")
    vehicles: List[Vehicle] = Field(..., description="Lista de todos los vehículos disponibles.")
    depots: List[Location] = Field(..., description="Lista de ubicaciones que actúan como depósitos.") # Podría derivarse o ser explícita
    solver_config: SolverConfig = Field(default_factory=SolverConfig, description="Configuración específica para el comportamiento del solver.")
    matrix_profile: str = Field("driving-car", description="Perfil de enrutamiento para ORS (ej. driving-car, cycling-road).")
    use_distance_matrix: bool = Field(True, description="Indica si se debe usar una matriz de distancias precalculada o calcularla dinámicamente.")
    distance_matrix: Optional[List[List[float]]] = Field(None, description="Matriz de distancias precalculada (opcional).")
    duration_matrix: Optional[List[List[float]]] = Field(None, description="Matriz de duraciones precalculada (opcional).")

    @validator('depots', pre=True, always=True)
    def validate_depots_from_locations(cls, v, values):
        locations = values.get('locations')
        if not locations:
            raise ValueError("Se requieren ubicaciones (locations) para definir los depósitos.")
        
        # Si 'v' (depots) no se proporciona explícitamente, se infiere de las locations con demanda 0
        if v is None:
            inferred_depots = [loc for loc in locations if loc.demand == 0]
            if not inferred_depots:
                # Permitir que no haya depósitos si se define un start/end_location_id en los vehículos
                # O si es un problema de tipo 'open VRP' donde no se regresa al depósito.
                # Por ahora, asumimos que al menos un depósito es necesario si no se especifica explícitamente.
                # Esta lógica podría necesitar refinamiento basado en casos de uso más complejos.
                pass # No se levanta error si no hay depósitos inferidos, puede ser intencional
            return inferred_depots
        
        # Si 'v' se proporciona, validar que los IDs de los depósitos existan en 'locations'
        depot_ids_in_locations = {loc.id for loc in locations}
        for depot_loc in v:
            if depot_loc.id not in depot_ids_in_locations:
                raise ValueError(f"El depósito con ID '{depot_loc.id}' no se encuentra en la lista de ubicaciones.")
        return v

    def get_summary(self) -> Dict[str, Any]:
        """Devuelve un resumen del problema."""
        return {
            "num_locations": len(self.locations),
            "num_vehicles": len(self.vehicles),
            "num_depots": len(self.depots),
            "solver_config": self.solver_config.model_dump() if self.solver_config else None,
            "matrix_profile": self.matrix_profile
        }

# --- Solución avanzada con peso y volumen ---
class VRPTWSolution(VRPSolution):
    """Solución de un problema VRP con ventanas, PdD, peso y volumen."""
    total_weight: float = Field(..., description="Peso total transportado")
    total_volume: float = Field(..., description="Volumen total transportado")
    total_time: Optional[float] = Field(None, description="Tiempo total de la solución en segundos")
