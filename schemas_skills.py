from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal
import logging
from vrp_constants import (
    DEFAULT_SERVICE_TIME, DEFAULT_BUFFER_MINUTES, DEFAULT_PEAK_HOURS, DEFAULT_PEAK_MULTIPLIER,
    DEFAULT_TIME_WINDOW, DEFAULT_UNITS, DEFAULT_MODE
)
from schemas_matrix import MatrixApiConfig
from enum import Enum

class OptimizationProfile(str, Enum):
    """Perfiles predefinidos para la optimización de rutas.
    
    Estos perfiles ofrecen configuraciones predefinidas para diferentes escenarios
    operativos, optimizando el balance entre tiempo de cómputo y calidad de solución.
    """
    STANDARD_OPERATIONS = "standard_operations"
    FAST_DELIVERY = "fast_delivery"
    COST_EFFICIENT = "cost_efficient"
    EXTENDED_OPERATIONS = "extended_operations"
    RAPID_RESPONSE = "rapid_response"
    BALANCED = "balanced"

class OptimizationObjective(str, Enum):
    MINIMIZE_DISTANCE = "minimize_distance"
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_COST = "minimize_cost"

class SolverOptions(BaseModel):
    """Opciones de configuración para el solver de optimización.
    
    Se puede configurar manualmente cada parámetro o usar un perfil predefinido.
    Si se especifica un perfil, sobrescribirá los demás parámetros.
    
    Perfiles predefinidos disponibles:
    - STANDARD_OPERATIONS: Configuración equilibrada para operaciones diurnas estándar.
    - FAST_DELIVERY: Prioriza velocidad de entrega sobre distancia total.
    - COST_EFFICIENT: Optimizado para minimizar distancias y costos operativos.
    - EXTENDED_OPERATIONS: Para planificación detallada con tiempo de cómputo extendido.
    - RAPID_RESPONSE: Para obtener soluciones rápidas en aplicaciones interactivas.
    - BALANCED: Equilibrio general entre distancia, tiempo y costos.
    
    Ejemplo de uso con perfil predefinido:
        opciones = SolverOptions(profile=OptimizationProfile.FAST_DELIVERY)
        
    Ejemplo de configuración manual:
        opciones = SolverOptions(
            optimization_objective=OptimizationObjective.MINIMIZE_DISTANCE,
            first_solution_strategy="SAVINGS",
            time_limit_sec=60
        )
    """
    # Perfil predefinido (opcional)
    profile: Optional[OptimizationProfile] = None
    
    # Objetivo principal de optimización
    optimization_objective: OptimizationObjective = OptimizationObjective.MINIMIZE_DISTANCE
    
    # Tiempo máximo de optimización en segundos
    time_limit_sec: int = 30
    
    # Estrategia para la solución inicial
    first_solution_strategy: Literal[
        "AUTOMATIC", "PATH_CHEAPEST_ARC", "PATH_MOST_CONSTRAINED_ARC",
        "SAVINGS", "SWEEP", "CHRISTOFIDES", "ALL_UNPERFORMED",
        "BEST_INSERTION", "PARALLEL_CHEAPEST_INSERTION",
        "SEQUENTIAL_CHEAPEST_INSERTION", "LOCAL_CHEAPEST_INSERTION",
        "GLOBAL_CHEAPEST_ARC", "LOCAL_CHEAPEST_ARC",
        "FIRST_UNBOUND_MIN_VALUE"
    ] = "SAVINGS"
    
    # Metaheurística para búsqueda local
    local_search_metaheuristic: Literal[
        "AUTOMATIC", "GREEDY_DESCENT", "GUIDED_LOCAL_SEARCH",
        "SIMULATED_ANNEALING", "TABU_SEARCH"
    ] = "GUIDED_LOCAL_SEARCH"
    
    # Restricciones
    max_route_duration_min: int = 720  # 12 horas por defecto
    allow_dropping_nodes: bool = False  # Si permite no visitar algunos nodos
    
    # Configuración avanzada
    log_search: bool = False
    solution_limit: Optional[int] = None  # Límite de soluciones a explorar
    
    # Campo para descripción (no afecta la optimización)
    description: Optional[str] = None
    
    def model_post_init(self, __context):
        """Aplica la configuración del perfil si se especifica."""
        if self.profile:
            self._apply_profile()
    
    def _apply_profile(self):
        """Aplica la configuración correspondiente al perfil seleccionado."""
        profiles = {
            OptimizationProfile.STANDARD_OPERATIONS: {
                "description": "Configuración equilibrada para operaciones estándar durante el día.",
                "optimization_objective": OptimizationObjective.MINIMIZE_DISTANCE,
                "first_solution_strategy": "PARALLEL_CHEAPEST_INSERTION",
                "local_search_metaheuristic": "GUIDED_LOCAL_SEARCH",
                "time_limit_sec": 30,
                "max_route_duration_min": 480,  # 8 horas
                "allow_dropping_nodes": False,
                "log_search": False,
                "solution_limit": 1
            },
            OptimizationProfile.FAST_DELIVERY: {
                "description": "Prioriza la velocidad de entrega sobre la distancia.",
                "optimization_objective": OptimizationObjective.MINIMIZE_TIME,
                "first_solution_strategy": "PATH_CHEAPEST_ARC",
                "local_search_metaheuristic": "GUIDED_LOCAL_SEARCH",
                "time_limit_sec": 15,
                "max_route_duration_min": 480,  # 8 horas
                "allow_dropping_nodes": False,
                "log_search": False,
                "solution_limit": 1
            },
            OptimizationProfile.COST_EFFICIENT: {
                "description": "Optimizado para minimizar distancias y costos operativos.",
                "optimization_objective": OptimizationObjective.MINIMIZE_DISTANCE,
                "first_solution_strategy": "SAVINGS",
                "local_search_metaheuristic": "GREEDY_DESCENT",
                "time_limit_sec": 60,
                "max_route_duration_min": 600,  # 10 horas
                "allow_dropping_nodes": False,
                "log_search": False,
                "solution_limit": 1
            },
            OptimizationProfile.EXTENDED_OPERATIONS: {
                "description": "Para planificación detallada con tiempo de cómputo extendido.",
                "optimization_objective": OptimizationObjective.MINIMIZE_DISTANCE,
                "first_solution_strategy": "CHRISTOFIDES",
                "local_search_metaheuristic": "SIMULATED_ANNEALING",
                "time_limit_sec": 300,  # 5 minutos
                "max_route_duration_min": 720,  # 12 horas
                "allow_dropping_nodes": False,
                "log_search": True,
                "solution_limit": 5
            },
            OptimizationProfile.RAPID_RESPONSE: {
                "description": "Obtener soluciones rápidas para aplicaciones interactivas.",
                "optimization_objective": OptimizationObjective.MINIMIZE_TIME,
                "first_solution_strategy": "BEST_INSERTION",
                "local_search_metaheuristic": "GREEDY_DESCENT",
                "time_limit_sec": 5,
                "max_route_duration_min": 480,  # 8 horas
                "allow_dropping_nodes": True,
                "log_search": False,
                "solution_limit": 1
            },
            OptimizationProfile.BALANCED: {
                "description": "Equilibrio general entre distancia, tiempo y costos.",
                "optimization_objective": OptimizationObjective.MINIMIZE_COST,
                "first_solution_strategy": "PARALLEL_CHEAPEST_INSERTION",
                "local_search_metaheuristic": "GUIDED_LOCAL_SEARCH",
                "time_limit_sec": 45,
                "max_route_duration_min": 600,  # 10 horas
                "allow_dropping_nodes": False,
                "log_search": False,
                "solution_limit": 3
            }
        }
        
        config = profiles.get(self.profile, {})
        for key, value in config.items():
            setattr(self, key, value)

logger = logging.getLogger(__name__)

class SkillsVehicle(BaseModel):
    """
    Modelo de vehículo para VRP con validación estricta.
    """
    id: int
    vehicle_uuid: Optional[str] = None
    depot_id: int = 0  # ID del depósito al que está asignado el vehículo
    start_lat: float
    start_lon: float
    end_lat: Optional[float] = None
    end_lon: Optional[float] = None
    provided_skills: Optional[List[str]] = Field(default_factory=list)
    capacity_weight: Optional[float] = None
    capacity_volume: Optional[float] = None
    capacity_quantity: Optional[int] = None
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    use_quantity: Optional[bool] = False
    use_weight: Optional[bool] = False
    use_volume: Optional[bool] = False
    plate_number: Optional[str] = None  # Número de placa del vehículo

class SkillsLocation(BaseModel):
    """
    Modelo de ubicación/cliente para VRP con validación estricta y valores por defecto.
    """
    id: int
    is_depot: Optional[bool] = False  # Indica si la ubicación es un depósito
    client_uuid: Optional[str] = None
    name: Optional[str] = None
    lat: float
    lon: float
    demand: Optional[int] = 0
    weight: Optional[float] = 0.0
    volume: Optional[float] = 0.0
    time_window: Optional[List[int]] = Field(default_factory=lambda: DEFAULT_TIME_WINDOW)  # [start, end] en minutos desde medianoche
    service_time: int = DEFAULT_SERVICE_TIME  # Tiempo de servicio en minutos
    required_skills: Optional[List[str]] = Field(default_factory=list)

    @field_validator('lat')
    def validate_lat(cls, v):
        if not -90 <= v <= 90:
            raise ValueError("Latitud debe estar entre -90 y 90")
        return v

    @field_validator('lon')
    def validate_lon(cls, v):
        if not -180 <= v <= 180:
            raise ValueError("Longitud debe estar entre -180 y 180")
        return v

    @field_validator('service_time', 'weight', 'volume', 'demand')
    def non_negative(cls, v):
        if v < 0:
            logger.error("Valor negativo detectado en campo numérico")
            raise ValueError("El valor no puede ser negativo")
        return v

class VRPSkillsRequest(BaseModel):
    """
    Request principal para el endpoint VRP. Incluye validación estricta y valores por defecto centralizados.
    """
    locations: List[SkillsLocation]
    vehicles: List[SkillsVehicle]
    num_vehicles: int = 1
    depot: int = 0
    strict_mode: Optional[bool] = False  # Si True, exige solución "todo o nada"; si False permite solución parcial
    buffer_minutes: Optional[int] = DEFAULT_BUFFER_MINUTES  # Buffer de tráfico en minutos por trayecto
    peak_hours: Optional[list] = Field(default_factory=lambda: DEFAULT_PEAK_HOURS)  # Lista de franjas horarias
    peak_multiplier: Optional[float] = DEFAULT_PEAK_MULTIPLIER
    peak_buffer_minutes: Optional[int] = 20  # Buffer extra en minutos para horas pico
    mode: Optional[str] = DEFAULT_MODE
    units: Optional[str] = DEFAULT_UNITS
    include_polylines: bool = True  # Si False, omite el cálculo de polylines
    detail_level: str = "full"      # "minimal" o "full"; controla nivel de detalle en la respuesta
    
    # Configuración del solver
    solver_options: Optional[SolverOptions] = Field(
        default_factory=SolverOptions,
        description="Opciones avanzadas para el solver de optimización"
    )
    
    # Configuración de la API de matrices
    matrix_config: MatrixApiConfig = Field(
        default_factory=MatrixApiConfig,
        description="Configuración para la API de matriz de distancias"
    )
