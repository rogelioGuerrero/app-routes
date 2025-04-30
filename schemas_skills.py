from pydantic import BaseModel, Field, validator
from typing import List, Optional
import logging
from vrp_constants import (
    DEFAULT_SERVICE_TIME, DEFAULT_BUFFER_MINUTES, DEFAULT_PEAK_HOURS, DEFAULT_PEAK_MULTIPLIER,
    DEFAULT_TIME_WINDOW, DEFAULT_UNITS, DEFAULT_MODE
)

logger = logging.getLogger(__name__)

class SkillsVehicle(BaseModel):
    """
    Modelo de vehículo para VRP con validación estricta.
    """
    id: int
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

class SkillsLocation(BaseModel):
    """
    Modelo de ubicación/cliente para VRP con validación estricta y valores por defecto.
    """
    id: int
    name: Optional[str] = None
    lat: float
    lon: float
    demand: Optional[int] = 0
    weight: Optional[float] = 0.0
    volume: Optional[float] = 0.0
    time_window: Optional[List[int]] = Field(default_factory=lambda: DEFAULT_TIME_WINDOW)  # [start, end] en minutos desde medianoche
    service_time: int = DEFAULT_SERVICE_TIME  # Tiempo de servicio en minutos
    required_skills: Optional[List[str]] = Field(default_factory=list)

    @validator('lat')
    def validate_lat(cls, v):
        if not -90 <= v <= 90:
            raise ValueError("Latitud debe estar entre -90 y 90")
        return v

    @validator('lon')
    def validate_lon(cls, v):
        if not -180 <= v <= 180:
            raise ValueError("Longitud debe estar entre -180 y 180")
        return v

    @validator('service_time', 'weight', 'volume', 'demand')
    def non_negative(cls, v):
        if v < 0:
            logger.error("Valor negativo detectado en campo numérico")
            raise ValueError("El valor no puede ser negativo")
        return v

from typing import Optional

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
