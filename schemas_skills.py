from pydantic import BaseModel, Field
from typing import List, Optional

class SkillsVehicle(BaseModel):
    id: int
    start_lat: float
    start_lon: float
    end_lat: Optional[float] = None
    end_lon: Optional[float] = None
    provided_skills: Optional[List[str]] = []
    capacity_weight: Optional[float] = None
    capacity_volume: Optional[float] = None
    capacity_quantity: Optional[int] = None
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    use_quantity: Optional[bool] = False
    use_weight: Optional[bool] = False
    use_volume: Optional[bool] = False

class SkillsLocation(BaseModel):
    id: int
    name: Optional[str] = None
    lat: float
    lon: float
    demand: Optional[int] = 0
    weight: Optional[float] = 0.0
    volume: Optional[float] = 0.0
    time_window: Optional[List[int]] = None  # [start, end] en minutos desde medianoche
    service_time: int = 5  # Tiempo de servicio en minutos (default=5)
    required_skills: Optional[List[str]] = []

from typing import Optional

class VRPSkillsRequest(BaseModel):
    locations: List[SkillsLocation]
    vehicles: List[SkillsVehicle]
    num_vehicles: int = 1
    depot: int = 0
    strict_mode: Optional[bool] = False  # Si True, exige solución "todo o nada"; si False permite solución parcial
    buffer_minutes: Optional[int] = 10  # Buffer de tráfico en minutos por trayecto (default 10)
    peak_hours: Optional[list] = None  # Lista de franjas horarias [['07:00','09:00'], ...]
    peak_buffer_minutes: Optional[int] = 20  # Buffer extra en minutos para horas pico (default 20)

    mode: Optional[str] = "driving"
    units: Optional[str] = "metric"
    include_polylines: bool = True  # Si False, omite el cálculo de polylines
    detail_level: str = "full"      # "minimal" o "full"; controla nivel de detalle en la respuesta
