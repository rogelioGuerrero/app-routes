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

class VRPSkillsRequest(BaseModel):
    locations: List[SkillsLocation]
    vehicles: List[SkillsVehicle]
    num_vehicles: int = 1
    depot: int = 0
    mode: Optional[str] = "driving"
    units: Optional[str] = "metric"
