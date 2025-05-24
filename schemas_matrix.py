from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from enum import Enum

class MatrixProvider(str, Enum):
    """Proveedores de matriz de distancias soportados."""
    ORS = "ors"  # OpenRouteService
    GOOGLE = "google"
    OSRM = "osrm"

class MatrixApiConfig(BaseModel):
    """Configuración para la API de matriz de distancias.
    
    Atributos:
        provider: Proveedor del servicio de matriz de distancias.
        api_key: Clave API para autenticación (opcional según el proveedor).
        base_url: URL base del servicio (opcional, se usan URLs por defecto).
        profile: Perfil de ruta (driving, walking, cycling, etc.).
        metrics: Métricas a calcular (distance, duration, etc.).
        units: Unidades de medida (km/mi).
        optimize: Si se debe optimizar la consulta para múltiples orígenes/destinos.
        max_locations: Número máximo de ubicaciones por solicitud.
        rate_limit: Límite de solicitudes por minuto.
        timeout_seconds: Tiempo máximo de espera para la respuesta.
    """
    provider: MatrixProvider = MatrixProvider.ORS
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    profile: str = "driving-car"
    metrics: List[str] = ["distance", "duration"]
    units: str = "km"
    optimize: bool = True
    max_locations: int = 50
    rate_limit: int = 40
    timeout_seconds: int = 30

    @field_validator('metrics')
    def validate_metrics(cls, v):
        valid_metrics = ["distance", "duration", "duration_factor", "weight", "route"]
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(f"Métrica no válida: {metric}. Debe ser uno de {valid_metrics}")
        return v

    @field_validator('units')
    def validate_units(cls, v):
        if v not in ["km", "mi"]:
            raise ValueError("Las unidades deben ser 'km' o 'mi'")
        return v

    @field_validator('profile')
    def validate_profile(cls, v, values):
        provider = values.data.get('provider', 'ors')
        
        if provider == MatrixProvider.ORS:
            valid_profiles = [
                "driving-car", "driving-hgv", "foot-walking", "foot-hiking",
                "cycling-regular", "cycling-road", "cycling-mountain", "cycling-tour"
            ]
        elif provider == MatrixProvider.GOOGLE:
            valid_profiles = ["driving", "walking", "bicycling", "transit"]
        else:  # OSRM
            valid_profiles = ["car", "bike", "foot"]

        if v not in valid_profiles:
            raise ValueError(f"Perfil no válido para {provider}. Debe ser uno de {valid_profiles}")
        return v
