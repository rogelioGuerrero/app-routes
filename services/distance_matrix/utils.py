"""Utilidades para trabajar con matrices de distancia."""

from typing import List, Dict, Any, Optional, Tuple, Literal
import asyncio
import logging
import math
from .base import MatrixResult

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importación local para evitar dependencia circular
def _get_distance_matrix_factory():
    from . import DistanceMatrixFactory
    return DistanceMatrixFactory

async def get_euclidean_distance_matrix(locations: List[Dict[str, float]], **kwargs) -> List[List[float]]:
    """
    Calcula la matriz de distancias euclidianas entre ubicaciones.
    
    Args:
        locations: Lista de diccionarios con 'lat' y 'lng'
        **kwargs: Argumentos adicionales (ignorados)
        
    Returns:
        Matriz de distancias en metros
    """
    n = len(locations)
    distances = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                distances[i][j] = 0.0
            else:
                # Fórmula de Haversine para distancia entre dos puntos en la Tierra
                lat1, lng1 = math.radians(locations[i]['lat']), math.radians(locations[i]['lng'])
                lat2, lng2 = math.radians(locations[j]['lat']), math.radians(locations[j]['lng'])
                
                # Diferencia de latitud y longitud
                dlat = lat2 - lat1
                dlng = lng2 - lng1
                
                # Fórmula de Haversine
                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                
                # Radio de la Tierra en metros (promedio)
                R = 6371000  # metros
                distances[i][j] = R * c
    
    return distances

async def get_matrix_with_fallback(
    locations: List[Dict[str, float]],
    metrics: list[Literal['distances', 'durations']] = None,
    providers: Optional[List[Tuple[str, dict]]] = None,
    **kwargs
) -> Tuple[MatrixResult, str]:
    """
    Obtiene las matrices de distancias y/o tiempos intentando múltiples proveedores en secuencia.
    
    Orden de intento por defecto:
    1. OpenRouteService (ORS)
    2. Google Maps
    3. Distancia euclidiana (solo distancias)
    
    Args:
        locations: Lista de ubicaciones con 'lat' y 'lng'
        metrics: Lista de métricas a obtener ('distances' y/o 'durations')
        providers: Lista de proveedores a intentar con sus configuraciones
        **kwargs: Argumentos adicionales para los proveedores
        
    Returns:
        Tupla con (resultado, nombre_del_proveedor_usado)
        
    Example:
        locations = [
            {'lat': 40.7128, 'lng': -74.0060},  # NYC
            {'lat': 34.0522, 'lng': -118.2437},  # LA
        ]
        
        # Obtener solo distancias
        result, provider = await get_matrix_with_fallback(
            locations,
            metrics=['distances']
        )
        
        # Obtener distancias y duraciones con configuración personalizada
        providers = [
            ('ors', {'profile': 'driving-car'}),
            ('google', {'mode': 'driving'}),
            ('euclidean', {})  # Solo para distancias
        ]
        result, provider = await get_matrix_with_fallback(
            locations,
            metrics=['distances', 'durations'],
            providers=providers
        )
    """
    DistanceMatrixFactory = _get_distance_matrix_factory()
    
    if not metrics:
        metrics = ['distances', 'durations']
        
    if providers is None:
        providers = [
            ('ors', {'profile': 'driving-car'}),
            ('google', {'mode': 'driving'}),
            ('euclidean', {}) if 'distances' in metrics else ()
        ]
    else:
        # Filtrar 'euclidean' si no se solicitan distancias
        providers = [p for p in providers if p[0] != 'euclidean' or 'distances' in metrics]
    
    last_error = None
    
    for provider_name, provider_config in providers:
        try:
            if provider_name == 'euclidean':
                if 'durations' in metrics and len(metrics) == 1:
                    logger.warning("No se pueden obtener duraciones con el método euclidiano. Usando tiempos estimados basados en distancia.")
                
                logger.info("Usando distancia euclidiana (modo offline)")
                distances = await get_euclidean_distance_matrix(locations)
                
                # Para la duración, asumimos una velocidad promedio (ej: 50 km/h)
                durations = [[d / (50/3.6) for d in row] for row in distances] if 'durations' in metrics else []
                
                return MatrixResult(
                    distances=distances if 'distances' in metrics else [],
                    durations=durations if 'durations' in metrics else [],
                    provider='euclidean'
                ), 'euclidean'
                
            logger.info(f"Intentando obtener matriz con {provider_name}...")
            
            # Crear instancia del proveedor (sin pasar parámetros específicos del método)
            provider = DistanceMatrixFactory.create_provider(
                provider_name=provider_name,
                api_key=provider_config.get('api_key')
            )
            
            # Obtener las matrices (pasar parámetros específicos del método aquí)
            async with provider as p:
                result = await p.get_matrix(
                    locations=locations,
                    metrics=metrics,
                    **{k: v for k, v in {**kwargs, **provider_config}.items() 
                       if k not in ['api_key']}  # Excluir parámetros del constructor
                )
            
            logger.info(f"Matrices obtenidas exitosamente con {provider_name}")
            return result, provider_name
            
        except Exception as e:
            last_error = e
            logger.warning(
                f"Error con {provider_name}: {str(e)}\n"
                f"Tipo de error: {type(e).__name__}\n"
                "Intentando con el siguiente proveedor...",
                exc_info=True
            )
            continue
    
    # Si llegamos aquí, todos los proveedores fallaron
    error_msg = "No se pudo obtener la matriz de ningún proveedor"
    if last_error:
        error_msg += f": {str(last_error)}"
    raise RuntimeError(error_msg) from last_error


# Alias para compatibilidad hacia atrás
async def get_distance_matrix_with_fallback(
    locations: List[Dict[str, float]],
    providers: Optional[List[Tuple[str, dict]]] = None,
    **kwargs
) -> Tuple[List[List[float]], str]:
    """
    Versión antigua que solo devuelve distancias.
    Se mantiene para compatibilidad hacia atrás.
    """
    result, provider = await get_matrix_with_fallback(
        locations=locations,
        metrics=['distances'],
        providers=providers,
        **kwargs
    )
    return result.distances, provider

# Alias para compatibilidad hacia atrás
async def get_distance_matrix(
    locations: List[Dict[str, float]],
    provider: str = 'ors',
    api_key: Optional[str] = None,
    **kwargs
) -> List[List[float]]:
    DistanceMatrixFactory = _get_distance_matrix_factory()
    """
    Obtiene la matriz de distancias usando un proveedor específico.
    
    Args:
        locations: Lista de ubicaciones con 'lat' y 'lng'
        provider: Proveedor a utilizar ('ors', 'google' o 'euclidean')
        api_key: Clave de API opcional
        **kwargs: Argumentos adicionales para el proveedor
        
    Returns:
        Matriz de distancias en metros
    """
    if provider == 'euclidean':
        return await get_euclidean_distance_matrix(locations)
        
    provider_instance = DistanceMatrixFactory.create_provider(
        provider_name=provider,
        api_key=api_key,
        **kwargs
    )
    
    return await provider_instance.get_distance_matrix(
        locations=locations,
        **kwargs
    )

def calculate_euclidean_distance(
    loc1: Dict[str, float], 
    loc2: Dict[str, float],
    scale_factor: float = 1000  # Para convertir a metros
) -> float:
    """
    Calcula la distancia euclidiana entre dos puntos geográficos.
    
    Args:
        loc1: Primer punto con 'lat' y 'lng'
        loc2: Segundo punto con 'lat' y 'lng'
        scale_factor: Factor para escalar la distancia (por defecto 1000 para metros)
        
    Returns:
        Distancia aproximada en metros
    """
    from math import sqrt, cos, radians
    
    # Radio de la Tierra en kilómetros
    R = 6371.0
    
    lat1, lon1 = radians(loc1['lat']), radians(loc1['lng'])
    lat2, lon2 = radians(loc2['lat']), radians(loc2['lng'])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Fórmula de Haversine
    a = (sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Distancia en kilómetros * scale_factor
    return (R * c) * scale_factor

async def get_euclidean_distance_matrix(locations: List[Dict[str, float]]) -> List[List[float]]:
    """
    Calcula la matriz de distancias euclidianas entre ubicaciones.
    
    Args:
        locations: Lista de ubicaciones con 'lat' y 'lng'
        
    Returns:
        Matriz de distancias en metros
        
    Nota:
        Utiliza la fórmula de Haversine para calcular distancias geodésicas.
    """
    from math import radians, sin, cos, sqrt, atan2
    
    # Radio de la Tierra en metros
    R = 6371000
    
    n = len(locations)
    matrix = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        lat1 = radians(locations[i]['lat'])
        lng1 = radians(locations[i]['lng'])
        
        for j in range(n):
            if i == j:
                matrix[i][j] = 0.0
                continue
                
            lat2 = radians(locations[j]['lat'])
            lng2 = radians(locations[j]['lng'])
            
            # Fórmula de Haversine
            dlat = lat2 - lat1
            dlng = lng2 - lng1
            
            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlng / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            
            matrix[i][j] = R * c
    
    return matrix
