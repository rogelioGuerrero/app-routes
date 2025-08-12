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

async def get_euclidean_distance_matrix(
    locations: List[Dict[str, float]],
    distance_factor: float = 1.5,
    avg_speed_kmh: float = 15.0,
    **kwargs
) -> List[List[float]]:
    """
    Calcula una matriz de distancias en metros usando fórmula de Haversine con un
    factor de corrección opcional para aproximar rutas reales.
    
    Args:
        locations: Lista de diccionarios con 'lat' y 'lng'
        distance_factor: Factor para ajustar la distancia en línea recta a distancia real (default: 1.5)
        avg_speed_kmh: (No usado aquí) Se mantiene por compatibilidad.
        **kwargs: Argumentos adicionales para compatibilidad
        
    Returns:
        Matriz de distancias (metros)
    """
    n = len(locations)
    distances = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                distances[i][j] = 0.0
                continue
            # Haversine
            lat1 = math.radians(locations[i]['lat'])
            lng1 = math.radians(locations[i]['lng'])
            lat2 = math.radians(locations[j]['lat'])
            lng2 = math.radians(locations[j]['lng'])
            dlat = lat2 - lat1
            dlng = lng2 - lng1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            R = 6371000  # metros
            distances[i][j] = (R * c) * float(distance_factor)
    
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
                
                # Para la duración, asumimos una velocidad promedio de 15 km/h para zonas urbanas
                durations = [[d / (15/3.6) for d in row] for row in distances] if 'durations' in metrics else []
                
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
                       if k not in ['api_key', 'session']}  # Excluir parámetros del constructor y 'session'
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
    Calcula la distancia euclidiana entre dos puntos geográficos usando la fórmula de Haversine.
    
    Args:
        loc1: Primer punto con 'lat' y 'lng'
        loc2: Segundo punto con 'lat' y 'lng'
        scale_factor: Factor para escalar la distancia (por defecto 1000 para metros)
        
    Returns:
        Distancia aproximada en metros
        
    Nota:
        - Usa la fórmula de Haversine para distancias geodésicas
        - Considera la curvatura terrestre
        - Devuelve distancia en metros por defecto
    """
    # Radio de la Tierra en kilómetros
    R = 6371.0
    
    # Convertir grados a radianes
    lat1 = math.radians(loc1['lat'])
    lon1 = math.radians(loc1['lng'])
    lat2 = math.radians(loc2['lat'])
    lon2 = math.radians(loc2['lng'])
    
    # Diferencias
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Fórmula de Haversine
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distancia en metros (multiplicar por 1000 para convertir de km a m)
    distance = R * c * 1000
    
    # Aplicar factor de escala si es necesario
    return distance * (scale_factor / 1000)

 
