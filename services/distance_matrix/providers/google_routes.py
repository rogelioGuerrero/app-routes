"""
Proveedor de matrices de distancia/tiempo usando Google Routes API v2.

Esta implementación utiliza el endpoint computeRouteMatrix de la API de Google Routes (v2),
que es la versión más reciente y potente para cálculos de matrices de distancia/tiempo.
"""
import os
import asyncio
import aiohttp
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Literal

from ..base import MatrixProvider, MatrixResult

logger = logging.getLogger(__name__)

class GoogleRoutesMatrix(MatrixProvider):
    """
    Implementación de MatrixProvider para Google Routes API v2.
    
    Documentación oficial: 
    https://developers.google.com/maps/documentation/routes/compute_route_matrix
    
    Límites de la API:
    - Hasta 50 orígenes por solicitud
    - Hasta 50 destinos por solicitud
    - Hasta 2,500 elementos por minuto por proyecto
    - Hasta 100,000 elementos por día por proyecto
    """
    
    BASE_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el proveedor de Google Routes API.
        
        Args:
            api_key: Clave de API de Google Cloud Platform con habilitadas las APIs:
                   - Routes API
                   - Distance Matrix API
                   Si no se proporciona, se intentará obtener de la variable de entorno GOOGLE_MAPS_API_KEY.
        """
        self.api_key = api_key or os.getenv('GOOGLE_MAPS_API_KEY')
        if not self.api_key:
            raise ValueError("Se requiere una clave de API de Google Maps. "
                          "Configure la variable de entorno GOOGLE_MAPS_API_KEY o pase la clave directamente.")
        
        self.session = None
        self.last_request_time = 0.0
        self.min_request_interval = 0.1  # 10 solicitudes/segundo por defecto
        self.max_origins = 50  # Máx orígenes por solicitud
        self.max_destinations = 50  # Máx destinos por solicitud
        self.delay_between_requests = 0.1  # Segundos entre solicitudes
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _make_request(
        self,
        origins: List[Dict[str, float]],
        destinations: Optional[List[Dict[str, float]]] = None,
        travel_mode: str = 'DRIVE',
        routing_preference: str = 'TRAFFIC_AWARE_OPTIMAL',
        units: str = 'METRIC'
    ) -> Dict[str, Any]:
        """
        Realiza una petición a la API de Google Routes v2 computeRouteMatrix.
        
        Args:
            origins: Lista de diccionarios con 'lat' y 'lng' como claves
            destinations: Lista de diccionarios con 'lat' y 'lng' como claves
            travel_mode: Modo de viaje (DRIVE, WALK, BICYCLE, TWO_WHEELER, TRANSIT)
            routing_preference: Preferencia de ruteo (TRAFFIC_AWARE_OPTIMAL, TRAFFIC_AWARE, TRAFFIC_UNAWARE)
            units: Unidades de medida (METRIC o IMPERIAL)
            
        Returns:
            Respuesta de la API como diccionario
            
        Raises:
            Exception: Si hay un error en la solicitud o respuesta
        """
        # Rate limiting
        await self._enforce_rate_limit()
        
        # Si no se especifican destinos, usamos los orígenes (matriz completa)
        if destinations is None:
            destinations = origins
        
        # Validar parámetros
        if travel_mode not in ['DRIVE', 'WALK', 'BICYCLE', 'TWO_WHEELER', 'TRANSIT']:
            travel_mode = 'DRIVE'
            
        if routing_preference not in ['TRAFFIC_AWARE_OPTIMAL', 'TRAFFIC_AWARE', 'TRAFFIC_UNAWARE']:
            routing_preference = 'TRAFFIC_AWARE_OPTIMAL'
            
        # Construir el cuerpo de la solicitud
        body = {
            "origins": [{"waypoint": {"location": {"latLng": {"latitude": o["lat"], "longitude": o["lng"]}}}}
                     for o in origins],
            "destinations": [{"waypoint": {"location": {"latLng": {"latitude": d["lat"], "longitude": d["lng"]}}}}
                           for d in destinations],
            "travelMode": travel_mode,
            "routingPreference": routing_preference,
            "units": units,
            "languageCode": "es-MX"
        }
        
        # Configurar headers para autenticación
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': self.api_key,
            'X-Goog-FieldMask': 'originIndex,destinationIndex,duration,distanceMeters,status,condition',
        }
        
        # Realizar la petición POST
        timeout = aiohttp.ClientTimeout(total=60)  # 60 segundos de timeout
        
        try:
            logger.debug(f"Enviando solicitud a {self.BASE_URL} con {len(origins)} orígenes y {len(destinations)} destinos")
            
            async with self.session.post(
                self.BASE_URL,
                headers=headers,
                json=body,
                timeout=timeout
            ) as response:
                response_text = await response.text()
                
                if response.status != 200:
                    error_msg = f"Error en la API (HTTP {response.status}): {response_text[:500]}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
                try:
                    return await response.json()
                except Exception as e:
                    error_msg = f"Error al decodificar JSON: {str(e)}\nRespuesta: {response_text[:500]}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
        except asyncio.TimeoutError:
            error_msg = "Tiempo de espera agotado al conectar con Google Routes API (60s)"
            logger.error(error_msg)
            raise Exception(error_msg)
                
        except aiohttp.ClientError as e:
            error_msg = f"Error de conexión con Google Routes API: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
                
        except Exception as e:
            error_msg = f"Error inesperado: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)

    async def _enforce_rate_limit(self):
        """Asegura que se cumpla el intervalo mínimo entre peticiones."""
        now = time.time()
        elapsed = now - self.last_request_time
        
        if elapsed < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - elapsed)
            
        self.last_request_time = time.time()

    async def get_matrix(
        self,
        locations: List[Dict[str, float]],
        metrics: list[Literal['distances', 'durations']] = None,
        profile: str = 'driving',
        **kwargs
    ) -> MatrixResult:
        """
        Obtiene las matrices de distancias y/o tiempos usando Google Routes API v2.
        
        Esta implementación maneja automáticamente matrices grandes dividiéndolas
        en chunks más pequeños que cumplan con los límites de la API.
        
        Args:
            locations: Lista de diccionarios con 'lat' y 'lng'
            metrics: Lista de métricas a obtener ('distances', 'durations' o ambas)
            profile: Perfil de ruta (ej: 'driving', 'walking', 'bicycling', 'transit')
            **kwargs: Parámetros adicionales:
                     - routing_preference: 'TRAFFIC_AWARE_OPTIMAL' (predeterminado), 'TRAFFIC_AWARE', 'TRAFFIC_UNAWARE'
                     - units: 'METRIC' (predeterminado, metros/segundos) o 'IMPERIAL' (millas/horas)
                     
        Returns:
            Objeto MatrixResult con las matrices solicitadas
            
        Raises:
            Exception: Si ocurre un error en la petición
        """
        if not metrics:
            metrics = ['distances', 'durations']
        
        # Validar métricas
        valid_metrics = {'distances', 'durations'}
        invalid_metrics = set(metrics) - valid_metrics
        if invalid_metrics:
            raise ValueError(f"Métricas no soportadas: {invalid_metrics}. Válidas: {valid_metrics}")
        
        # Inicializar sesión si es necesario
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        n = len(locations)
        if n == 0:
            return MatrixResult(distances=[], durations=[], provider='google_routes')
        
        # Mapear perfiles de Google
        profile_mapping = {
            'driving': 'DRIVE',
            'walking': 'WALK',
            'bicycling': 'BICYCLE',
            'transit': 'TRANSIT'
        }
        
        travel_mode = profile_mapping.get(profile.lower(), 'DRIVE')
        routing_preference = kwargs.get('routing_preference', 'TRAFFIC_AWARE_OPTIMAL')
        units = kwargs.get('units', 'METRIC')
        
        # Inicializar matrices de resultado (en metros y segundos)
        distance_matrix = [[0.0] * n for _ in range(n)] if 'distances' in metrics else []
        duration_matrix = [[0.0] * n for _ in range(n)] if 'durations' in metrics else []
        
        # Preparar ubicaciones para la API
        locations_list = [{"lat": loc['lat'], "lng": loc['lng']} for loc in locations]
        
        # Calcular chunks basados en límites
        max_origins = min(self.max_origins, n)
        max_destinations = min(self.max_destinations, n)
        
        # Procesar por chunks
        try:
            for i in range(0, n, max_origins):
                origins_chunk = locations_list[i:i + max_origins]
                chunk_origins_size = len(origins_chunk)
                
                for j in range(0, n, max_destinations):
                    destinations_chunk = locations_list[j:j + max_destinations]
                    chunk_destinations_size = len(destinations_chunk)
                    
                    logger.debug(f"Procesando chunk: {chunk_origins_size} orígenes (desde #{i}) x {chunk_destinations_size} destinos (desde #{j})")
                    
                    # Realizar la petición para este chunk
                    response = await self._make_request(
                        origins=origins_chunk,
                        destinations=destinations_chunk,
                        travel_mode=travel_mode,
                        routing_preference=routing_preference,
                        units=units
                    )
                    
                    # Procesar resultados exitosos
                    for item in response:
                        origin_idx = item.get('originIndex')
                        dest_idx = item.get('destinationIndex')
                        
                        if origin_idx is None or dest_idx is None:
                            continue
                            
                        # Convertir a índices globales
                        global_origin_idx = i + origin_idx
                        global_dest_idx = j + dest_idx
                        
                        # Verificar límites
                        if (global_origin_idx >= n or global_dest_idx >= n or
                            global_origin_idx < 0 or global_est_idx < 0):
                            continue
                        
                        # Estado de la ruta
                        status = item.get('status', {})
                        if isinstance(status, dict) and status.get('code', 0) != 0:
                            logger.warning(f"Ruta no encontrada de {global_origin_idx} a {global_dest_idx}: {status}")
                            continue
                        
                        # Extraer distancia y duración
                        distance_m = item.get('distanceMeters', 0)
                        duration_seconds = item.get('duration')
                        
                        # Si duration viene como string (ej: "600s")
                        if isinstance(duration_seconds, str) and duration_seconds.endswith('s'):
                            try:
                                duration_seconds = int(duration_seconds[:-1])
                            except (ValueError, TypeError):
                                duration_seconds = 0
                        
                        # Actualizar matrices
                        if 'distances' in metrics:
                            distance_matrix[global_origin_idx][global_dest_idx] = distance_m  # en metros
                        if 'durations' in metrics:
                            duration_matrix[global_origin_idx][global_dest_idx] = duration_seconds  # en segundos
                
                    # Esperar entre solicitudes para evitar límites de tasa
                    if j + max_destinations < n:
                        await asyncio.sleep(self.delay_between_requests)
                
                # Esperar entre filas de chunks
                if i + max_origins < n:
                    await asyncio.sleep(self.delay_between_requests * 2)
            
            # Asegurar que la diagonal principal sea 0 (de un punto a sí mismo)
            for i in range(n):
                if 'distances' in metrics:
                    distance_matrix[i][i] = 0.0
                if 'durations' in metrics:
                    duration_matrix[i][i] = 0.0
            
            return MatrixResult(
                distances=distance_matrix if 'distances' in metrics else [],
                durations=duration_matrix if 'durations' in metrics else [],
                provider='google_routes'
            )
            
        except Exception as e:
            error_msg = f"Error al obtener matriz de Google Routes: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)
