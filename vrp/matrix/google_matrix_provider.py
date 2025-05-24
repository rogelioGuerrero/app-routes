"""
Proveedor de matrices de distancia y tiempo usando Google Routes API v2.

Esta implementación utiliza el endpoint computeRouteMatrix de la API de Google Routes (v2),
que es la versión más reciente y potente para cálculos de matrices de distancia/tiempo.
"""
import asyncio
import aiohttp
import logging
import time
import json
from typing import List, Dict, Any, Tuple, Optional
from vrp_utils import add_warning

logger = logging.getLogger(__name__)

class GoogleMatrixProvider:
    """
    Proveedor de matrices de distancia y tiempo usando Google Routes API v2.
    
    Documentación oficial: 
    https://developers.google.com/maps/documentation/routes/compute_route_matrix
    
    Límites de la API:
    - Hasta 50 orígenes por solicitud
    - Hasta 50 destinos por solicitud
    - Hasta 2,500 elementos por minuto por proyecto
    - Hasta 100,000 elementos por día por proyecto
    """
    
    def __init__(self, api_key: str):
        """
        Inicializa el proveedor de Google Routes API.
        
        Args:
            api_key: Clave de API de Google Cloud Platform con habilitadas las APIs:
                    - Routes API
                    - Distance Matrix API
        """
        self.api_key = api_key
        self.base_url = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
        self.last_request_time = 0.0
        self.min_request_interval = 0.1  # 10 solicitudes/segundo por defecto
        self.max_origins = 50  # Máx orígenes por solicitud
        self.max_destinations = 50  # Máx destinos por solicitud
        self.max_elements_per_request = 2500  # Límite de elementos por solicitud (50*50)
        self.delay_between_requests = 0.1  # Segundos entre solicitudes
        self.max_elements_per_minute = 3000  # Límite de tasa, no tocar 3000

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
            origins: Lista de diccionarios con 'lat' y 'lon' como claves
            destinations: Lista de diccionarios con 'lat' y 'lon' como claves
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
            "origins": [{"waypoint": {"location": {"latLng": {"latitude": o["lat"], "longitude": o["lon"]}}}}
                     for o in origins],
            "destinations": [{"waypoint": {"location": {"latLng": {"latitude": d["lat"], "longitude": d["lon"]}}}}
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
            logger.debug(f"Enviando solicitud a {self.base_url} con {len(origins)} orígenes y {len(destinations)} destinos")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=body,
                    timeout=timeout
                ) as response:
                    response_text = await response.text()
                    
                    if response.status != 200:
                        error_msg = f"Error en la API (HTTP {response.status}): {response_text[:500]}"
                        logger.error(error_msg)
                        return {
                            "error": error_msg,
                            "status_code": response.status,
                            "details": response_text[:1000]
                        }
                    
                    try:
                        return await response.json()
                    except json.JSONDecodeError as e:
                        error_msg = f"Error al decodificar JSON: {str(e)}\nRespuesta: {response_text[:500]}"
                        logger.error(error_msg)
                        return {"error": error_msg, "status_code": 500}
                    
        except asyncio.TimeoutError:
            error_msg = "Tiempo de espera agotado al conectar con Google Routes API (60s)"
            logger.error(error_msg)
            return {"error": error_msg, "status_code": 408}
            
        except aiohttp.ClientError as e:
            error_msg = f"Error de conexión con Google Routes API: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "status_code": 500}
            
        except Exception as e:
            error_msg = f"Error inesperado: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg, "status_code": 500}

    async def _enforce_rate_limit(self):
        """Asegura que se cumpla el intervalo mínimo entre peticiones."""
        now = time.time()
        elapsed = now - self.last_request_time
        
        if elapsed < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - elapsed)
            
        self.last_request_time = time.time()

    async def get_matrix(
        self,
        request: Any
    ) -> Tuple[List[List[float]], List[List[float]], List[Dict]]:
        """
        Implementación de la interfaz de MatrixProvider usando Routes API v2 (computeRouteMatrix).
        
        Utiliza chunking para manejar grandes conjuntos de datos, dividiendo las solicitudes
        en bloques más pequeños para cumplir con los límites de la API.
        """
        warnings: List[Dict] = []
        locations = getattr(request, 'locations', [])
        n = len(locations)
        if n == 0:
            return [], [], warnings

        # Inicializar matrices de resultado
        dist_matrix = [[0.0] * n for _ in range(n)]
        time_matrix = [[0.0] * n for _ in range(n)]
        
        # Obtener parámetros de la solicitud
        # travel_mode tiene prioridad sobre mode
        mode_raw = getattr(request, 'travel_mode', None) or getattr(request, 'mode', 'DRIVE')
        mode_raw = mode_raw.upper()
        # Mapear a valores válidos
        mode_mapping = {
            'DRIVING': 'DRIVE', 'DRIVE': 'DRIVE',
            'WALKING': 'WALK', 'WALK': 'WALK',
            'BICYCLING': 'BICYCLE', 'BICYCLE': 'BICYCLE',
            'TWO_WHEELER': 'TWO_WHEELER', 'TRANSIT': 'TRANSIT'
        }
        travel_mode = mode_mapping.get(mode_raw, 'DRIVE')
        units = getattr(request, 'units', 'METRIC').upper()
        routing_preference = getattr(request, 'routing_preference', 'TRAFFIC_AWARE_OPTIMAL')
        
        # Preparar ubicaciones para la API
        locations_list = [{"lat": loc.lat, "lon": loc.lon} for loc in locations]
        
        # Calcular chunks basados en límites
        max_origins = min(self.max_origins, n)
        max_destinations = min(self.max_destinations, n)
        
        # Asegurar que no excedamos el máximo de elementos por solicitud
        max_elements = min(self.max_elements_per_request, 625)  # Límite de la API
        
        # Ajustar tamaños de chunk si es necesario
        if max_origins * max_destinations > max_elements:
            # Reducir el tamaño de los chunks para cumplir con el límite
            max_origins = min(max_origins, int((max_elements) ** 0.5))
            max_destinations = min(max_destinations, max_elements // max_origins if max_origins > 0 else max_destinations)
        
        logger.info(f"Procesando matriz {n}x{n} en chunks de máximo {max_origins}x{max_destinations}")
        
        # Contadores para métricas
        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        
        # Procesar por chunks
        for i in range(0, n, max_origins):
            origins_chunk = locations_list[i:i + max_origins]
            chunk_origins_size = len(origins_chunk)
            
            for j in range(0, n, max_destinations):
                destinations_chunk = locations_list[j:j + max_destinations]
                chunk_destinations_size = len(destinations_chunk)
                
                logger.info(f"Procesando chunk: {chunk_origins_size} orígenes (desde #{i}) x {chunk_destinations_size} destinos (desde #{j})")
                
                total_requests += 1
                
                try:
                    # Realizar la petición para este chunk
                    response = await self._make_request(
                        origins=origins_chunk,
                        destinations=destinations_chunk,
                        travel_mode=travel_mode,
                        routing_preference=routing_preference,
                        units=units
                    )
                    
                    # Verificar si hubo un error en la respuesta
                    if 'error' in response:
                        error_msg = response.get('error', 'Error desconocido')
                        status_code = response.get('status_code', 'N/A')
                        logger.error(f"Error en chunk ({i},{j}): {error_msg} (Código: {status_code})")
                        warnings = add_warning(warnings, 'GOOGLE_CHUNK_ERROR', 
                                            f"Error en chunk ({i},{j}): {error_msg}")
                        failed_requests += 1
                        continue
                    
                    # Procesar resultados exitosos
                    successful_requests += 1
                    
                    # La API devuelve una lista de elementos con originIndex y destinationIndex
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
                            global_origin_idx < 0 or global_dest_idx < 0):
                            continue
                        
                        # Estado de la ruta
                        status = item.get('status')
                        if isinstance(status, dict) and status.get('code', 0) != 0:
                            code = status.get('code')
                            warnings = add_warning(warnings, 'GOOGLE_ELEMENT_FAILED', 
                                                f"Elemento ({global_origin_idx},{global_dest_idx}) status {code}")
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
                        
                        # Actualizar matrices (convertir a km y minutos)
                        dist_matrix[global_origin_idx][global_dest_idx] = distance_m / 1000.0  # m → km
                        time_matrix[global_origin_idx][global_dest_idx] = duration_seconds / 60.0  # s → min
                
                except Exception as e:
                    error_msg = str(e)[:500]
                    logger.error(f"Excepción en chunk ({i},{j}): {error_msg}", exc_info=True)
                    warnings = add_warning(warnings, 'GOOGLE_CHUNK_EXCEPTION', 
                                          f"Excepción en chunk ({i},{j}): {error_msg}")
                    failed_requests += 1
                
                # Esperar entre solicitudes para evitar límites de tasa
                if j + max_destinations < n:
                    await asyncio.sleep(self.delay_between_requests)
            
            # Esperar entre filas de chunks
            if i + max_origins < n:
                await asyncio.sleep(self.delay_between_requests * 2)  # Esperar un poco más entre filas
        
        # Estadísticas de la ejecución
        logger.info(f"Solicitudes completadas: {successful_requests}/{total_requests} "
                   f"({(successful_requests/max(1, total_requests))*100:.1f}% de éxito)")
        
        # Verificar si se obtuvieron resultados válidos
        has_valid_data = any(any(val > 0 for val in row) for row in dist_matrix)
        if not has_valid_data:
            error_msg = "No se obtuvieron datos válidos de Google Routes API"
            logger.error(error_msg)
            warnings = add_warning(warnings, 'GOOGLE_NO_VALID_DATA', error_msg)
            # Devolver matrices vacías para indicar error total
            return [], [], warnings
        else:
            logger.info(f"Matriz {n}x{n} generada exitosamente con Google Routes API v2")
        
        # Asegurar que la diagonal principal sea 0 (de un punto a sí mismo)
        for i in range(n):
            dist_matrix[i][i] = 0.0
            time_matrix[i][i] = 0.0

        return dist_matrix, time_matrix, warnings
