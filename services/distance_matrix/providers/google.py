import aiohttp
import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Literal

from ..base import MatrixProvider, MatrixResult

logger = logging.getLogger(__name__)

class GoogleMatrix(MatrixProvider):
    """Implementación de MatrixProvider para Google Maps."""
    
    BASE_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el cliente de Google Maps.
        
        Args:
            api_key: Clave de API de Google Maps. Si no se proporciona,
                   se intentará obtener de la variable de entorno GOOGLE_MAPS_API_KEY.
        """
        self.api_key = api_key or os.getenv('GOOGLE_MAPS_API_KEY')
        if not self.api_key:
            raise ValueError("Se requiere una clave de API de Google Maps")
            
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def get_matrix(
        self,
        locations: List[Dict[str, float]],
        metrics: list[Literal['distances', 'durations']] = None,
        profile: str = 'driving',
        **kwargs
    ) -> MatrixResult:
        """
        Obtiene las matrices de distancias y/o tiempos usando la API de Google Maps.
        
        Args:
            locations: Lista de diccionarios con 'lat' y 'lng'
            metrics: Lista de métricas a obtener ('distances', 'durations' o ambas)
            profile: Modo de transporte ('driving', 'walking', 'bicycling', 'transit')
            **kwargs: Parámetros adicionales para la API de Google
            
        Returns:
            Objeto MatrixResult con las matrices solicitadas
        """
        self.validate_locations(locations)
        
        if not metrics:
            metrics = ['distances', 'durations']
            
        # Validar métricas
        valid_metrics = {'distances', 'durations'}
        invalid_metrics = set(metrics) - valid_metrics
        if invalid_metrics:
            raise ValueError(f"Métricas no soportadas: {invalid_metrics}. Válidas: {valid_metrics}")
        
        # Convertir ubicaciones a formato de cadena para la API
        origins = [f"{loc['lat']},{loc['lng']}" for loc in locations]
        
        # La API de Google tiene un límite de 25 orígenes/destinos por solicitud
        max_batch = 25
        n = len(origins)
        
        # Inicializar matrices con ceros
        distance_matrix = [[0.0] * n for _ in range(n)]
        duration_matrix = [[0.0] * n for _ in range(n)]
        
        try:
            if self.session is None:
                self.session = aiohttp.ClientSession()
            
            # Procesar en lotes
            for i in range(0, n, max_batch):
                batch_origins = origins[i:i + max_batch]
                
                for j in range(0, n, max_batch):
                    batch_destinations = origins[j:j + max_batch]
                    
                    params = {
                        'origins': '|'.join(batch_origins),
                        'destinations': '|'.join(batch_destinations),
                        'mode': profile,
                        'key': self.api_key,
                        **kwargs
                    }
                    
                    # Asegurarse de que se soliciten las métricas necesarias
                    if 'distance' not in params.get('fields', ''):
                        params['fields'] = 'rows/elements/status,rows/elements/distance,rows/elements/duration'
                    
                    async with self.session.get(
                        self.BASE_URL, 
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status != 200:
                            error = await response.text()
                            raise Exception(f"Error de Google Maps ({response.status}): {error}")
                        
                        data = await response.json()
                        
                        if data['status'] != 'OK':
                            raise Exception(f"Error de Google Maps: {data.get('error_message', 'Error desconocido')}")
                        
                        # Procesar la respuesta y llenar las matrices
                        for oi, origin in enumerate(data['rows'], i):
                            for di, element in enumerate(origin['elements'], j):
                                if element['status'] == 'OK':
                                    if 'distances' in metrics and 'distance' in element:
                                        distance_matrix[oi][di] = element['distance']['value']  # en metros
                                    if 'durations' in metrics and 'duration' in element:
                                        duration_matrix[oi][di] = element['duration']['value']  # en segundos
                                else:
                                    # Si hay un error en un par, usar valores muy grandes
                                    if 'distances' in metrics:
                                        distance_matrix[oi][di] = float('inf')
                                    if 'durations' in metrics:
                                        duration_matrix[oi][di] = float('inf')
            
            return MatrixResult(
                distances=distance_matrix if 'distances' in metrics else [],
                durations=duration_matrix if 'durations' in metrics else [],
                provider='google'
            )
            
        except asyncio.TimeoutError:
            raise Exception("Tiempo de espera agotado al conectar con Google Maps")
        except Exception as e:
            logger.error(f"Error en Google Maps: {str(e)}", exc_info=True)
            raise Exception(f"Error al obtener matriz de Google: {str(e)}")
    
    # Mantener compatibilidad con código existente
    async def get_distance_matrix(
        self,
        locations: List[Dict[str, float]],
        **kwargs
    ) -> List[List[float]]:
        """Método de compatibilidad para obtener solo distancias."""
        result = await self.get_matrix(locations, metrics=['distances'], **kwargs)
        return result.distances
        
    async def get_duration_matrix(
        self,
        locations: List[Dict[str, float]],
        **kwargs
    ) -> List[List[float]]:
        """Método de compatibilidad para obtener solo duraciones."""
        result = await self.get_matrix(locations, metrics=['durations'], **kwargs)
        return result.durations
