"""Servicio para obtener polilíneas de rutas de OpenRouteService."""
import os
import logging
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class RouteService:
    """Servicio para obtener polilíneas de rutas de OpenRouteService."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Inicializa el servicio con la clave API de ORS."""
        self.api_key = api_key or os.getenv('ORS_API_KEY')
        if not self.api_key:
            raise ValueError("Se requiere una clave API de OpenRouteService")
        self.base_url = "https://api.openrouteservice.org/v2/directions/"
        
    async def get_route_polyline(self, coordinates: List[Tuple[float, float]], 
                               profile: str = 'driving-car') -> Optional[Dict[str, Any]]:
        """Obtiene la polilínea para una ruta entre múltiples puntos.
        
        Args:
            coordinates: Lista de tuplas (longitud, latitud) que representan la ruta.
            profile: Perfil de enrutamiento (ej. 'driving-car', 'foot-walking').
            
        Returns:
            Diccionario con la respuesta de la API de ORS o None en caso de error.
        """
        if len(coordinates) < 2:
            logger.warning("Se requieren al menos 2 puntos para calcular una ruta")
            return None
            
        url = f"{self.base_url}{profile}/geojson"
        headers = {
            'Authorization': self.api_key,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'application/json, application/geo+json'
        }
        
        # Formatear coordenadas para la API de ORS: [[lon1, lat1], [lon2, lat2], ...]
        coords_formatted = [[lon, lat] for lon, lat in coordinates]
        
        # Para rutas con múltiples waypoints, usamos 'optimized_waypoints' para permitir reordenación
        payload = {
            "coordinates": coords_formatted,
            "instructions": "false",
            "geometry": "true",
            "geometry_format": "geojson",
            "elevation": "false",
            "preference": "recommended"
        }
        
        if len(coords_formatted) > 2:
            payload["optimize_waypoints"] = True
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        error_text = await response.text()
                        logger.error(f"Error al obtener la ruta: {response.status} - {error_text}")
                        return None
        except Exception as e:
            logger.error(f"Excepción al obtener la ruta: {str(e)}")
            return None
    
    def get_route_polyline_sync(self, coordinates: List[Tuple[float, float]], 
                              profile: str = 'driving-car') -> Optional[Dict[str, Any]]:
        """Versión síncrona de get_route_polyline."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.get_route_polyline(coordinates, profile))
