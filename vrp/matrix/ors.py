import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import requests
from vrp_utils import add_warning

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ORSMatrixProvider:
    """
    Proveedor de matrices de distancia/tiempo usando OpenRouteService (ORS).
    
    Esta implementación es simple y directa, diseñada para ser usada con el VRP.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # URL base incluye profile por defecto
        self.base_url = "https://api.openrouteservice.org/v2/matrix/driving-car"
        self.headers = {
            'Authorization': f"{self.api_key}",
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8'
        }
        self.last_request_time = 0
        self.min_request_interval = 1.5  # segundos entre peticiones

    async def _make_request(
        self,
        coords: List[List[float]],
        units: str = 'km',
        sources: Optional[List[int]] = None,
        destinations: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Realiza una petición a la API de ORS Matrix."""
        # Rate limiting simple
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        body = {
            "locations": coords,
            "metrics": ["distance", "duration"],
            "units": units
        }
        
        if sources is not None:
            body["sources"] = sources
        if destinations is not None:
            body["destinations"] = destinations
        
        try:
            self.last_request_time = time.time()
            response = requests.post(
                self.base_url,
                json=body,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error en petición ORS: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg += f" - {error_data}"
                except:
                    error_msg += f" - {e.response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    async def get_matrix(
        self,
        request: Any
    ) -> Tuple[List[List[float]], List[List[float]], List[Dict]]:
        """
        Genera matrices de distancia (km) y tiempo (min).
        
        Args:
            request: Objeto con atributos locations, units y profile
            
        Returns:
            Tupla con (distance_matrix, time_matrix, warnings)
        """
        # Parámetros del request
        locations = getattr(request, 'locations', [])
        n = len(locations)
        if n == 0:
            return [], [], []
        # Selección de unidades (km por defecto, o 'mi' si imperial)
        units_raw = getattr(request, 'units', 'metric').lower()
        units = 'mi' if units_raw == 'imperial' else 'km'
        # Selección de profile para URL
        profile = getattr(request, 'profile', None)
        base_url = f"https://api.openrouteservice.org/v2/matrix/{profile}" if profile else self.base_url
        coords = [[loc.lon, loc.lat] for loc in locations]
        warnings = []
        
        # Inicializar matrices de resultado
        distance_matrix = [[0.0] * n for _ in range(n)]
        time_matrix = [[0.0] * n for _ in range(n)]
        
        # Tamaño máximo de chunk (ajustado a 55 para estar seguros)
        MAX_CHUNK = 55
        
        # Si el tamaño es manejable en una sola petición
        if n <= MAX_CHUNK:
            data = await self._make_request(coords, units=units)
            for i in range(n):
                for j in range(n):
                    distance_matrix[i][j] = data['distances'][i][j]  # km
                    time_matrix[i][j] = data['durations'][i][j] / 60  # s -> min
            return distance_matrix, time_matrix, warnings
        
        # Usar chunking con orígenes y destinos separados
        num_chunks = (n + MAX_CHUNK - 1) // MAX_CHUNK
        total_requests = num_chunks * num_chunks
        add_warning(
            warnings, 
            'ORS_CHUNKING_USED', 
            f'Usando chunking para matriz {n}x{n}: {num_chunks}x{num_chunks} chunks ({total_requests} peticiones)'
        )
        
        # Procesar en chunks
        for i in range(0, n, MAX_CHUNK):
            sources = list(range(i, min(i + MAX_CHUNK, n)))
            
            for j in range(0, n, MAX_CHUNK):
                dests = list(range(j, min(j + MAX_CHUNK, n)))
                
                # Si hay algún error en un chunk, fallar completamente
                data = await self._make_request(coords, units=units, sources=sources, destinations=dests)
                
                # Procesar resultados del chunk
                for si, gi in enumerate(sources):
                    for dj, gj in enumerate(dests):
                        distance_matrix[gi][gj] = data['distances'][si][dj]  # km
                        time_matrix[gi][gj] = data['durations'][si][dj] / 60  # s -> min
        
        return distance_matrix, time_matrix, warnings