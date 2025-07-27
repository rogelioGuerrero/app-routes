"""
Proveedor de matrices con sistema de fallback automático.

Este módulo implementa un proveedor de matrices que intenta múltiples fuentes
en orden de preferencia hasta encontrar una que funcione.
"""
import logging
from .cache import matrix_cache
from typing import List, Dict, Any, Optional, Literal

from .base import MatrixProvider, MatrixResult
from .providers.ors import ORSMatrix
from .providers.google_routes import GoogleRoutesMatrix

logger = logging.getLogger(__name__)

class FallbackMatrixProvider(MatrixProvider):
    """
    Proveedor de matrices con fallback automático:
    1. Primero intenta con ORS (OpenRouteService)
    2. Luego con Google Routes
    3. Finalmente usa distancia euclidiana como último recurso
    """
    
    def __init__(self, cache_enabled: bool = True):
        """
        Inicializa el proveedor con fallback, cargando las claves desde las
        variables de entorno.

        Args:
            cache_enabled: Si se debe usar caché para los resultados.
        """
        import os
        ors_key = os.getenv('ORS_API_KEY')
        google_key = os.getenv('GOOGLE_MAPS_API_KEY')

        self.ors_provider = ORSMatrix(api_key=ors_key) if ors_key else None
        self.google_provider = GoogleRoutesMatrix(api_key=google_key) if google_key else None
        self.cache_enabled = cache_enabled
        self.warnings: List[Dict[str, Any]] = []
        
        if not any([self.ors_provider, self.google_provider]):
            logger.warning("No se proporcionaron claves de API. Solo se usará distancia euclidiana.")
            
    def _add_warning(self, code: str, message: str, **kwargs) -> Dict[str, Any]:
        """
        Agrega una advertencia a la lista de advertencias.
        
        Args:
            code: Código de la advertencia
            message: Mensaje descriptivo
            **kwargs: Metadatos adicionales
            
        Returns:
            Diccionario con la advertencia generada
        """
        warning = {"code": code, "message": message, **kwargs}
        self.warnings.append(warning)
        logger.warning(f"[{code}] {message}")
        return warning
    
    async def _get_euclidean_matrix(
        self, 
        locations: List[Dict[str, float]],
        metrics: list[Literal['distances', 'durations']] = None
    ) -> MatrixResult:
        """
        Calcula la matriz de distancias usando distancia euclidiana.
        
        Args:
            locations: Lista de ubicaciones con 'lat' y 'lng'
            metrics: Lista de métricas a calcular
            
        Returns:
            MatrixResult con las matrices calculadas
            
        Nota:
            - Las distancias se devuelven en metros
            - Las duraciones se devuelven en segundos (asumiendo 50 km/h)
        """
        if not metrics:
            metrics = ['distances', 'durations']
            
        n = len(locations)
        distances = [[0.0] * n for _ in range(n)] if 'distances' in metrics else []
        durations = [[0.0] * n for _ in range(n)] if 'durations' in metrics else []
        
        try:
            # Si solo hay una ubicación, devolver matrices vacías
            if n <= 1:
                return MatrixResult(distances, durations, "euclidean")
                
            # Calcular matriz de distancias
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                        
                    try:
                        # Coordenadas en grados decimales
                        lat1, lng1 = locations[i]["lat"], locations[i]["lng"]
                        lat2, lng2 = locations[j]["lat"], locations[j]["lng"]
                        
                        # Distancia euclidiana aproximada (km)
                        # 1 grado ≈ 111.32 km (variación de latitud)
                        # Ajuste por longitud (cos(lat_media) * 111.32)
                        avg_lat = (lat1 + lat2) / 2
                        lat_km = 111.32
                        lng_km = abs(111.32 * (0.9984 + 0.0016 * avg_lat / 90))
                        
                        dlat = (lat1 - lat2) * lat_km
                        dlng = (lng1 - lng2) * lng_km
                        dist_km = (dlat**2 + dlng**2)**0.5
                        
                        if 'distances' in metrics:
                            distances[i][j] = dist_km * 1000  # a metros
                            
                        if 'durations' in metrics:
                            # Tiempo estimado a 50 km/h (segundos)
                            speed_kmh = 50
                            durations[i][j] = (dist_km / speed_kmh) * 3600
                            
                    except (KeyError, TypeError, ValueError) as e:
                        logger.error(f"Error calculando distancia euclidiana: {e}")
                        if 'distances' in metrics:
                            distances[i][j] = float('inf')
                        if 'durations' in metrics:
                            durations[i][j] = float('inf')
            
            self._add_warning(
                "EUCLIDEAN_FALLBACK",
                "Se usó distancia euclidiana como último recurso. Las distancias son aproximadas."
            )
            return MatrixResult(
                distances=distances if 'distances' in metrics else [],
                durations=durations if 'durations' in metrics else [],
                provider="euclidean"
            )
            
        except Exception as e:
            error_msg = f"Error al calcular matriz euclidiana: {str(e)}"
            logger.critical(error_msg, exc_info=True)
            raise Exception(error_msg)
    
    async def get_matrix(
        self,
        locations: List[Dict[str, float]],
        metrics: list[Literal['distances', 'durations']] = None,
        profile: str = 'driving',
        **kwargs
    ) -> MatrixResult:
        """
        Obtiene las matrices de distancia/tiempo con fallback automático.
        
        Args:
            locations: Lista de ubicaciones con 'lat' y 'lng'
            metrics: Lista de métricas a obtener ('distances', 'durations' o ambas)
            profile: Perfil de ruta (driving, walking, etc.)
            **kwargs: Parámetros adicionales para los proveedores
            
        Returns:
            MatrixResult con las matrices solicitadas
            
        Raises:
            ValueError: Si no hay suficientes ubicaciones o son inválidas
            Exception: Si todos los proveedores fallan
        """
        if not metrics:
            metrics = ['distances', 'durations']
            
        # Validar ubicaciones
        if not locations:
            raise ValueError("No se proporcionaron ubicaciones")
            
        if len(locations) < 2:
            # Para una sola ubicación, devolver matrices vacías
            n = len(locations)
            return MatrixResult(
                distances=[[0.0] * n for _ in range(n)] if 'distances' in metrics else [],
                durations=[[0.0] * n for _ in range(n)] if 'durations' in metrics else [],
                provider="none"
            )
        
        # 0. Intentar devolver desde caché -------------------------------
        if self.cache_enabled:
            cached = matrix_cache.get(locations, profile)
            if cached:
                self._add_warning(
                    "CACHE_HIT",
                    "Se devolvió matriz desde caché",
                    provider=cached.provider
                )
                return cached

        # 1. Intentar con ORS si está disponible
        if self.ors_provider:
            try:
                logger.info("Intentando obtener matriz con ORS...")
                # Mapear perfiles genéricos a los que entiende ORS
                ors_profile = profile
                if profile == 'driving':
                    ors_profile = 'driving-car'
                elif profile == 'truck':
                    ors_profile = 'driving-hgv'
                result = await self.ors_provider.get_matrix(
                    locations=locations,
                    metrics=metrics,
                    profile=ors_profile,
                    **kwargs
                )
                logger.info("✓ Matriz generada exitosamente con ORS")
                if self.cache_enabled:
                    matrix_cache.set(locations, profile, result)
                return result
                
            except Exception as e:
                self._add_warning(
                    "ORS_FALLBACK",
                    f"No se pudo usar ORS: {str(e)[:200]}",
                    error_type=type(e).__name__
                )
                logger.warning(f"Fallo con ORS: {str(e)}")
        
        # 2. Intentar con Google Routes si está disponible
        if self.google_provider:
            try:
                logger.info("Intentando obtener matriz con Google Routes...")
                result = await self.google_provider.get_matrix(
                    locations=locations,
                    metrics=metrics,
                    profile=profile,
                    **kwargs
                )
                self._add_warning(
                    "GOOGLE_FALLBACK_USED",
                    "Se usó Google Routes como fallback de ORS"
                )
                logger.info("✓ Matriz generada exitosamente con Google Routes")
                if self.cache_enabled:
                    matrix_cache.set(locations, profile, result)
                return result
                
            except Exception as e:
                self._add_warning(
                    "GOOGLE_FALLBACK_FAILED",
                    f"No se pudo usar Google Routes: {str(e)[:200]}",
                    error_type=type(e).__name__
                )
                logger.warning(f"Fallo con Google Routes: {str(e)}")
        
        # 3. Usar distancia euclidiana como último recurso
        logger.warning("Usando distancia euclidiana como último recurso")
        result = await self._get_euclidean_matrix(locations, metrics)
        if self.cache_enabled:
            matrix_cache.set(locations, profile, result)
        return result
    
    def get_warnings(self) -> List[Dict[str, Any]]:
        """
        Obtiene la lista de advertencias acumuladas.
        
        Returns:
            Lista de diccionarios con información de advertencias
        """
        return self.warnings.copy()  # Devolver una copia para evitar modificaciones externas


# Ejemplo de uso
async def example_usage():
    """Ejemplo de uso del FallbackMatrixProvider."""
    import os
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Obtener claves de API de variables de entorno
    ors_key = os.getenv('ORS_API_KEY')
    google_key = os.getenv('GOOGLE_MAPS_API_KEY')
    
    # Crear proveedor con fallback
    provider = FallbackMatrixProvider(
        ors_api_key=ors_key,
        google_api_key=google_key,
        cache_enabled=True
    )
    
    # Ubicaciones de ejemplo (CDMX, Guadalajara, Monterrey)
    locations = [
        {"lat": 19.4326, "lng": -99.1332},  # CDMX
        {"lat": 20.6597, "lng": -103.3496},  # Guadalajara
        {"lat": 25.6866, "lng": -100.3161}   # Monterrey
    ]
    
    try:
        # Obtener matriz de distancias y duraciones
        result = await provider.get_matrix(
            locations=locations,
            metrics=['distances', 'durations'],
            profile='driving'
        )
        
        # Mostrar resultados
        print(f"Proveedor usado: {result.provider}")
        print(f"Matriz de distancias (metros): {result.distances}")
        print(f"Matriz de duraciones (segundos): {result.durations}")
        
        # Mostrar advertencias si las hay
        for warning in provider.get_warnings():
            print(f"Advertencia [{warning['code']}]: {warning['message']}")
            
    except Exception as e:
        print(f"Error al obtener la matriz: {str(e)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
