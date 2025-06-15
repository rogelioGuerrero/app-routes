import logging
import asyncio # Added for potential async operations if provider is async
from typing import List, Any, Dict, Tuple

from services.distance_matrix.cache import MatrixCache
from services.distance_matrix.providers.ors import ORSMatrix
from services.distance_matrix.providers.google import GoogleMatrix
from services.distance_matrix.fallback_provider import FallbackMatrixProvider
from services.distance_matrix.base import MatrixResult # Assuming MatrixResult is here

logger = logging.getLogger(__name__)

class MatrixAdapter:
    """
    Actúa como una fachada (Facade) para obtener matrices de costos (distancia/tiempo)
    de diferentes proveedores, con una capa de caché integrada.

    Esta clase simplifica el acceso a los servicios de matrices, ocultando la
    complejidad de seleccionar un proveedor, manejar errores o fallbacks, y gestionar
    la caché de resultados para evitar llamadas repetidas a las APIs externas.
    """

    def __init__(self):
        """Inicializa el adaptador y la caché de matrices."""
        self.matrix_cache = MatrixCache()
        self._provider_factory = {
            'ors': ORSMatrix,
            'google': GoogleMatrix
        }

    async def get_matrix(
        self,
        coordinates: List[Dict[str, float]], # Changed type to List[Dict[str, float]]
        profile: str = 'driving-car',
        force_refresh: bool = False
    ) -> MatrixResult: # Changed return type to MatrixResult
        """
        Obtiene una matriz de distancias para un conjunto de coordenadas.

        Gestiona la caché y la selección de proveedores.

        Args:
            coordinates: Una lista de coordenadas en formato [lon, lat].
            profile: El perfil de enrutamiento a utilizar (ej: 'driving-car').
            force_refresh: Si es True, ignora la caché y solicita nuevos datos.

        Returns:
            Una matriz de distancias en metros.
        """
        logger.info(f"MatrixAdapter: Solicitando matriz para {len(coordinates)} ubicaciones con perfil '{profile}'.")
        if not force_refresh:
            # MatrixCache.get ahora toma 'coordinates' y 'profile' para generar la clave internamente.
            cached_result: Optional[MatrixResult] = self.matrix_cache.get(coordinates, profile)
            logger.debug(f"MatrixAdapter: Checked cache for profile '{profile}'. Found: {cached_result is not None}")
            if cached_result:
                logger.info(f"MatrixAdapter: MatrixResult obtenido de la caché para perfil '{profile}'.")
                # Asegurarse de que los metadatos de 'from_cache' estén presentes si se recuperó de la caché
                if not hasattr(cached_result, 'metadata') or cached_result.metadata is None:
                    cached_result.metadata = {}
                if isinstance(cached_result.metadata, dict):
                    cached_result.metadata['from_cache'] = True # Marcar explícitamente
                return cached_result
        else:
            logger.info("Refresco forzado, se ignorará la caché existente para esta solicitud.")

        logger.info("MatrixAdapter: MatrixResult no encontrado en caché o se forzó refresco. Contactando al proveedor.")
        try:
            # Lógica de selección de proveedor (aquí se podría extender)
            provider_name = 'ors'  # Por ahora, usamos ORS como principal
            logger.debug(f"MatrixAdapter: Attempting to get provider: {provider_name}")
            provider_name = 'ors'  # Por ahora, usamos ORS como principal
            logger.debug(f"MatrixAdapter: Attempting to get provider: {provider_name}")
            # ORSMatrix (y potencialmente otros proveedores) implementan __aenter__ y __aexit__
            async with self._get_provider(provider_name) as provider:
                logger.debug(f"MatrixAdapter: Provider '{provider_name}' obtained and context entered: {provider is not None}")
                logger.debug(f"MatrixAdapter: Calling provider.get_matrix for profile '{profile}' with {len(coordinates)} coordinates.")
                matrix_result: Optional[MatrixResult] = await provider.get_matrix(coordinates, profile=profile)
                logger.debug(f"MatrixAdapter: Provider.get_matrix returned. Result is None: {matrix_result is None}. Result type: {type(matrix_result)}")
                
                if matrix_result and matrix_result.distances: # Check if distances are present
                    logger.info(f"MatrixAdapter: MatrixResult obtenido de {provider_name}. Guardando en caché.")
                    self.matrix_cache.set(coordinates, profile, matrix_result)
                    return matrix_result
                else:
                    logger.warning(f"MatrixAdapter: El proveedor ({provider_name}) no devolvió datos o matriz de distancias vacía. Usando fallback.")
                    # El fallback no debería ocurrir dentro del 'async with' si el proveedor mismo es el fallback
                    # pero aquí el fallback es una alternativa si el proveedor principal falla o devuelve vacío.
                    return self._get_fallback_matrix_result(len(coordinates), profile)
            # El bloque 'async with' asegura que provider.__aexit__ se llame, cerrando la sesión de aiohttp.

        except Exception as e:
            logger.error(
                f"MatrixAdapter: Error al obtener MatrixResult del proveedor: {e}. Usando fallback.",
                exc_info=True
            )
            return self._get_fallback_matrix_result(len(coordinates), profile)

    def _get_provider(self, name: str, **kwargs: Any) -> Any:
        """Fábrica simple para instanciar proveedores de matrices."""
        provider_class = self._provider_factory.get(name.lower())
        if not provider_class:
            raise ValueError(f"Proveedor '{name}' no soportado.")
        return provider_class(**kwargs)

    def _get_fallback_matrix_result(self, num_locations: int, profile: str) -> MatrixResult:
        """Invoca al proveedor de matrices de respaldo y devuelve un MatrixResult."""
        logger.warning(f"MatrixAdapter: Generando un MatrixResult de fallback para {num_locations} ubicaciones, perfil '{profile}'.")
        fallback_provider = FallbackMatrixProvider()
        # El fallback provider devuelve (distance_matrix, duration_matrix)
        # Asumimos que son matrices llenas de 0.0 o valores de error si no se pueden calcular
        distance_matrix, duration_matrix = fallback_provider.get_matrices(num_locations)
        logger.debug(f"MatrixAdapter: Fallback provider returned distances: {distance_matrix is not None}, durations: {duration_matrix is not None}")
        return MatrixResult(
            distances=distance_matrix if distance_matrix is not None else [[0.0]*num_locations for _ in range(num_locations)],
            durations=duration_matrix if duration_matrix is not None else [[0.0]*num_locations for _ in range(num_locations)],
            provider='fallback'
        )
