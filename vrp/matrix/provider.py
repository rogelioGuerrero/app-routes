import logging
import traceback
from types import SimpleNamespace
from typing import List, Tuple, Dict, Any, Optional

from vrp.matrix.ors import ORSMatrixProvider
from vrp.matrix.google_matrix_provider import GoogleMatrixProvider
from vrp_utils import add_warning

logger = logging.getLogger(__name__)

class MatrixProviderError(Exception):
    """Excepción personalizada para errores del proveedor de matrices"""
    def __init__(self, message: str, provider: str, original_error: Exception = None):
        self.provider = provider
        self.original_error = original_error
        full_message = f"[{provider}] {message}"
        if original_error:
            full_message += f"\nError original: {str(original_error)}"
        super().__init__(full_message)

class MatrixProvider:
    """
    Proveedor unificado de matrices de distancia/tiempo:
    - Primero intenta con ORS
    - Luego Google Distance Matrix
    - Finalmente fallback euclidiano
    """
    def __init__(self, ors_api_key: str, google_api_key: Optional[str] = None):
        self.ors_provider = ORSMatrixProvider(ors_api_key)
        self.google_api_key = google_api_key

    async def get_matrix(
        self,
        request: Any
    ) -> Tuple[List[List[float]], List[List[float]], List[Dict[str, Any]]]:
        """
        Obtiene las matrices de distancia y tiempo, con fallback a Google y luego a distancia euclidiana.
        
        Args:
            request: Objeto con las ubicaciones y parámetros de la solicitud
            
        Returns:
            Tupla con (distance_matrix, time_matrix, warnings)
            
        Raises:
            MatrixProviderError: Si no se pueden generar las matrices con ningún método
        """
        locations = request.locations
        warnings: List[Dict[str, Any]] = []
        
        if not locations:
            logger.warning("No se proporcionaron ubicaciones para generar la matriz")
            return [], [], warnings

        # 1. Intento con ORS
        try:
            logger.info("[1/3] Intentando obtener matriz con ORS...")
            # Call ORS provider with the same request object containing locations, units, profile, etc.
            dist_matrix, time_matrix, ors_warnings = await self.ors_provider.get_matrix(request)
            warnings.extend(ors_warnings)
            
            if not dist_matrix or not time_matrix:
                logger.warning("ORS devolvió matrices vacías")
                raise ValueError("ORS devolvió matrices vacías")
                
            logger.info("✓ Matriz generada exitosamente con ORS")
            return dist_matrix, time_matrix, warnings
            
        except Exception as e:
            error_msg = f"No se pudo obtener la matriz con ORS: {str(e)}"
            logger.warning(error_msg, exc_info=True)
            warnings = add_warning(
                warnings, 
                'ORS_FALLBACK', 
                f"No se pudo usar ORS: {str(e)[:150]}"
            )

        # 2. Fallback a Google Distance Matrix
        if self.google_api_key:
            try:
                logger.info("[2/3] Intentando obtener matriz con Google...")
                provider = GoogleMatrixProvider(self.google_api_key)
                # Crear un objeto request compatible con GoogleMatrixProvider
                from types import SimpleNamespace
                google_request = SimpleNamespace(
                    locations=request.locations,
                    mode=getattr(request, 'mode', 'driving'),
                    units=getattr(request, 'units', 'metric'),
                    routing_preference=getattr(request, 'routing_preference', 'TRAFFIC_AWARE')
                )
                dist_matrix, time_matrix, google_warnings = await provider.get_matrix(google_request)
                warnings.extend(google_warnings)
                
                if not dist_matrix or not time_matrix:
                    logger.warning("Google devolvió matrices vacías")
                    raise ValueError("Google devolvió matrices vacías")
                
                logger.info("✓ Matriz generada exitosamente con Google")
                warnings = add_warning(
                    warnings, 
                    'GOOGLE_FALLBACK_USED', 
                    'Se usó Google Distance Matrix como fallback de ORS'
                )
                return dist_matrix, time_matrix, warnings
                
            except Exception as e:
                error_msg = f"No se pudo obtener la matriz con Google: {str(e)}"
                logger.warning(error_msg, exc_info=True)
                warnings = add_warning(
                    warnings, 
                    'GOOGLE_FALLBACK_FAILED', 
                    f"No se pudo usar Google: {str(e)[:150]}"
                )
        else:
            logger.warning("No se proporcionó GOOGLE_API_KEY, omitiendo Google Distance Matrix")

        # 3. Fallback euclidiano
        logger.warning("[3/3] Usando distancia euclidiana como último recurso")
        try:
            n = len(locations)
            if n == 0:
                logger.warning("No hay ubicaciones para calcular matriz euclidiana")
                return [], [], warnings
                
            logger.debug(f"Calculando matriz euclidiana para {n} ubicaciones...")
            dist_matrix = [[0.0] * n for _ in range(n)]
            time_matrix = [[0.0] * n for _ in range(n)]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    try:
                        lat1, lon1 = locations[i].lat, locations[i].lon
                        lat2, lon2 = locations[j].lat, locations[j].lon
                        # Distancia euclidiana aproximada (km)
                        dist = ((lat1 - lat2)**2 + (lon1 - lon2)**2)**0.5 * 111.32
                        dist_matrix[i][j] = dist
                        # Tiempo estimado a 30 km/h (min)
                        time_matrix[i][j] = dist / 30 * 60
                    except Exception as e:
                        logger.error(f"Error calculando distancia euclidiana entre {i} y {j}: {e}")
                        dist_matrix[i][j] = float('inf')
                        time_matrix[i][j] = float('inf')
            
            logger.info("✓ Matriz euclidiana generada exitosamente")
            warnings = add_warning(
                warnings,
                'EUCLIDEAN_FALLBACK',
                'Se usó distancia euclidiana como último recurso. Las distancias son aproximadas.'
            )
            return dist_matrix, time_matrix, warnings
            
        except Exception as e:
            error_msg = f"Error fatal al generar matriz euclidiana: {str(e)}"
            logger.critical(error_msg, exc_info=True)
            raise MatrixProviderError(
                message=error_msg,
                provider="Euclidean",
                original_error=e
            )
            
        # Si llegamos aquí, todos los métodos fallaron
        error_msg = "Todos los métodos para generar la matriz han fallado"
        logger.critical(error_msg)
        raise MatrixProviderError(
            message=error_msg,
            provider="All",
            original_error=None
        )
