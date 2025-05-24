"""
Servicios para el módulo VRP.

Este módulo contiene la lógica de negocio para el cálculo de rutas VRP,
separada de los endpoints de la API para mejor mantenibilidad y pruebas.
"""
import os
from dotenv import load_dotenv
load_dotenv()
import time
import logging
import uuid
from typing import Dict, List, Tuple, Optional, Any
from fastapi import HTTPException, status
from types import SimpleNamespace
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
from vrp.matrix.provider import MatrixProvider, MatrixProviderError
from vrp_utils import validate_request_coords, add_warning

# Configuración de logging
logger = logging.getLogger(__name__)

class VRPError(Exception):
    """Excepción personalizada para errores del VRP."""
    def __init__(self, message: str, status_code: int = 500, details: Any = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

def create_matrix_provider() -> MatrixProvider:
    """
    Crea y retorna una instancia de MatrixProvider con las credenciales configuradas.
    
    Returns:
        MatrixProvider: Instancia configurada del proveedor de matrices
    """
    # Crear MatrixProvider con claves de entorno
    ors_key = os.getenv('ORS_API_KEY', '')
    google_key = os.getenv('GOOGLE_API_KEY', '')
    return MatrixProvider(ors_key, google_key)

async def validate_vrp_request(request) -> Tuple[bool, Dict]:
    """
    Valida la solicitud VRP.
    
    Args:
        request: Objeto de solicitud VRP
        
    Returns:
        Tuple[bool, Dict]: (is_valid, error_response)
    """
    # Validación de ubicaciones
    if not hasattr(request, 'locations') or not request.locations:
        return False, {
            "message": "Se requiere al menos una ubicación",
            "status_code": status.HTTP_400_BAD_REQUEST
        }
        
    # Validación de vehículos
    if not hasattr(request, 'vehicles') or not request.vehicles:
        return False, {
            "message": "Se requiere al menos un vehículo",
            "status_code": status.HTTP_400_BAD_REQUEST
        }
        
    return True, {}

async def generate_distance_matrices(
    request,
    request_id: str,
    warnings: List[Dict[str, Any]]
) -> Tuple[Optional[List[List[float]]], Optional[List[List[float]]], List[Dict[str, Any]]]:
    """
    Genera las matrices de distancia y tiempo.
    
    Args:
        request: Objeto de solicitud VRP con atributos:
                - locations: Lista de ubicaciones con lat/lon
                - mode: Modo de viaje (opcional, predeterminado: 'driving')
        request_id: Identificador único de la solicitud
        warnings: Lista para almacenar advertencias
        
    Returns:
        Tuple con (distance_matrix, time_matrix, updated_warnings)
    """
    logger.info(f"[{request_id}] Iniciando generación de matrices...")
    start_time = time.time()
    
    try:
        provider = create_matrix_provider()
        logger.info(f"[{request_id}] Usando proveedor: {provider.__class__.__name__}")
        
        # Combinar configuración de matriz y parámetros específicos para el proveedor
        params = request.matrix_config.model_dump(exclude_none=True)
        # Sobrescribir con params específicos por proveedor
        if hasattr(request, 'google_params') and request.google_params:
            params.update(request.google_params)
        if hasattr(request, 'ors_params') and request.ors_params:
            params.update(request.ors_params)
            
        # Construir request dinámico
        provider_request = SimpleNamespace(
            locations=request.locations,
            **params
        )
        
        # Obtener matrices del proveedor
        logger.info(f"[{request_id}] Solicitando matrices al proveedor...")
        dist_matrix, time_matrix, matrix_warnings = await provider.get_matrix(provider_request)
        
        warnings.extend(matrix_warnings)
        
        if not dist_matrix or not time_matrix:
            raise VRPError("Las matrices generadas están vacías")
            
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] ✓ Matrices generadas en {elapsed:.2f}s")
        logger.debug(f"[{request_id}] Tamaño matriz: {len(dist_matrix)}x{len(dist_matrix[0]) if dist_matrix else 0}")
        
        return dist_matrix, time_matrix, warnings
        
    except MatrixProviderError as e:
        logger.warning(f"[{request_id}] Falló {e.provider}, intentando con el siguiente proveedor...")
        if hasattr(e, 'original_error'):
            logger.debug(f"[{request_id}] Error detallado: {str(e.original_error)}")
            
        error_msg = f"Error en el proveedor de matrices ({e.provider}): {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")
        
        # Agregar advertencia sobre el fallo
        add_warning(warnings, 
                   code=f"{e.provider.upper()}_FAILED",
                   message=f"Fallo en {e.provider}. {str(e)}",
                   context={"provider": e.provider})
        
        # Relanzar para que el manejador de errores lo procese
        raise VRPError(
            message=error_msg,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"provider": e.provider, "original_error": str(e.original_error) if hasattr(e, 'original_error') else str(e)}
        )
    except Exception as e:
        error_msg = f"Error inesperado al generar matrices: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}", exc_info=True)
        raise VRPError(
            message=error_msg,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error_type": e.__class__.__name__, "error": str(e)}
        )

def build_vrp_response(
    dist_matrix: Optional[List[List[float]]] = None,
    time_matrix: Optional[List[List[float]]] = None,
    request_id: Optional[str] = None,
    start_time: Optional[float] = None,
    warnings: Optional[List[Dict[str, Any]]] = None,
    num_locations: Optional[int] = None,
    **kwargs
) -> Dict:
    """
    Construye la respuesta del servicio VRP.
    
    Esta función es compatible con versiones anteriores pero permite parámetros opcionales.
    
    Args:
        dist_matrix: Matriz de distancias (opcional)
        time_matrix: Matriz de tiempos (opcional)
        request_id: Identificador único de la solicitud (opcional)
        start_time: Tiempo de inicio de la solicitud (opcional)
        warnings: Lista de advertencias (opcional)
        num_locations: Número de ubicaciones (opcional)
        **kwargs: Argumentos adicionales para compatibilidad con versiones futuras
        
    Returns:
        Dict: Respuesta del servicio VRP
    """
    # Manejar parámetros opcionales
    processing_time = None
    if start_time is not None:
        processing_time = round(time.time() - start_time, 4)
    
    # Construir la respuesta básica
    response = {
        "solution": {},
        "metadata": {
            "status": "success"
        },
        "warnings": warnings or []
    }
    
    # Agregar matrices si están presentes
    if dist_matrix is not None:
        response["solution"]["distance_matrix"] = dist_matrix
    if time_matrix is not None:
        response["solution"]["time_matrix"] = time_matrix
    
    # Agregar metadatos si están presentes
    if request_id is not None:
        response["metadata"]["request_id"] = request_id
    if processing_time is not None:
        response["metadata"]["processing_time_seconds"] = processing_time
    if num_locations is not None:
        response["metadata"]["num_locations"] = num_locations
    
    # Agregar cualquier otro metadato adicional
    if kwargs:
        response["metadata"].update(kwargs)
    
    return response
