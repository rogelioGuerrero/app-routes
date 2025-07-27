"""Módulo para el manejo de matrices de distancia con múltiples proveedores."""

from typing import Optional, Dict, Type, Any, List, Tuple, Literal, TYPE_CHECKING, Union
import os

# Type alias para type hints
if TYPE_CHECKING:
    from .base import MatrixProvider, MatrixResult
    from .providers.ors import ORSMatrix
    from .providers.google import GoogleMatrix
    MatrixProviderType = MatrixProvider
else:
    MatrixProviderType = 'MatrixProvider'  # type: ignore

# Importaciones diferidas para evitar dependencias circulares
ORSMatrix = None
GoogleMatrix = None

class DistanceMatrixFactory:
    """Fábrica para crear instancias de proveedores de matriz de distancia."""
    
    # Inicializado en _init_providers para manejar importaciones diferidas
    _providers_initialized = False
    _providers: Dict[str, Type[MatrixProviderType]] = {}
    
    @classmethod
    def _init_providers(cls) -> None:
        if not cls._providers_initialized:
            # Importaciones diferidas para evitar dependencias circulares
            from .providers.ors import ORSMatrix
            from .providers.google import GoogleMatrix
            
            cls._providers = {
                'ors': ORSMatrix,
                'google': GoogleMatrix,
            }
            cls._providers_initialized = True
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, Type[MatrixProviderType]]:
        """
        Devuelve un diccionario de proveedores disponibles.
        
        Returns:
            Dict[str, Type[MatrixProvider]]: Mapeo de nombres de proveedores a sus clases
        """
        cls._init_providers()
        return cls._providers.copy()
    
    @classmethod
    def create_provider(
        cls,
        provider_name: str,
        api_key: Optional[str] = None,
        **kwargs: Any
    ) -> MatrixProviderType:
        """
        Crea una instancia del proveedor de matriz de distancia especificado.
        
        Args:
            provider_name: Nombre del proveedor ('ors' o 'google')
            api_key: Clave de API opcional. Si no se proporciona, se intentará obtener
                   de las variables de entorno.
            **kwargs: Argumentos adicionales para el constructor del proveedor
            
        Returns:
            Instancia de MatrixProvider
            
        Raises:
            ValueError: Si el proveedor no es soportado o falta la clave de API
            RuntimeError: Si hay un error al crear el proveedor
        """
        # Asegurarse de que los proveedores estén inicializados
        cls._init_providers()
        
        # Obtener la clase del proveedor
        provider_class = cls._providers.get(provider_name.lower())
        if not provider_class:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Proveedor no soportado: {provider_name}. "
                f"Opciones: {available}"
            )
        
        # Si no se proporciona una clave, intentar obtenerla de las variables de entorno
        if not api_key:
            env_vars = {
                'ors': 'ORS_API_KEY',
                'google': 'GOOGLE_MAPS_API_KEY'
            }
            env_var = env_vars.get(provider_name.lower())
            api_key = os.getenv(env_var) if env_var else None
            
            if not env_var or not api_key:
                error_msg = (
                    f"Se requiere una clave de API para {provider_name}. "
                    f"Proporciónela o establezca la variable de entorno {env_var}"
                ) if env_var else f"Se requiere una clave de API para {provider_name}"
                raise ValueError(error_msg)
        
        try:
            # Importación local para evitar dependencias circulares
            from .base import MatrixProvider
            return provider_class(api_key=api_key, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Error al crear el proveedor {provider_name}: {str(e)}"
            ) from e
    
    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any]
    ) -> MatrixProviderType:
        """
        Crea un proveedor de matriz de distancia a partir de una configuración.
        
        Args:
            config: Diccionario con la configuración. Debe contener al menos 'provider'.
                  
        Returns:
            Instancia de MatrixProvider configurada
            
        Raises:
            ValueError: Si la configuración no es válida
            
        Example:
            config = {
                'provider': 'ors',
                'api_key': 'tu_api_key',
                'profile': 'driving-car'
            }
            provider = DistanceMatrixFactory.from_config(config)
        """
        if not isinstance(config, dict):
            raise ValueError("La configuración debe ser un diccionario")
            
        # Hacer una copia para no modificar el diccionario original
        config = config.copy()
        provider_name = config.pop('provider', None)
        if not provider_name:
            raise ValueError("La configuración debe incluir un 'provider'")
            
        # Extraer api_key del diccionario si está presente
        api_key = config.pop('api_key', None)
        
        return cls.create_provider(provider_name, api_key=api_key, **config)
