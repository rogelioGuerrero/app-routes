"""Módulo de caché para matrices de distancia.

Este módulo proporciona un sistema de caché para almacenar y recuperar
matrices de distancia y duración, optimizado para entornos serverless como Render.
"""

import hashlib
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any, Tuple

from .base import MatrixResult # Added for MatrixCache.set type hint

# Configuración de logging
logger = logging.getLogger(__name__)

# Intenta importar diskcache, si falla usaremos un caché en memoria
try:
    from diskcache import Cache
    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False
    logger.warning("diskcache no está instalado. Usando caché en memoria.")


class MatrixCache:
    """Implementación de caché para matrices de distancia.
    
    Usa diskcache si está disponible, de lo contrario usa un diccionario en memoria.
    """
    _instance = None
    # Configuración optimizada para Render
    CACHE_DIR = "/tmp/vrp_matrix_cache"  # En Render, /tmp es temporal pero persistente durante la vida del servicio
    CACHE_SIZE_LIMIT = 50 * 1024 * 1024  # 50MB - Ajustado para el plan gratuito de Render
    CACHE_DEFAULT_TIMEOUT = 21600  # 6 horas en segundos - Balance entre rendimiento y actualización

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MatrixCache, cls).__new__(cls)
            cls._instance._initialize_cache()
        return cls._instance

    def _initialize_cache(self):
        """Inicializa el caché en disco o en memoria."""
        self._memory_only = not HAS_DISKCACHE
        
        if not self._memory_only:
            try:
                # Crear directorio si no existe
                os.makedirs(self.CACHE_DIR, exist_ok=True)
                self.cache = Cache(
                    directory=self.CACHE_DIR,
                    size_limit=self.CACHE_SIZE_LIMIT,
                    eviction_policy='least-recently-used',
                    disk_min_file_size=1024,
                    disk_pickle_protocol=4
                )
                logger.debug(f"Caché en disco inicializado en {self.CACHE_DIR}")
            except Exception as e:
                logger.warning(f"No se pudo inicializar caché en disco: {e}. Usando memoria.")
                self._memory_only = True
        
        if self._memory_only:
            self.cache = {}
            logger.debug("Usando caché en memoria")
        self.enabled = True # Cache system is now considered enabled

    # La siguiente definición de _generate_key es correcta y se mantiene.
    # La definición incorrecta de clear @staticmethod ha sido eliminada.

    @staticmethod
    def _generate_key(locations: List[Dict[str, Any]], profile: str) -> str:
        logger.debug(f"MatrixCache._generate_key: INICIO - llamado con locations (tipo: {type(locations)}, es lista: {isinstance(locations, list)}, longitud si lista: {len(locations) if isinstance(locations, list) else 'N/A'})")
        if isinstance(locations, list) and len(locations) > 0:
            logger.debug(f"MatrixCache._generate_key: Primer elemento de locations (tipo: {type(locations[0])}): {str(locations[0])[:200]}") # Log primer elemento
        """Genera una clave única basada en las ubicaciones.
        
        Args:
            locations: Lista de diccionarios con 'lat' y 'lng'
            
        Returns:
            str: Clave única para el caché
        """
        # Ordenar ubicaciones para consistencia
        # Loguear el tipo de cada 'loc' antes de procesar para depurar
        if isinstance(locations, list) and len(locations) > 0:
            for i, loc_item in enumerate(locations[:3]): # Loguear los primeros 3 items
                logger.debug(f"MatrixCache._generate_key: Procesando loc_item #{i} (tipo: {type(loc_item)}): {str(loc_item)[:200]}")

        sorted_locations = sorted(
            [(loc.get('lat'), loc.get('lng')) for loc in locations],
            key=lambda x: (x[0], x[1])
        )
        # Incluir el perfil en la clave para asegurar unicidad
        key_data = json.dumps((sorted_locations, profile), sort_keys=True).encode()
        return f"matrix_{hashlib.sha256(key_data).hexdigest()}"

    def get(self, locations: List[Dict[str, Any]], profile: str) -> Optional[MatrixResult]:
        logger.debug(f"MatrixCache.get: INICIO - locations (len: {len(locations) if locations else 'None'}), profile: {profile}")
        if not self.enabled:
            logger.debug("MatrixCache.get: Cache no habilitado.")
            return None
        try:
            key = MatrixCache._generate_key(locations, profile)
            logger.debug(f"MatrixCache.get: Clave generada: {key}")
            
            # Obtener datos del caché (diskcache o dict)
            cached_value = self.cache.get(key)

            if cached_value is None:
                logger.debug(f"MatrixCache.get: Cache miss para clave: {key}")
                return None

            if isinstance(cached_value, MatrixResult):
                logger.debug(f"MatrixCache.get: Cache hit! Devolviendo MatrixResult para clave: {key}")
                # Actualizar timestamp de 'from_cache' si es necesario o añadirlo si no existe
                if hasattr(cached_value, 'metadata') and isinstance(cached_value.metadata, dict):
                    cached_value.metadata['from_cache'] = True
                    cached_value.metadata['cache_retrieved_at'] = time.time()
                elif not hasattr(cached_value, 'metadata') or cached_value.metadata is None:
                    # Si metadata no existe o es None, inicializarlo
                    cached_value.metadata = {'from_cache': True, 'cache_retrieved_at': time.time()}
                return cached_value
            else:
                logger.warning(f"MatrixCache.get: Cache hit pero el valor no es MatrixResult (tipo: {type(cached_value)}). Tratando como miss. Clave: {key}")
                # Opcionalmente, eliminar la entrada inválida: self.cache.delete(key)
                return None
        except Exception as e:
            logger.error(f"MatrixCache.get: Error al leer del caché. Error: {e}", exc_info=True)
            return None

    def set(self, locations: List[Dict[str, Any]], profile: str, value: MatrixResult) -> None:
        """Guarda un valor (MatrixResult) en el caché usando una clave pre-generada."""
        if not self.enabled:
            logger.debug("MatrixCache.set: Cache no habilitado, no se guarda nada.")
            return
        if value is None:
            logger.warning("MatrixCache.set: Se intentó guardar un valor None. No se guarda nada.")
            return
        if not locations:
            logger.warning("MatrixCache.set: Se intentó guardar con 'locations' vacías. No se guarda nada.")
            return

        try:
            key = MatrixCache._generate_key(locations, profile)
            logger.debug(f"MatrixCache.set: Intentando guardar en caché con clave (generada de locations y profile): {key}")
            
            # Añadir metadatos de caché al objeto antes de guardarlo, si no existen
            if not hasattr(value, 'metadata') or value.metadata is None:
                value.metadata = {}
            if isinstance(value.metadata, dict):
                 value.metadata['cache_stored_at'] = time.time()
                 value.metadata['cache_profile'] = profile # Guardar perfil usado para esta entrada

            if self._memory_only:
                self.cache[key] = value
            else:
                self.cache.set(key, value, expire=self.CACHE_DEFAULT_TIMEOUT)
            logger.debug(f"MatrixCache.set: Valor guardado exitosamente en caché para clave: {key}")
        except Exception as e:
            logger.error(f"MatrixCache.set: Error al guardar en caché. Clave intentada: {key if 'key' in locals() else 'No generada'}. Error: {e}", exc_info=True)

    def clear(self) -> None:
        """Limpia el caché.
        Este es el método de instancia correcto para limpiar el caché.
        """
        try:
            if self._memory_only:
                # Para el caché en memoria (dict), self.cache es el diccionario mismo
                self.cache.clear()
            elif hasattr(self.cache, 'clear'):
                # Para diskcache, self.cache es el objeto Cache
                self.cache.clear()
            else:
                # Si self.cache no es lo esperado, re-inicializar como fallback
                logger.warning("El objeto de caché no tiene método 'clear', reinicializando.")
                self._initialize_cache() # Esto creará un nuevo self.cache vacío

            logger.info("Caché de matrices limpiado exitosamente.")
        except Exception as e:
            logger.error(f"Error al limpiar el caché: {e}")

    def __del__(self):
        """Asegura que el caché se cierre correctamente."""
        if not self._memory_only and hasattr(self, 'cache'):
            self.cache.close()


# Instancia global del caché
matrix_cache = MatrixCache()

# Ejemplo de uso:
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Ejemplo de uso
    locations = [
        {"lat": 40.7128, "lng": -74.0060},  # NYC
        {"lat": 34.0522, "lng": -118.2437},  # LA
        {"lat": 41.8781, "lng": -87.6298},   # Chicago
    ]
    
    # Datos de ejemplo
    matrix_data = {
        "distances": [[0, 100, 200], [100, 0, 150], [200, 150, 0]],
        "durations": [[0, 60, 120], [60, 0, 90], [120, 90, 0]]
    }
    
    # Guardar en caché
    cache = MatrixCache()
    cache.set(locations, matrix_data)
    
    # Recuperar del caché
    cached_data = cache.get(locations)
    print(f"Datos del caché: {cached_data is not None}")
    
    # Limpiar caché
    cache.clear()
