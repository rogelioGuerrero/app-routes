from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass

@dataclass
class MatrixResult:
    """Resultado de una solicitud de matriz de distancias/tiempos."""
    distances: List[List[float]]  # Matriz de distancias en metros
    durations: List[List[float]]   # Matriz de duraciones en segundos
    provider: str                 # Nombre del proveedor usado
    
    def __post_init__(self):
        # Validar que las matrices sean cuadradas y del mismo tamaño
        n = len(self.distances)
        if not all(len(row) == n for row in self.distances):
            raise ValueError("La matriz de distancias no es cuadrada")
            
        if not all(len(row) == n for row in self.durations):
            raise ValueError("La matriz de duraciones no es cuadrada")
            
        if len(self.distances) != len(self.durations):
            raise ValueError("Las matrices de distancias y duraciones tienen tamaños diferentes")


class MatrixProvider(ABC):
    """Interfaz base para proveedores de matrices de distancia/tiempo."""
    
    @abstractmethod
    async def get_matrix(
        self,
        locations: List[Dict[str, float]],
        metrics: list[Literal['distances', 'durations']] = None,
        profile: str = 'driving',
        **kwargs
    ) -> MatrixResult:
        """
        Obtiene las matrices de distancias y/o tiempos entre ubicaciones.
        
        Args:
            locations: Lista de ubicaciones con 'lat' y 'lng'
            metrics: Lista de métricas a obtener ('distances', 'durations' o ambas)
            profile: Perfil de ruta (depende del proveedor)
            **kwargs: Parámetros adicionales específicos del proveedor
            
        Returns:
            Objeto MatrixResult con las matrices solicitadas
            
        Nota:
            - Las distancias deben estar en metros
            - Las duraciones deben estar en segundos
        """
        pass
    
    @staticmethod
    def validate_locations(locations: List[Dict[str, Any]]) -> None:
        """Valida el formato de las ubicaciones."""
        if not locations:
            raise ValueError("La lista de ubicaciones no puede estar vacía")
            
        for i, loc in enumerate(locations):
            if 'lat' not in loc or 'lng' not in loc:
                raise ValueError(f"Ubicación {i} debe tener 'lat' y 'lng'")
            
            try:
                lat = float(loc['lat'])
                lng = float(loc['lng'])
                if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                    raise ValueError(
                        f"Coordenadas fuera de rango en ubicación {i}. "
                        f"Latitud debe estar entre -90 y 90, longitud entre -180 y 180"
                    )
            except (ValueError, TypeError) as e:
                if 'fuera de rango' in str(e):
                    raise
                raise ValueError(f"Coordenadas inválidas en ubicación {i}: {e}")
    
    # Métodos de conveniencia para compatibilidad hacia atrás
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
        """Método de conveniencia para obtener solo duraciones."""
        result = await self.get_matrix(locations, metrics=['durations'], **kwargs)
        return result.durations
