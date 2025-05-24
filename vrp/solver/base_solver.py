from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


class SolverType(str, Enum):
    """Tipos de solvers disponibles."""
    OR_TOOLS = "or_tools"
    # Futuros solvers podrían ser: HEURISTIC, LKH, ETC.


@dataclass
class VRPSolution:
    """Representa la solución de un problema VRP."""
    
    # Lista de rutas, donde cada ruta es una lista de índices de ubicaciones
    routes: List[List[int]]
    
    # Distancia total de cada ruta (en las mismas unidades que la matriz de entrada)
    distances: List[float]
    
    # Tiempo total de cada ruta (en las mismas unidades que la matriz de entrada)
    times: List[float]
    
    # Carga de cada vehículo (si aplica)
    loads: List[float]
    
    # Índices de ubicaciones no asignadas
    unassigned: List[int]
    
    # Metadatos adicionales sobre la solución
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la solución a un diccionario."""
        return {
            "routes": self.routes,
            "distances": self.distances,
            "times": self.times,
            "loads": self.loads,
            "unassigned": self.unassigned,
            "metadata": self.metadata
        }


class BaseVRPSolver(ABC):
    """Interfaz base para los solvers de VRP."""
    
    @property
    @abstractmethod
    def solver_type(self) -> SolverType:
        """Tipo de solver."""
        pass
    
    @abstractmethod
    async def solve(
        self,
        distance_matrix: List[List[float]],
        time_matrix: List[List[float]],
        locations: List[Any],
        vehicles: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> VRPSolution:
        """
        Resuelve un problema VRP.
        
        Args:
            distance_matrix: Matriz de distancias entre ubicaciones
            time_matrix: Matriz de tiempos entre ubicaciones
            locations: Lista de ubicaciones (incluyendo depósitos)
            vehicles: Lista de vehículos con sus características
            constraints: Restricciones adicionales
            **kwargs: Parámetros adicionales específicos del solver
                
        Returns:
            VRPSolution con las rutas y métricas
            
        Raises:
            ValueError: Si los parámetros de entrada no son válidos
        """
        pass
