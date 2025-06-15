"""Clase base abstracta para solucionadores de VRP."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from ortools.constraint_solver import routing_enums_pb2 as routing_enums
import logging
import sys
import os

# Añadir el directorio raíz al path para importaciones absolutas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Location, Vehicle
from models.vrp_models import VRPSolution, VRPSolutionStatus

# Configurar logging
logger = logging.getLogger(__name__)

class BaseVRPSolver(ABC):
    """
    Clase base abstracta para implementaciones de solucionadores VRP.
    
    Esta clase define la interfaz que deben implementar todos los solucionadores VRP.
    """
    
    def __init__(self):
        """Inicializa el solucionador con datos por defecto."""
        self.data = {}
        self.locations = []
        self.vehicles = []
        self.distance_matrix = []
        self.duration_matrix = []
        self._is_loaded = False
        self._solution = None
        self.optimization_profile: Optional[Dict[str, Any]] = None
        logger.debug("Inicializado BaseVRPSolver")
    
    @abstractmethod
    def load_problem(
        self,
        distance_matrix: List[List[Union[int, float]]],
        locations: List[Union[Dict[str, Any], Location]],
        vehicles: List[Union[Dict[str, Any], Vehicle]],
        duration_matrix: Optional[List[List[Union[int, float]]]] = None,
        optimization_profile: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Carga un problema VRP en el solucionador.
        
        Args:
            distance_matrix: Matriz de distancias entre ubicaciones (NxN)
            locations: Lista de ubicaciones (el primer elemento debe ser el depósito)
            vehicles: Lista de vehículos disponibles
            duration_matrix: Matriz opcional de duraciones entre ubicaciones (NxN)
            optimization_profile: Perfil de optimización para el problema
            **kwargs: Argumentos adicionales específicos de la implementación
            
        Raises:
            ValueError: Si los datos de entrada no son válidos
        """
        # Validar y convertir ubicaciones
        self.locations = [
            loc if isinstance(loc, Location) else Location(**loc) 
            for loc in locations
        ]
        
        # Validar y convertir vehículos
        self.vehicles = [
            veh if isinstance(veh, Vehicle) else Vehicle(**veh)
            for veh in vehicles
        ]
        
        # Validar matrices
        if not distance_matrix or not all(len(row) == len(distance_matrix) for row in distance_matrix):
            raise ValueError("La matriz de distancias debe ser cuadrada y no vacía")
            
        if len(distance_matrix) != len(self.locations):
            raise ValueError(
                f"La matriz de distancias debe tener el mismo tamaño que el número de ubicaciones. "
                f"Esperado: {len(self.locations)}, Obtenido: {len(distance_matrix)}"
            )
            
        if duration_matrix is not None:
            if len(duration_matrix) != len(self.locations):
                raise ValueError(
                    f"La matriz de duraciones debe tener el mismo tamaño que el número de ubicaciones. "
                    f"Esperado: {len(self.locations)}, Obtenido: {len(duration_matrix)}"
                )
            if any(len(row) != len(duration_matrix) for row in duration_matrix):
                raise ValueError("La matriz de duraciones debe ser cuadrada")
            self.duration_matrix = duration_matrix
            
        self.distance_matrix = distance_matrix
        self._is_loaded = True
        self._solution = None
        
        # Cargar perfil de optimización
        self.optimization_profile = optimization_profile or {}
        
        logger.info(f"Problema cargado con {len(self.locations)} ubicaciones y {len(self.vehicles)} vehículos")
    
    @abstractmethod
    def solve(
        self,
        time_limit_seconds: int = 30,
        optimization_profile: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> VRPSolution:
        """
        Resuelve el problema VRP cargado.
        
        Args:
            time_limit_seconds: Tiempo máximo de resolución en segundos
            optimization_profile: Perfil de optimización para la resolución
            **kwargs: Argumentos adicionales específicos de la implementación
            
        Returns:
            VRPSolution: Solución del problema
            
        Raises:
            RuntimeError: Si el problema no ha sido cargado o no se puede resolver
        """
        if optimization_profile is not None:
            self.optimization_profile = optimization_profile
        if not self._is_loaded:
            raise RuntimeError("No se ha cargado ningún problema. Use load_problem() primero.")
    
    def clear(self) -> None:
        """Limpia el estado del solucionador para resolver un nuevo problema."""
        self.data = {}
        self.locations = []
        self.vehicles = []
        self.distance_matrix = []
        self.duration_matrix = []
        self._is_loaded = False
        self._solution = None
        self.optimization_profile = None
        logger.debug("Solucionador reiniciado")
    
    @property
    def is_loaded(self) -> bool:
        """Indica si se ha cargado un problema en el solucionador."""
        return self._is_loaded
    
    @property
    def has_solution(self) -> bool:
        """Indica si existe una solución disponible."""
        return self._solution is not None
    
    def get_solution(self) -> Optional[VRPSolution]:
        """
        Devuelve la solución actual si existe.
        
        Returns:
            VRPSolution o None si no hay solución disponible
        """
        return self._solution
