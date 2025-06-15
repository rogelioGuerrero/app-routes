from abc import ABC, abstractmethod
from typing import Any, Dict

class Constraint(ABC):
    """Clase base abstracta para todas las restricciones del VRP."""
    
    @abstractmethod
    def apply(self, routing, manager, data: Dict[str, Any]):
        """
        Aplica la restricción al modelo de enrutamiento.
        
        Args:
            routing: Modelo de enrutamiento de OR-Tools
            manager: Manager de índices de OR-Tools
            data: Datos del problema
        """
        pass
