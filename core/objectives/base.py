from abc import ABC, abstractmethod
from typing import Any, Dict

class Objective(ABC):
    """Clase base abstracta para todos los objetivos del VRP."""
    
    @abstractmethod
    def apply(self, routing, manager, data: Dict[str, Any]):
        """
        Aplica el objetivo al modelo de enrutamiento.
        
        Args:
            routing: Modelo de enrutamiento de OR-Tools
            manager: Manager de índices de OR-Tools
            data: Datos del problema
        """
        pass
