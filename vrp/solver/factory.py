from typing import Type, Dict, Optional
from .base_solver import BaseVRPSolver, SolverType
from .or_tools_solver import ORToolsSolver


class SolverFactory:
    """Factory para crear instancias de solvers VRP."""
    
    # Mapeo de tipos de solver a sus clases implementadoras
    _solvers: Dict[SolverType, Type[BaseVRPSolver]] = {
        SolverType.OR_TOOLS: ORToolsSolver,
        # Registrar nuevos solvers aquí
    }
    
    @classmethod
    def create_solver(
        cls, 
        solver_type: SolverType = SolverType.OR_TOOLS,
        **kwargs
    ) -> BaseVRPSolver:
        """
        Crea una instancia del solver especificado.
        
        Args:
            solver_type: Tipo de solver a crear
            **kwargs: Argumentos adicionales para el constructor del solver
                
        Returns:
            Instancia del solver solicitado
            
        Raises:
            ValueError: Si el tipo de solver no está soportado
        """
        solver_class = cls._solvers.get(solver_type)
        if not solver_class:
            raise ValueError(f"Solver no soportado: {solver_type}")
            
        return solver_class(**kwargs)
    
    @classmethod
    def register_solver(
        cls, 
        solver_type: SolverType, 
        solver_class: Type[BaseVRPSolver]
    ) -> None:
        """
        Registra un nuevo tipo de solver.
        
        Args:
            solver_type: Tipo de solver
            solver_class: Clase que implementa el solver
                
        Raises:
            TypeError: Si la clase no hereda de BaseVRPSolver
        """
        if not issubclass(solver_class, BaseVRPSolver):
            raise TypeError("La clase del solver debe heredar de BaseVRPSolver")
            
        cls._solvers[solver_type] = solver_class


# Instancia de solver por defecto para facilitar el uso
default_solver = SolverFactory.create_solver(SolverType.OR_TOOLS)
