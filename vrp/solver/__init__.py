"""
Módulo para la resolución de problemas de enrutamiento de vehículos (VRP).

Este módulo proporciona una interfaz unificada para resolver problemas VRP
utilizando diferentes algoritmos y bibliotecas subyacentes.
"""
from .base_solver import SolverType, VRPSolution, BaseVRPSolver
from .or_tools_solver import ORToolsSolver
from .factory import SolverFactory, default_solver

# Exportar símbolos principales
__all__ = [
    'SolverType',
    'VRPSolution',
    'BaseVRPSolver',
    'ORToolsSolver',
    'SolverFactory',
    'default_solver'
]
