"""Adaptador para integrar el CVRPSolver con el sistema de matrices."""

from typing import List, Dict, Any, Optional, Tuple
import logging
import time
from datetime import datetime

from ..base_solver import BaseVRPSolver
from core.matrix_adapter import MatrixAdapter
from models.vrp_models import VRPSolution
from .solver import CVRPSolver

logger = logging.getLogger(__name__)

class CVRPSolverAdapter:
    """
    Adaptador que integra el CVRPSolver con el sistema de matrices de distancia/tiempo.
    
    Este adaptador se encarga de:
    1. Obtener matrices de distancia y tiempo usando el sistema de proveedores
    2. Manejar el caché de matrices
    3. Proporcionar una interfaz limpia para resolver problemas CVRP
    """
    
    def __init__(self):
        """Inicializa el adaptador con un CVRPSolver y MatrixAdapter."""
        self.solver = CVRPSolver()
        self.matrix_adapter = MatrixAdapter()
        self._is_loaded = False
    
    def load_problem(
        self,
        distance_matrix: List[List[float]],
        locations: List[Dict[str, Any]],
        vehicles: List[Dict[str, Any]],
        duration_matrix: Optional[List[List[float]]] = None,
        **kwargs
    ) -> None:
        """
        Carga un problema VRP en el solucionador.
        
        Args:
            distance_matrix: Matriz de distancias entre ubicaciones (NxN)
            locations: Lista de ubicaciones (el primer elemento debe ser el depósito)
            vehicles: Lista de vehículos disponibles
            duration_matrix: Matriz opcional de duraciones entre ubicaciones (NxN)
            **kwargs: Argumentos adicionales
        """
        # Si ya estamos en el proceso de cargar un problema con matrices, no hacer nada
        if hasattr(self, '_loading_with_matrices'):
            return
            
        # Cargar el problema en el solucionador
        self.solver.load_problem(
            distance_matrix=distance_matrix,
            locations=locations,
            vehicles=vehicles,
            duration_matrix=duration_matrix,
            **kwargs
        )
        self._is_loaded = True
    
    def solve(self, time_limit_seconds: int = 30, **kwargs) -> VRPSolution:
        """
        Resuelve el problema VRP cargado.
        
        Args:
            time_limit_seconds: Tiempo máximo de resolución en segundos
            **kwargs: Argumentos adicionales para el solucionador
            
        Returns:
            VRPSolution: Solución del problema
        """
        if not self._is_loaded:
            raise RuntimeError("Debe cargar un problema primero con load_problem()")
            
        return self.solver.solve(time_limit_seconds=time_limit_seconds, **kwargs)
    
    async def _get_matrices(self, coords, providers=None, force_refresh=False, **kwargs):
        # Delegar en MatrixAdapter
        return await self.matrix_adapter.get_matrices(coords, providers, force_refresh, **kwargs)
    
    async def solve_cvrp(
        self,
        locations: List[Dict[str, Any]],
        vehicles: List[Dict[str, Any]],
        providers: Optional[List[Tuple[str, dict]]] = None,
        time_limit_seconds: int = 30,
        force_refresh: bool = False,
        optimization_profile: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> VRPSolution:
        """
        Resuelve un problema CVRP usando el sistema de matrices.
        
        Args:
            locations: Lista de ubicaciones con 'lat', 'lng' y 'demand'
            vehicles: Lista de vehículos con 'capacity'
            providers: Lista de proveedores a usar (si None, usa los predeterminados)
            time_limit_seconds: Tiempo máximo de resolución en segundos
            force_refresh: Si es True, ignora el caché y obtiene nuevas matrices
            optimization_profile: Perfil de optimización para el solucionador
            **kwargs: Argumentos adicionales para los proveedores
            
        Returns:
            VRPSolution: Solución del problema
        """
        start_time = time.time()
        
        # Validar ubicaciones
        if not locations:
            raise ValueError("La lista de ubicaciones no puede estar vacía")
            
        # Validar vehículos
        if not vehicles:
            raise ValueError("La lista de vehículos no puede estar vacía")
        
        # Extraer coordenadas para la matriz
        coords = [{'lat': loc['lat'], 'lng': loc['lng']} for loc in locations]
        
        logger.info(f"Resolviendo CVRP para {len(coords)} ubicaciones y {len(vehicles)} vehículos...")
        
        try:
            # Obtener matrices (con o sin caché según force_refresh)
            logger.info("Obteniendo matrices del proveedor...")
            distance_matrix, duration_matrix, provider_used, from_cache = await self._get_matrices(
                coords=coords,
                providers=providers,
                force_refresh=force_refresh,
                **kwargs
            )
            
            # Cargar el problema en el solucionador
            try:
                # Marcar que estamos cargando con matrices para evitar llamadas duplicadas
                self._loading_with_matrices = True
                
                self.solver.load_problem(
                    distance_matrix=distance_matrix,
                    duration_matrix=duration_matrix,
                    locations=locations,
                    vehicles=vehicles,
                    optimization_profile=optimization_profile
                )
                
                logger.info("Problema cargado exitosamente en el solucionador")
                
                # Resolver el problema
                logger.info(f"Iniciando resolución CVRP (límite: {time_limit_seconds}s)...")
                solution = self.solver.solve(
                    time_limit_seconds=time_limit_seconds,
                    optimization_profile=optimization_profile
                )
                
                # Calcular tiempo de ejecución
                execution_time = time.time() - start_time
                
                # Añadir metadatos adicionales
                solution.metadata.update({
                    'matrix_provider': provider_used,
                    'num_locations': len(locations),
                    'num_vehicles': solution.total_vehicles_used,
                    'from_cache': from_cache,
                    'execution_time_seconds': round(execution_time, 2),
                    'cached_at': datetime.utcnow().isoformat() if from_cache else None
                })
                
                logger.info(
                    f"Solución encontrada en {execution_time:.2f}s | "
                    f"Distancia: {solution.total_distance} | "
                    f"Vehículos usados: {solution.metadata.get('total_vehicles_used', 0)}"
                )
                
                return solution
                
            except Exception as load_error:
                logger.error(f"Error al cargar el problema en el solucionador: {str(load_error)}", exc_info=True)
                logger.error(f"Tipo de error: {type(load_error).__name__}")
                if hasattr(load_error, '__traceback__'):
                    import traceback
                    logger.error("Traceback completo: " + ''.join(traceback.format_tb(load_error.__traceback__)))
                raise
                
            finally:
                # Asegurarse de limpiar la bandera
                if hasattr(self, '_loading_with_matrices'):
                    delattr(self, '_loading_with_matrices')
                    
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error al resolver el problema CVRP: {error_msg}", exc_info=True)
            
            # Determinar el tipo de error
            if "caché" in error_msg.lower() or "cache" in error_msg.lower():
                status = "INVALID"
            elif "tiempo" in error_msg.lower() or "timeout" in error_msg.lower():
                status = "NO_SOLUTION_FOUND"
            else:
                status = "ROUTE_FAILED"
            
            # Crear respuesta de error con estado válido
            error_solution = VRPSolution(
                status=status,
                routes=[],
                total_distance=0,
                total_load=0,
                total_vehicles_used=0,
                metadata={
                    'error': error_msg,
                    'error_type': type(e).__name__,
                    'execution_time_seconds': round(time.time() - start_time, 2),
                    'num_locations': len(locations),
                    'num_vehicles': len(vehicles),
                    'timestamp': datetime.utcnow().isoformat(),
                    'warnings': [f"Error: {error_msg}"]
                }
            )
            
            logger.info("Solución de error generada")
            return error_solution

async def example_usage():
    """Ejemplo de uso del CVRPSolverAdapter."""
    # Datos de ejemplo
    locations = [
        {'id': 'depot', 'lat': 40.7128, 'lng': -74.0060, 'demand': 0},  # NYC (depósito)
        {'id': 'loc1', 'lat': 34.0522, 'lng': -118.2437, 'demand': 10},  # LA
        {'id': 'loc2', 'lat': 41.8781, 'lng': -87.6298, 'demand': 15},   # Chicago
        {'id': 'loc3', 'lat': 29.7604, 'lng': -95.3698, 'demand': 20},   # Houston
    ]
    
    vehicles = [
        {'id': 'veh1', 'capacity': 50},
        {'id': 'veh2', 'capacity': 30}
    ]
    
    # Configuración de proveedores (opcional)
    providers = [
        ('ors', {'profile': 'driving-car'}),
        ('google', {'mode': 'driving'}),
        ('euclidean', {})
    ]
    
    # Crear y usar el adaptador
    adapter = CVRPSolverAdapter()
    
    try:
        solution = await adapter.solve_cvrp(
            locations=locations,
            vehicles=vehicles,
            providers=providers,
            time_limit_seconds=10
        )
        
        # Mostrar resultados
        print("\n=== Solución CVRP ===")
        print(f"Estado: {solution.status}")
        print(f"Distancia total: {solution.total_distance} metros")
        print(f"Tiempo de ejecución: {solution.metadata.get('execution_time_seconds', 'N/A')} segundos")
        print(f"Desde caché: {solution.metadata.get('from_cache', False)}")
        print(f"Proveedor de matrices: {solution.metadata.get('matrix_provider', 'N/A')}")
        
        for i, route in enumerate(solution.routes):
            print(f"\nRuta {i+1} (Distancia: {route.distance} metros):")
            print(" -> ".join(str(loc) for loc in route.locations))
            
        # Segunda ejecución para probar la caché local
        print("\n--- Segunda llamada (debería usar caché local) ---")
        cached_solution = await adapter.solve_cvrp(
            locations=locations,
            vehicles=vehicles,
            providers=providers,
            time_limit_seconds=10
        )
        print(f"Tiempo de ejecución (caché): {cached_solution.metadata.get('execution_time_seconds', 'N/A')} segundos")
        print(f"Desde caché: {cached_solution.metadata.get('from_cache', False)}")
        print(f"Proveedor de matrices (caché): {cached_solution.metadata.get('matrix_provider', 'N/A')}")
    except Exception as e:
        print(f"Error: {str(e)}")

# Para probar el ejemplo:
# import asyncio
# asyncio.run(example_usage())
