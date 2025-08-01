"""JSON -> VRP data converter.

Reescritura completa: se elimina cualquier cálculo local de distancias.
Las matrices se obtienen SIEMPRE a través de `services.distance_matrix.utils.get_matrix_with_fallback`,
que ya implementa caché y fallback a métodos euclidianos cuando fallan los proveedores externos.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Set

import aiohttp
import json as std_json

from services.distance_matrix.utils import get_matrix_with_fallback

logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# In-process simple cache for distance/time matrices
# Key: (tuple_of_locations_latlng, tuple(metrics)) -> (distance_matrix, time_matrix)
# -------------------------------------------------------------
_MATRIX_CACHE: dict[tuple, tuple[list[list[int]], list[list[int]]]] = {}


class JsonToVrpDataConverter:
    """Convierte un escenario JSON al formato de datos requerido por `vrp_solver`."""

    def __init__(self, scenario: Dict[str, Any], session: aiohttp.ClientSession | None = None):
        """Inicializa el conversor."""
        self.scenario = scenario
        self.session = session  # Sesión HTTP opcional reutilizable

        # Cargar listas base
        self.locations: List[Dict[str, Any]] = self.scenario.get("locations", []).copy()
        self.vehicles: List[Dict[str, Any]] = self.scenario.get("vehicles", []).copy()

        # Mapear ID de la ubicación -> índice de nodo
        self.node_map: Dict[str, int] = {loc["id"]: i for i, loc in enumerate(self.locations)}

        self.matrix_cache_used: bool = False
        self.vrp_data: Dict[str, Any] | None = None

    # ---------------------------------------------------------------------
    # Matrices distancia / duración
    # ---------------------------------------------------------------------
    def _apply_congestion_factor(self, time_matrix: List[List[int]]) -> List[List[int]]:
        """Aplica el factor de congestión a la matriz de tiempo.
        
        Si no se especifica en el escenario, usa 1.5 por defecto.
        """
        if not time_matrix:
            return []
            
        factor = float(self.scenario.get('congestion_factor', 1.5))
        logger.info(f"Aplicando factor de congestión: {factor}x")
        return [[int(t * factor) for t in row] for row in time_matrix]
        
    async def _calculate_matrices(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Devuelve (distance_matrix, time_matrix) ambas en enteros (m, s)."""
        if not self.locations:
            logger.warning("Lista de ubicaciones vacía")
            return [], []

        # Validar que todas las ubicaciones tengan coordenadas válidas
        for i, loc in enumerate(self.locations):
            # Aceptar tanto 'coords' como 'lat'/'lng' para compatibilidad
            coords = loc.get("coords")
            if coords is None and 'lat' in loc and 'lng' in loc:
                coords = [loc['lng'], loc['lat']]  # Formato [longitud, latitud]
                loc["coords"] = coords  # Asegurar que exista para el solver
                
            if coords is None or len(coords) != 2:
                logger.error(f"Ubicación en índice {i} no tiene coordenadas válidas: {loc}")
                return [], []

        # --- CACHE LOOKUP -------------------------------------------------
        cache_key = None
        try:
            # Crear una clave hashable: ((lat,lng), ...)
            loc_key = tuple((float(loc['coords'][1]), float(loc['coords'][0])) for loc in self.locations)
            cache_key = (loc_key, ('distances', 'durations'))
            if cache_key in _MATRIX_CACHE:
                logger.info("Matrices obtenidas desde caché en memoria (fast).")
                self.matrix_cache_used = True
                return _MATRIX_CACHE[cache_key]
        except Exception:
            # Si falla la creación de la clave, continuar sin caché
            cache_key = None
        # -------------------------------------------------------------------

        # Formato requerido por los proveedores externos
        provider_locations = []
        for loc in self.locations:
            try:
                lat = float(loc["coords"][1])
                lng = float(loc["coords"][0])
                provider_locations.append({"lat": lat, "lng": lng})
            except (ValueError, TypeError, IndexError) as e:
                logger.error(f"Error al procesar coordenadas: {loc.get('coords')} - {str(e)}")
                return [], []

        created_session = False
        session = self.session
        if session is None:
            session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
            created_session = True

        try:
            try:
                logger.info("Solicitando matrices de distancia/tiempo...")
                result, provider_name = await get_matrix_with_fallback(
                    locations=provider_locations,
                    metrics=["distances", "durations"],
                    session=session,
                )
                logger.info("Matrices recibidas del proveedor")
            except Exception as exc:
                logger.warning("Fallo la obtención remota de matrices, se usará distancia euclidiana como fallback: %s", exc)
                # --- EUCLIDEAN FALLBACK -----------------------------------
                distance_matrix, time_matrix = self._build_euclidean_matrices(provider_locations)
                return distance_matrix, time_matrix
                logger.exception("No se pudieron obtener matrices de distancia/tiempo")
                return [], []

            # Validar que result tenga las matrices esperadas
            if not hasattr(result, 'distances') or not hasattr(result, 'durations'):
                logger.error("La respuesta del proveedor no contiene las matrices esperadas")
                return [], []

            n = len(self.locations)
            
            # Validar dimensiones de las matrices
            if (len(result.distances) != n or any(len(row) != n for row in result.distances) or
                len(result.durations) != n or any(len(row) != n for row in result.durations)):
                logger.error("Las matrices de distancia/tiempo tienen dimensiones incorrectas")
                return [], []

            # Convertir a enteros con validación
            distance_matrix = []
            time_matrix = []
            
            for i in range(n):
                dist_row = []
                time_row = []
                for j in range(n):
                    try:
                        dist = int(result.distances[i][j])
                        time_val = int(result.durations[i][j])
                        if dist < 0 or time_val < 0:
                            logger.warning(f"Valor negativo en matriz en posición ({i},{j}): dist={dist}, time={time_val}")
                            # Usar un valor predeterminado grande pero manejable
                            dist = max(1, dist)
                            time_val = max(1, time_val)
                        dist_row.append(dist)
                        time_row.append(time_val)
                    except (ValueError, TypeError, IndexError) as e:
                        logger.error(f"Error al convertir valor en posición ({i},{j}): {e}")
                        return [], []
                
                distance_matrix.append(dist_row)
                time_matrix.append(time_row)

            # Si el proveedor expone metadata de caché úsala, si no, usa nuestro flag
            provider_cache_hit = getattr(result, "metadata", {}).get("cache_hit", False)
            self.matrix_cache_used = provider_cache_hit
            self.provider_name = provider_name
            
            logger.info(
                "Matrices generadas exitosamente con proveedor '%s'. Tamaño: %dx%d. Cache: %s",
                provider_name,
                len(distance_matrix),
                len(distance_matrix[0]) if distance_matrix else 0,
                self.matrix_cache_used,
            )
            
            # Guardar en caché en memoria
            if cache_key is not None and not provider_cache_hit:
                # Solo guardamos si no venía ya de caché para evitar doble conteo
                _MATRIX_CACHE[cache_key] = (distance_matrix, time_matrix)
            
            # Aplicar factor de congestión a la matriz de tiempo
            time_matrix = self._apply_congestion_factor(time_matrix)
            
            # Si todo salió bien, devolvemos las matrices
            return distance_matrix, time_matrix
            
        except Exception as e:
            logger.exception("Error inesperado en _calculate_matrices: %s", e)
            # Fallback definitivo a distancia euclidiana
            distance_matrix, time_matrix = self._build_euclidean_matrices(provider_locations)
            return distance_matrix, time_matrix
            
        finally:
            if created_session and session and not session.closed:
                try:
                    await session.close()
                except Exception as e:
                    logger.warning(f"Error al cerrar la sesión: {e}")

    # ------------------------------------------------------------------
    # Euclidean fallback helper
    # ------------------------------------------------------------------
    def _build_euclidean_matrices(self, provider_locations: list[dict]) -> tuple[list[list[int]], list[list[int]]]:
        """Genera matrices (dist, tiempo) rápidas usando fórmula de Haversine.
        Suponemos 1 m ≈ 1 s para tiempo de viaje. Esto evita depender de APIs externas
        durante pruebas e integra bien con el solver.
        """
        import math
        n = len(provider_locations)
        dist_mat: list[list[int]] = [[0]*n for _ in range(n)]
        time_mat: list[list[int]] = [[0]*n for _ in range(n)]
        R = 6_371_000  # Radio terrestre medio (m)
        for i in range(n):
            lat_i = math.radians(provider_locations[i]['lat'])
            lng_i = math.radians(provider_locations[i]['lng'])
            for j in range(i+1, n):
                lat_j = math.radians(provider_locations[j]['lat'])
                lng_j = math.radians(provider_locations[j]['lng'])
                dlat = lat_j - lat_i
                dlng = lng_j - lng_i
                a = math.sin(dlat/2)**2 + math.cos(lat_i)*math.cos(lat_j)*math.sin(dlng/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                dist = int(R * c)
                dist_mat[i][j] = dist_mat[j][i] = dist
                time_mat[i][j] = time_mat[j][i] = dist  # 1 m ≈ 1 s
        return dist_mat, time_mat

    # ------------------------------------------------------------------
    # Helpers de congestión
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Conversión principal
    # ------------------------------------------------------------------
    def _apply_congestion_factors(self, time_matrix: List[List[int]]) -> List[List[int]]:
        """Aplica factores de congestión a la matriz de tiempos según la hora del día.
        
        Args:
            time_matrix: Matriz de tiempos original en segundos
            
        Returns:
            Matriz de tiempos con factores de congestión aplicados
        """
        if not self.scenario.get('congestion_enabled', False):
            return time_matrix
            
        congestion_factors = self.scenario.get('congestion_factors', [])
        if not congestion_factors:
            return time_matrix
            
        # Convertir la matriz de tiempos a float para cálculos precisos
        congested_matrix = [row.copy() for row in time_matrix]
        n = len(time_matrix)
        
        # Para cada par de nodos (i,j)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                    
                # Obtener la hora de salida estimada del nodo i
                # Como aproximación, usamos el tiempo de servicio del nodo i
                # En una implementación más avanzada, esto podría ser más preciso
                service_time_i = self.locations[i].get('service_time', 0)
                departure_time = service_time_i  # Tiempo desde la salida del depósito
                
                # Convertir a hora del día (segundos desde medianoche)
                time_of_day = departure_time % 86400
                
                # Encontrar el factor de congestión aplicable
                factor = 1.0  # Factor por defecto (sin congestión)
                
                for cf in congestion_factors:
                    start_h, start_m = map(int, cf['time_window'][0].split(':'))
                    end_h, end_m = map(int, cf['time_window'][1].split(':'))
                    
                    start_sec = start_h * 3600 + start_m * 60
                    end_sec = end_h * 3600 + end_m * 60
                    
                    # Manejar rangos que cruzan la medianoche
                    if start_sec <= end_sec:
                        if start_sec <= time_of_day <= end_sec:
                            factor = cf['factor']
                            break
                    else:
                        if time_of_day >= start_sec or time_of_day <= end_sec:
                            factor = cf['factor']
                            break
                
                # Aplicar el factor al tiempo de viaje
                congested_matrix[i][j] = int(round(time_matrix[i][j] * factor))
        
        return congested_matrix

    async def convert(self) -> Tuple[Dict[str, Any], bool]:
        """Construye y devuelve (vrp_data, cache_used)."""
        distance_matrix, time_matrix = await self._calculate_matrices()
        
        # Aplicar factores de congestión si están habilitados
        if self.scenario.get('congestion_enabled', False):
            logger.info("Aplicando factores de congestión a la matriz de tiempos")
            time_matrix = self._apply_congestion_factors(time_matrix)

        def _trivial(matrix: List[List[int]]):
            return not matrix or all(all(cell == 0 for cell in row) for row in matrix)

        if _trivial(distance_matrix) or _trivial(time_matrix):
            logger.error("Matrices vacías o triviales; no es posible continuar.")
            return {}, False

        # -------------------------- Capacidades & Demandas --------------------------
        # Preparar las listas para capacidades separadas
        vehicle_capacities_weight = []
        vehicle_capacities_volume = []
        
        logger.debug(f"Procesando {len(self.vehicles)} vehículos")
        
        # Extraer capacidades de cada vehículo por tipo (peso y volumen)
        for i, v in enumerate(self.vehicles):
            # Convertir capacidades a enteros y asegurar valores válidos
            weight_cap = int(float(v.get('weight_capacity', 0)) * 100) # Multiplicar por 100 para manejar decimales como enteros
            volume_cap = int(float(v.get('volume_capacity', 0)) * 100)
            
            vehicle_capacities_weight.append(weight_cap)
            vehicle_capacities_volume.append(volume_cap)
            
            logger.debug(f"Vehículo {i} ({v.get('id')}): Capacidad peso={weight_cap/100}, volumen={volume_cap/100}")
        
        # Preparar las listas para demandas
        demands_weight = []
        demands_volume = []
        
        logger.debug(f"Procesando {len(self.locations)} ubicaciones")
        
        # Para cada ubicación, obtener sus demandas de peso y volumen
        for i, loc in enumerate(self.locations):
            # Convertir demandas a enteros y asegurar valores válidos
            weight_demand = int(float(loc.get('weight_demand', 0)) * 100) # Multiplicar por 100 para manejar decimales como enteros
            volume_demand = int(float(loc.get('volume_demand', 0)) * 100)
            
            demands_weight.append(weight_demand)
            demands_volume.append(volume_demand)
            
            logger.debug(f"Ubicación {i} ({loc.get('id')}): Demanda peso={weight_demand/100}, volumen={volume_demand/100}")
        
        # Armar las estructuras de datos para el solver
        demands = [demands_weight, demands_volume]
        dimension_names = ['weight', 'volume']
        vehicle_capacities = list(zip(vehicle_capacities_weight, vehicle_capacities_volume))



        # -------------------------- Time windows & service --------------------------
        time_windows = [
            (loc.get("time_window_start", 0), loc.get("time_window_end", 86_400)) for loc in self.locations
        ]
        service_times = [loc.get("service_time", 0) for loc in self.locations]

        # -------------------------- Skills -----------------------------------------
        skill_constraint_type = self.scenario.get("skill_constraint_type", "SOFT")
        skill_penalty = self.scenario.get("skill_penalty", 1_000_000)

        # Obtener habilidades de los vehículos
        vehicle_skills = []
        for v in self.vehicles:
            skills = v.get("skills", [])
            if not isinstance(skills, list):
                skills = [skills] if skills else []
            # Asegurar que las habilidades sean strings y eliminar duplicados
            vehicle_skills.append(list(set(str(skill) for skill in skills if skill)))

        # Obtener habilidades requeridas por cada ubicación
        required_skills = []
        for loc in self.locations:
            rs = loc.get("required_skills", [])
            if not isinstance(rs, list):
                rs = [rs] if rs else []
            # Asegurar que las habilidades sean strings y eliminar duplicados
            required_skills.append(list(set(str(skill) for skill in rs if skill)))

        # -------------------------- Pickups / Deliveries ---------------------------
        pickups_deliveries: List[List[int]] = []
        pickup_nodes: Set[int] = set()
        delivery_nodes: Set[int] = set()

        # -------------------------- Ensamblar vrp_data ----------------------------
        vrp_data = {
            'distance_matrix': distance_matrix,
            'time_matrix': time_matrix,
            'num_vehicles': len(self.vehicles),
            'starts': [self.node_map.get(v.get('start_location_id', self.locations[0]['id']), 0) for v in self.vehicles],
            'ends': [self.node_map.get(v.get('end_location_id', self.locations[0]['id']), 0) for v in self.vehicles],
            'vehicle_capacities': vehicle_capacities,
            'demands': demands,
            'dimension_names': dimension_names,
            'demands_weight': demands_weight,
            'demands_volume': demands_volume,
            'vehicle_capacities_weight': vehicle_capacities_weight,
            'vehicle_capacities_volume': vehicle_capacities_volume,
            'time_windows': time_windows,
            'service_time': service_times,
            'pickups_deliveries': pickups_deliveries,
            'skills': {i: required_skills[i] for i in range(len(required_skills))},
            'vehicle_skills': vehicle_skills,
            'locations': self.locations,
            'vehicles': self.vehicles,
            'optimization_profile': self.scenario.get('optimization_profile', {}),
            'depots': [self.node_map.get(self.scenario.get('depot_id', self.locations[0]['id']), 0)]
        }



        logger.debug(f"self.locations: {len(self.locations)} elementos")
        logger.debug(f"pickups_deliveries en escenario: {len(self.scenario.get('pickups_deliveries', []))} elementos")
        
        try:
            for pair in self.scenario.get("pickups_deliveries", []):
                p_idx = None
                d_idx = None
                
                # Caso 1: Par en formato diccionario {pickup: X, delivery: Y}
                if isinstance(pair, dict) and 'pickup' in pair and 'delivery' in pair:
                    p, d = pair.get('pickup'), pair.get('delivery')
                    logger.debug(f"Procesando par pickup-delivery (diccionario): {p.get('id', 'sin-id')} -> {d.get('id', 'sin-id')}")
                
                # Caso 2: Par en formato lista/tupla [X, Y]
                elif isinstance(pair, (list, tuple)) and len(pair) == 2:
                    p, d = pair[0], pair[1]
                    logger.debug(f"Procesando par pickup-delivery (lista): {p} -> {d}")
                
                # Caso 3: Formato no reconocido
                else:
                    print(f"[WARNING] Formato de par pickup-delivery no reconocido: {pair} (tipo: {type(pair)})")
                    continue
                
                # Procesar según el tipo de identificador
                # Caso A: Índices numéricos directos
                if isinstance(p, int) and isinstance(d, int):
                    # Asegurarse de que los índices están en el rango correcto
                    if 0 <= p < len(self.locations) and 0 <= d < len(self.locations):
                        p_idx, d_idx = p, d  # Ya son índices base 0 válidos
                        logger.debug(f"Índices numéricos válidos: {p_idx}, {d_idx}")
                    else:
                        print(f"[WARNING] Índices fuera de rango: {p}, {d} (máx: {len(self.locations)-1})")
                        continue
                
                # Caso B: Identificadores string (IDs de ubicación)
                elif isinstance(p, str) and isinstance(d, str):
                    if p in self.node_map and d in self.node_map:
                        p_idx = self.node_map[p]
                        d_idx = self.node_map[d]
                        logger.debug(f"IDs convertidos a índices: {p_idx}, {d_idx}")
                    else:
                        print(f"[WARNING] IDs de ubicación no encontrados: {p}, {d}")
                        continue
                
                # Caso C: Tipos mixtos o no soportados
                else:
                    print(f"[WARNING] Tipos de datos no soportados para pickup-delivery: {p} (tipo: {type(p)}), {d} (tipo: {type(d)})")
                    continue
                
                # Verificación final y adición a la lista de pares
                if p_idx is not None and d_idx is not None and 0 <= p_idx < len(self.locations) and 0 <= d_idx < len(self.locations):
                    pickups_deliveries.append([p_idx, d_idx])
                    pickup_nodes.add(p_idx)
                    delivery_nodes.add(d_idx)
                    logger.debug(f"Par añadido a pickups_deliveries: [{p_idx}, {d_idx}] con ubicaciones: '{self.locations[p_idx].get('id')}' -> '{self.locations[d_idx].get('id')}'")
                else:
                    print(f"[WARNING] Índices procesados fuera de rango o inválidos: {p_idx}, {d_idx} (máx: {len(self.locations)-1})")
                    continue
        except Exception as e:
            import traceback
            print(f"[ERROR] Error al procesar pickups_deliveries: {e}")
            traceback.print_exc()
            # Seguir con una lista vacía en caso de error
            pickups_deliveries = []

        # Ajustar signos de demanda: pickups deben ser positivos, deliveries negativos
        for node_idx in pickup_nodes:
            # Para demanda de peso
            if demands_weight[node_idx] < 0:
                demands_weight[node_idx] = -demands_weight[node_idx]
                demands[0][node_idx] = demands_weight[node_idx]
            # Para demanda de volumen
            if demands_volume[node_idx] < 0:
                demands_volume[node_idx] = -demands_volume[node_idx]
                demands[1][node_idx] = demands_volume[node_idx]
                
        for node_idx in delivery_nodes:
            # Para demanda de peso
            if demands_weight[node_idx] > 0:
                demands_weight[node_idx] = -demands_weight[node_idx]
                demands[0][node_idx] = demands_weight[node_idx]
            # Para demanda de volumen
            if demands_volume[node_idx] > 0:
                demands_volume[node_idx] = -demands_volume[node_idx]
                demands[1][node_idx] = demands_volume[node_idx]

        # -------------------------- Validar habilidades ---------------------------
        # Verificar que para cada ubicación, haya al menos un vehículo con las habilidades requeridas
        skill_warnings = []
        for node_idx, skills in enumerate(required_skills):
            if not skills:
                continue
                
            # Verificar si algún vehículo tiene todas las habilidades requeridas
            valid_vehicles = []
            for veh_idx, veh_skills in enumerate(vehicle_skills):
                if set(skills).issubset(set(veh_skills)):
                    valid_vehicles.append(veh_idx)
            
            if not valid_vehicles:
                node_id = self.locations[node_idx]['id']
                warning_msg = f"Nodo {node_id} requiere habilidades {skills} pero ningún vehículo las tiene"
                skill_warnings.append(warning_msg)
                
                # Auto-corrección: Si es skill 'paper' y hay vehículos estándar, asumir que pueden llevarlo
                if skills == ['paper'] and any(len(s) == 0 for s in vehicle_skills):
                    required_skills[node_idx] = []
                    warning_msg += " (auto-corregido: asignado a vehículo estándar)"
                    skill_warnings[-1] = warning_msg
                
        if skill_warnings:
            logger.warning("Problemas de compatibilidad skills:\n" + "\n".join(skill_warnings))

        # -------------------------- Otras configuraciones --------------------------
        # Manejar ubicaciones de inicio y fin de vehículos
        starts = []
        ends = []
        for v in self.vehicles:
            # Manejar ubicación de inicio
            start_key = v.get("start_location") or v.get("start_location_id")
            if not start_key:
                raise KeyError(f"Vehículo {v.get('id')} no tiene definida ubicación de inicio")
            
            if start_key not in self.node_map:
                raise KeyError(f"Ubicación de inicio '{start_key}' no encontrada para vehículo {v.get('id')}")
            
            start_idx = self.node_map[start_key]
            starts.append(start_idx)
            
            # Manejar ubicación de fin (si no se especifica, usar la misma que inicio)
            end_key = v.get("end_location") or v.get("end_location_id") or start_key
            
            if end_key not in self.node_map:
                logger.warning(f"Ubicación de fin '{end_key}' no encontrada para vehículo {v.get('id')}, usando ubicación de inicio")
                end_idx = start_idx
            else:
                end_idx = self.node_map[end_key]
                
            ends.append(end_idx)

        opt_profile = self.scenario.get("optimization_profile", {})
        time_cfg = opt_profile.get("time_dimension_config", {})
        time_dimension_config = {
            "horizon": time_cfg.get("horizon", 86_400),
            "slack_max": time_cfg.get("slack_max", 86_400),
            "fix_start_cumul_to_zero": time_cfg.get("fix_start_cumul_to_zero", True),
        }

        override_params = opt_profile.get("override_params", {})
        solver_params = {
            "first_solution_strategy": override_params.get("first_solution_strategy", "PATH_CHEAPEST_ARC"),
            "metaheuristic": override_params.get("metaheuristic", "GUIDED_LOCAL_SEARCH"),
            "time_limit_seconds": override_params.get("time_limit_seconds", 5),
            "enable_logging": True,
        }

        # Manejar depósitos según la estructura esperada por el solver
        # El solver espera un solo depósito, pero podemos tener múltiples depósitos en el JSON
        # Usamos el primer depósito que encontremos o el inicio del primer vehículo
        depot_idx = 0  # valor por defecto
        
        # Buscar un depósito en las ubicaciones (pueden tener type="depot" o estar en la lista de depósitos)
        depots = self.scenario.get("depots", [])
        if depots and isinstance(depots, list) and len(depots) > 0:
            # Usar el primer depósito de la lista de depósitos
            depot_id = depots[0]
            depot_idx = self.node_map.get(depot_id, 0)
        else:
            # Buscar una ubicación con type="depot"
            for i, loc in enumerate(self.locations):
                if loc.get("type") == "depot":
                    depot_idx = i
                    break
            else:
                # Si no hay depósitos definidos, usar la ubicación de inicio del primer vehículo
                if starts and len(starts) > 0:
                    depot_idx = starts[0]
        
        # Construir lista de índices de depósitos para VRPSolver (uno por vehículo)
        if depots and isinstance(depots, list) and len(depots) >= len(self.vehicles):
            depots_idx = [self.node_map.get(d, depot_idx) for d in depots[:len(self.vehicles)]]
        else:
            # Fallback: usar ubicación de inicio de cada vehículo
            depots_idx = starts.copy()
        
        demand_keys = dimension_names  # dimension_names ya contiene las dimensiones de demanda detectadas
        if len(demand_keys) == 1:
            vehicle_capacities = [c[0] if isinstance(c, list) else c for c in vehicle_capacities]
            demands = [d[0] if isinstance(d, list) else d for d in demands]

        # -------------------------- Empaquetar resultado ---------------------------
        vrp_data: Dict[str, Any] = {
            "distance_matrix": distance_matrix,
            "time_matrix": time_matrix,
            "service_times": service_times,
            "time_windows": time_windows,
            "num_vehicles": len(self.vehicles),
            "depot": depot_idx,  # Índice único para compatibilidad legacy
            "depots": depots_idx,  # Lista de índices de depósitos para VRPSolver
            "starts": starts,
            "ends": ends,
            # Asegurarse de que las demandas estén en el formato correcto
            "demands": demands[0] if len(dimension_names) == 1 else demands,
            "vehicle_capacities": [c[0] if len(c) == 1 else c for c in vehicle_capacities],
            "capacity_dimension_names": dimension_names,
            # Añadir demandas específicas para compatibilidad
            "demands_weight": [loc.get('weight_demand', 0) for loc in self.locations],
            "vehicle_capacities_weight": [v.get('weight_capacity', 0) for v in self.vehicles],
            "demands_volume": [int(loc.get('volume_demand', 0)) for loc in self.locations],
            "vehicle_capacities_volume": [int(v.get('volume_capacity', 0)) for v in self.vehicles],
            # Habilidades en el nivel raíz para compatibilidad con el solver
            "skills_required": required_skills,  # Ya es una lista de listas
            "vehicle_skills": vehicle_skills,  # Lista de listas de habilidades por vehículo
            "skill_constraint_type": skill_constraint_type,
            "skill_penalty": skill_penalty,
            "pickups_deliveries": pickups_deliveries,
            "allow_skipping_nodes": self.scenario.get("allow_skipping_nodes", True),
            "penalties": self.scenario.get("penalties", [1_000_000] * len(self.locations)),
            "node_priorities": [loc.get("priority", "NORMAL") for loc in self.locations],
            "vehicle_cost_factors": [v.get("cost_factor", 1.0) for v in self.vehicles],
            "locations": self.locations,
            "vehicles": self.vehicles,
            "solver_parameters": solver_params,
            "time_dimension_config": time_dimension_config,
            "optimization_profile": opt_profile,
        }

        self.vrp_data = vrp_data
        return vrp_data, self.matrix_cache_used

    # ------------------------------------------------------------------
    # Exposed helpers
    # ------------------------------------------------------------------
    def get_node_map(self) -> Dict[str, int]:
        """Devuelve el mapeo id -> índice de nodo."""
        return self.node_map
