"""
Módulo para validar la factibilidad de un problema VRP antes de resolverlo.
Contiene la clase VrpValidator que acumula todos los errores de validación.
"""
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class VrpValidator:
    """
    Valida la estructura y factibilidad de los datos de un problema VRP.
    Filtra nodos que no pueden ser atendidos y genera un informe detallado.
    """
    def __init__(self, data: Dict[str, Any]):
        self.original_data = data
        self.data = self._deep_copy(data)
        self.errors = []
        self.warnings = []
        self.excluded_nodes = []  # Nodos excluidos y razón
        self.locations = self.data.get('locations', [])
        self.vehicles = self.data.get('vehicles', [])
        self.location_ids = {loc.get('id') for loc in self.locations if loc.get('id')}
        
    def _deep_copy(self, data: Any) -> Any:
        """Crea una copia profunda de los datos."""
        if isinstance(data, dict):
            return {k: self._deep_copy(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._deep_copy(item) for item in data]
        else:
            return data

    def validate(self) -> Dict[str, Any]:
        """
        Ejecuta todas las validaciones y devuelve los datos limpios.
        
        Returns:
            Dict con:
            - 'cleaned_data': Datos listos para el solver
            - 'excluded_nodes': Lista de nodos excluidos con razones
            - 'warnings': Advertencias de validación
            - 'errors': Errores críticos (si los hay)
        """
        # Validación en proceso
        
        # Validaciones iniciales de estructura
        self._validate_structure()
        if self.errors:
            return self._build_validation_result()
        
        # Identificar nodos no atendibles
        self._identify_unserviceable_nodes()
        
        # Filtrar nodos no atendibles
        if self.excluded_nodes:
            self._filter_unserviceable_nodes()
            
        # Validar dimensiones de capacidad por demanda/capacidad dinámicas
        self._validate_capacity_dimensions()
        # Validar pares pick & delivery
        self._validate_pick_delivery_pairs()
        # Validar referencias internas pickup/delivery
        self._validate_internal_pickup_delivery_refs()

        # Validar el resto de las restricciones
        self._validate_capacity()
        self._validate_time_windows()
        self._validate_depots()
        
        # Si hay errores críticos, no continuar
        if self.errors:
            return self._build_validation_result()
            
        # Validación completada
        return self._build_validation_result()
        
    def _identify_unserviceable_nodes(self):
        """Identifica nodos que no pueden ser atendidos por ningún vehículo."""
        if not self.locations or not self.vehicles:
            return
            
        # Mapear vehículos por habilidades y capacidades
        vehicle_capacities = {}
        vehicle_skills = {}
        
        for vehicle in self.vehicles:
            vid = vehicle.get('id', 'unknown')
            vehicle_capacities[vid] = {
                'capacity': vehicle.get('capacity', float('inf')),
                'weight_capacity': vehicle.get('weight_capacity', float('inf')),
                'volume_capacity': vehicle.get('volume_capacity', float('inf'))
            }
            vehicle_skills[vid] = set(vehicle.get('skills', []))
        
        # Verificar cada ubicación
        for loc in self.locations:
            loc_id = loc.get('id')
            if not loc_id or loc.get('type') == 'depot':
                continue
                
            reasons = []
            
            # Verificar habilidades requeridas
            required_skills = set(loc.get('required_skills', []))
            if required_skills:
                # Verificar si hay al menos un vehículo que tenga TODAS las habilidades requeridas
                has_skills = any(
                    required_skills.issubset(skills)
                    for skills in vehicle_skills.values()
                )
                if not has_skills:
                    # Obtener lista de habilidades disponibles en la flota
                    available_skills = set()
                    for skills in vehicle_skills.values():
                        available_skills.update(skills)
                    
                    # Identificar habilidades faltantes
                    missing_skills = required_skills - available_skills
                    if missing_skills:
                        reasons.append(f"no hay vehículos con las habilidades requeridas: {sorted(required_skills)} (faltan: {sorted(missing_skills)})")
                    else:
                        # Hay vehículos con habilidades individuales, pero ninguno con la combinación exacta
                        reasons.append(f"no hay vehículos con la combinación exacta de habilidades: {sorted(required_skills)}")
            
            # Verificar capacidades solo si se permite omitir nodos
            # (para evitar asignaciones imposibles)
            if self.data.get('allow_skipping_nodes', False):
                demand = loc.get('demand', 0)
                weight_demand = loc.get('weight_demand', 0)
                volume_demand = loc.get('volume_demand', 0)
                
                if demand > 0:
                    max_capacity = max(v['capacity'] for v in vehicle_capacities.values())
                    if demand > max_capacity:
                        reasons.append(f"demanda ({demand}) excede la capacidad máxima de cualquier vehículo ({max_capacity})")
                        
                if weight_demand > 0:
                    max_weight = max(v['weight_capacity'] for v in vehicle_capacities.values())
                    if weight_demand > max_weight:
                        reasons.append(f"peso ({weight_demand}) excede la capacidad máxima de cualquier vehículo ({max_weight})")
                        
                if volume_demand > 0:
                    max_volume = max(v['volume_capacity'] for v in vehicle_capacities.values())
                    if volume_demand > max_volume:
                        reasons.append(f"volumen ({volume_demand}) excede la capacidad máxima de cualquier vehículo ({max_volume})")
            
            # Si hay razones para excluir, agregar a la lista
            if reasons:
                self.excluded_nodes.append({
                    'id': loc_id,
                    'name': loc.get('name', 'Sin nombre'),
                    'reasons': reasons
                })

    def _validate_structure(self):
        """Valida la presencia de claves y estructuras básicas."""
        if not isinstance(self.locations, list) or not self.locations:
            self.errors.append("El JSON debe contener una lista 'locations' no vacía.")
        if not isinstance(self.vehicles, list) or not self.vehicles:
            self.errors.append("El JSON debe contener una lista 'vehicles' no vacía.")

    def _validate_capacity(self):
        """Valida que la capacidad total sea suficiente para la demanda total."""
        total_capacity = {}
        capacity_dims = set()
        for v in self.vehicles:
            for key in v.keys():
                if key.endswith('_capacity'):
                    dim_name = key.replace('_capacity', '')
                    capacity_dims.add(dim_name)
                    total_capacity[dim_name] = total_capacity.get(dim_name, 0) + v[key]

        total_demand = {}
        for loc in self.locations:
            for key in loc.keys():
                if key.endswith('_demand'):
                    dim_name = key.replace('_demand', '')
                    if dim_name in capacity_dims:
                        total_demand[dim_name] = total_demand.get(dim_name, 0) + loc[key]

        # Si se permiten omitir nodos, la flota puede no cubrir el 100 % de la demanda.
        if self.data.get('allow_skipping_nodes'):
            return

        for dim, demand_val in total_demand.items():
            capacity_val = total_capacity.get(dim, 0)
            if demand_val > capacity_val:
                self.errors.append(
                    f"La demanda total para '{dim}' ({demand_val}) excede la capacidad total ({capacity_val}).")

    def _validate_time_windows(self):
        """Valida que las ventanas de tiempo (inicio <= fin) sean consistentes."""
        for item in self.locations + self.vehicles:
            if 'time_window' in item:
                tw = item['time_window']
                item_id = item.get('id', 'N/A')
                if not isinstance(tw, list) or len(tw) != 2:
                    self.errors.append(f"Ventana de tiempo para '{item_id}' debe ser una lista de 2 elementos.")
                    continue
                if not all(isinstance(t, (int, float)) for t in tw):
                    self.errors.append(f"Valores de ventana de tiempo para '{item_id}' deben ser numéricos.")
                    continue
                if tw[0] > tw[1]:
                    self.errors.append(f"Ventana de tiempo para '{item_id}' es inválida (inicio > fin): {tw}")


    # ----------------------------------------------------------------------
    # Validación de pares pickup-delivery
    # ----------------------------------------------------------------------
    def _validate_pick_delivery_pairs(self):
        """Valida que los pares pickup-delivery sean válidos y factibles."""
        if not self.data.get('pickup_delivery_pairs') and not self.data.get('pickups_deliveries'):
            return  # No hay pares que validar
            
        # Obtener la lista de pares (considerar ambos formatos posibles)
        pairs = self.data.get('pickup_delivery_pairs', []) or self.data.get('pickups_deliveries', [])
        
        if not pairs:
            return
            
        logger.debug(f"Validando {len(pairs)} pares pickup-delivery")
        
        # Mapeo de IDs de ubicación a índices
        location_id_to_index = {}
        for i, loc in enumerate(self.locations):
            if 'id' in loc:
                location_id_to_index[loc['id']] = i
        
        # Validar cada par
        for i, pair in enumerate(pairs):
            try:
                # Extraer pickup y delivery según el formato
                pickup = None
                delivery = None
                
                if isinstance(pair, dict):
                    pickup = pair.get('pickup')
                    delivery = pair.get('delivery')
                elif isinstance(pair, (list, tuple)) and len(pair) == 2:
                    pickup, delivery = pair
                    
                # Validar que ambos estén definidos
                if pickup is None or delivery is None:
                    self.errors.append(f"Par pickup-delivery #{i} incompleto: pickup={pickup}, delivery={delivery}")
                    continue
                    
                # Validar que ambos sean enteros o estén en el mapeo de ubicaciones
                pickup_idx = pickup
                delivery_idx = delivery
                
                # Si son IDs de ubicación, convertirlos a índices
                if not isinstance(pickup, int):
                    if pickup in location_id_to_index:
                        pickup_idx = location_id_to_index[pickup]
                    else:
                        self.errors.append(f"ID de pickup '{pickup}' no encontrado en la lista de ubicaciones")
                        continue
                        
                if not isinstance(delivery, int):
                    if delivery in location_id_to_index:
                        delivery_idx = location_id_to_index[delivery]
                    else:
                        self.errors.append(f"ID de delivery '{delivery}' no encontrado en la lista de ubicaciones")
                        continue
                
                # Validar que los índices estén dentro del rango
                if pickup_idx < 0 or pickup_idx >= len(self.locations):
                    self.errors.append(f"Índice de pickup {pickup_idx} fuera de rango (0-{len(self.locations)-1})")
                    continue
                    
                if delivery_idx < 0 or delivery_idx >= len(self.locations):
                    self.errors.append(f"Índice de delivery {delivery_idx} fuera de rango (0-{len(self.locations)-1})")
                    continue
                    
                # Validar que pickup y delivery no sean el mismo nodo
                if pickup_idx == delivery_idx:
                    self.errors.append(f"Pickup y delivery no pueden ser el mismo nodo: {pickup_idx}")
                    continue
                    
                # Validar que las demandas sean opuestas para garantizar balance
                pickup_location = self.locations[pickup_idx]
                delivery_location = self.locations[delivery_idx]
                
                # Las demandas deben ser opuestas (p.ej. pickup +N, delivery -N)
                pickup_demand = pickup_location.get('demand', 0)
                delivery_demand = delivery_location.get('demand', 0)
                
                if pickup_demand <= 0 and not pickup_location.get('type') == 'pickup':
                    self.warnings.append(f"Nodo pickup {pickup_idx} tiene demanda no positiva: {pickup_demand}")
                    
                if delivery_demand >= 0 and not delivery_location.get('type') == 'delivery':
                    self.warnings.append(f"Nodo delivery {delivery_idx} tiene demanda no negativa: {delivery_demand}")
                    
                # Si hay demandas de weight o volume, validarlas también
                for prefix in ['weight', 'volume']:
                    p_key = f"{prefix}_demand"
                    d_key = f"{prefix}_demand"
                    
                    if p_key in pickup_location or d_key in delivery_location:
                        p_val = pickup_location.get(p_key, 0)
                        d_val = delivery_location.get(d_key, 0)
                        
                        if p_val <= 0 and not pickup_location.get('type') == 'pickup':
                            self.warnings.append(f"Nodo pickup {pickup_idx} tiene {p_key} no positiva: {p_val}")
                            
                        if d_val >= 0 and not delivery_location.get('type') == 'delivery':
                            self.warnings.append(f"Nodo delivery {delivery_idx} tiene {d_key} no negativa: {d_val}")
                
                # Todo correcto para este par
                logger.debug(f"Par pickup-delivery #{i}: pickup={pickup_idx}, delivery={delivery_idx} validado correctamente")
                
            except Exception as e:
                self.errors.append(f"Error al validar par pickup-delivery #{i}: {e}")
        
        if self.errors:
            print(f"[WARNING] Se encontraron {len(self.errors)} errores en pares pickup-delivery")
        else:
            print(f"[INFO] Validación de pares pickup-delivery completada sin errores")
    
    # ----------------------------------------------------------------------
    # NUEVO: Validación de dimensiones de capacidad genéricas
    # ----------------------------------------------------------------------
    def _validate_capacity_dimensions(self):
        """Verifica que cada demanda *_demand tenga al menos un vehículo con capacidad suficiente."""
        if not self.locations or not self.vehicles:
            return

        # Descubrir todas las claves de demanda
        demand_keys = {k for loc in self.locations for k in loc if k.endswith('_demand') or k == 'demand'}
        if not demand_keys:
            return

        for dem_key in sorted(demand_keys):
            cap_key = 'capacity' if dem_key == 'demand' else dem_key.replace('_demand', '_capacity')
            max_cap = max((v.get(cap_key, 0) for v in self.vehicles), default=0)
            if max_cap <= 0:
                self.errors.append(
                    f"No existe capacidad '{cap_key}' en ningún vehículo mientras hay demandas '{dem_key}'.")
                continue
            for loc in self.locations:
                if loc.get('type') == 'depot':
                    continue
                val = loc.get(dem_key, 0)
                if val > max_cap:
                    self.excluded_nodes.append({
                        'id': loc.get('id'),
                        'name': loc.get('name', ''),
                        'reasons': [f"{dem_key}={val} excede la capacidad máxima {cap_key}={max_cap}"]
                    })
                    self.warnings.append(
                        f"{loc.get('id')} excluido: {dem_key}={val} excede {cap_key}={max_cap}")

    # ----------------------------------------------------------------------
    # Validación de pares pick & delivery por referencias cruzadas
    # ----------------------------------------------------------------------
    def _validate_internal_pickup_delivery_refs(self):
        """Valida coherencia básica de nodos pick & delivery cuando usan referencias cruzadas."""
        id_map = {loc.get('id'): loc for loc in self.locations if loc.get('id')}

        for loc in self.locations:
            pid = loc.get('pickup_id')
            did = loc.get('delivery_id')

            # Caso 1: Nodo es pickup
            if pid:
                if pid not in id_map:
                    self.errors.append(f"Ubicación '{loc.get('id')}' referencia pickup_id='{pid}' inexistente.")
                    continue
                other = id_map[pid]
                # Revisar demandas opuestas
                self._check_opposite_demands(loc, other)
            # Caso 2: Nodo es delivery
            if did:
                if did not in id_map:
                    self.errors.append(f"Ubicación '{loc.get('id')}' referencia delivery_id='{did}' inexistente.")
                    continue
                other = id_map[did]
                self._check_opposite_demands(other, loc)

    def _check_opposite_demands(self, pick_loc: dict, deliv_loc: dict):
        """Verifica que las demandas entre pick y delivery sean opuestas (suma cero)."""
        demand_keys = {k for k in pick_loc if k.endswith('_demand') or k == 'demand'}
        for dem in demand_keys:
            pick_val = pick_loc.get(dem, 0)
            deliv_val = deliv_loc.get(dem, 0)
            if pick_val == 0 and deliv_val == 0:
                continue
            if pick_val + deliv_val != 0:
                self.warnings.append(
                    f"Demandas inconsistentes entre pick '{pick_loc.get('id')}' y delivery '{deliv_loc.get('id')}' para '{dem}': {pick_val}+{deliv_val} != 0")

    def _validate_depots(self):
        """Comprueba que depósitos e IDs de inicio/fin existan en locations."""
        if not self.locations:
            return
            
        all_ids = {loc.get('id') for loc in self.locations if loc.get('id')}
        
        # Validar depósitos
        for depot_id in self.data.get('depots', []):
            if depot_id not in all_ids:
                self.errors.append(f"El depósito '{depot_id}' no se encuentra en la lista de ubicaciones.")
        
        # Validar inicios y finales de vehículos
        for v in self.vehicles:
            vid = v.get('id', 'N/A')
            for key in ('start_location', 'start_location_id', 'end_location', 'end_location_id'):
                loc_id = v.get(key)
                if loc_id and loc_id not in all_ids:
                    self.errors.append(f"Vehículo '{vid}' referencia '{key}'='{loc_id}' inexistente.")

    def _filter_unserviceable_nodes(self):
        """Filtra los nodos no atendibles de los datos."""
        if not self.excluded_nodes:
            return
            
        excluded_ids = {node['id'] for node in self.excluded_nodes}
        
        # Filtrar ubicaciones
        self.locations = [
            loc for loc in self.locations
            if loc.get('id') not in excluded_ids or loc.get('type') == 'depot'
        ]
        
        # Actualizar referencias en los datos
        self.data['locations'] = self.locations
        
        # Actualizar matrices si existen
        if 'distance_matrix' in self.data and self.data['distance_matrix']:
            self._filter_matrices(excluded_ids)
    
    def _filter_matrices(self, excluded_ids: set):
        """Filtra las filas/columnas correspondientes a nodos excluidos."""
        # Obtener índices de los nodos a mantener
        keep_indices = [
            i for i, loc in enumerate(self.original_data.get('locations', []))
            if loc.get('id') not in excluded_ids or loc.get('type') == 'depot'
        ]
        
        # Filtrar matrices
        for matrix_type in ['distance_matrix', 'time_matrix']:
            if matrix_type in self.data and self.data[matrix_type]:
                # Filtrar filas
                filtered_matrix = [
                    [row[j] for j in keep_indices]
                    for i, row in enumerate(self.original_data[matrix_type])
                    if i in keep_indices
                ]
                self.data[matrix_type] = filtered_matrix
    


    def _build_validation_result(self) -> Dict[str, Any]:
        """Construye el resultado de la validación."""
        return {
            'cleaned_data': self.data if not self.errors else None,
            'excluded_nodes': self.excluded_nodes,
            'warnings': self.warnings,
            'errors': self.errors.copy(),
            'is_valid': not bool(self.errors)
        }


def validate_vrp(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida y limpia los datos de un problema VRP.
    
    Args:
        data: Diccionario con los datos del problema VRP.
        
    Returns:
        Dict con:
        - 'cleaned_data': Datos listos para el solver (o None si hay errores)
        - 'excluded_nodes': Lista de nodos excluidos con razones
        - 'warnings': Advertencias de validación
        - 'errors': Errores críticos (si los hay)
        - 'is_valid': Si los datos son válidos para el solver
    """
    validator = VrpValidator(data)
    return validator.validate()


# --- Nuevas funciones de validación para compatibilidad con main.py ---

def validate_json(data: Dict[str, Any]) -> bool:
    """Valida la estructura básica del JSON."""
    validator = VrpValidator(data)
    validator._validate_structure()
    validator._validate_depots()
    return len(validator.errors) == 0
