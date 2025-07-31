"""solution_presenter.py
Presenta y valida la solución del VRP verificando las restricciones.
"""
import json

class JsonSolutionPresenter:
    """Presenta la solución del VRP en formato JSON verificando restricciones."""

    @staticmethod
    def format_time(seconds):
        if isinstance(seconds, (int, float)):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h:02d}:{m:02d}"
        return str(seconds)
        
    @classmethod
    def _get_unassigned_nodes(cls, raw_solution, locations):
        """Identifica los nodos no asignados en la solución."""
        all_nodes = set(range(len(locations)))
        visited_nodes = set()
        
        for route in raw_solution.get('routes', []):
            for node in route.get('route', []):
                node_index = next((i for i, loc in enumerate(locations) 
                                 if loc.get('id') == node.get('location_id')), None)
                if node_index is not None:
                    visited_nodes.add(node_index)
        
        unassigned_nodes = all_nodes - visited_nodes
        return [
            {
                'location_id': locations[i].get('id'),
                'location_name': locations[i].get('name'),
                'reason': 'No asignado'
            }
            for i in unassigned_nodes
            if i != 0  # No incluir el depósito si no fue asignado
        ]

    @classmethod
    def present(cls, raw_solution, request_data):
        """Procesa y presenta la solución del VRP."""
        # Obtener los datos del contenedor correcto
        # El request_data puede tener un campo 'data' o ser directamente los datos
        data = request_data.get('data', request_data)
        
        if not raw_solution or 'routes' not in raw_solution:
            return {
                "status": "error",
                "message": "No se encontró solución"
            }
            
        # Extraer datos importantes
        locations = data.get('locations', [])
        vehicles = data.get('vehicles', [])
        pickups_deliveries = data.get('pickups_deliveries', [])
        
        print(f"[DEBUG] Presenter - Vehículos disponibles: {len(vehicles)}")
        for i, v in enumerate(vehicles):
            print(f"[DEBUG] Vehículo {i}: ID={v.get('id')}, Peso={v.get('weight_capacity')}, Volumen={v.get('volume_capacity')}, Skills={v.get('skills')}")
        
        # Analizar demandas totales por clientes
        total_weight_demand = 0
        total_volume_demand = 0
        for loc in locations:
            if loc.get('type') != 'depot':  # No contar los depósitos
                weight_demand = loc.get('weight_demand', 0)
                volume_demand = loc.get('volume_demand', 0)
                # Solo sumar demandas positivas para evitar contar dos veces en pares pickup/delivery
                if weight_demand > 0:
                    total_weight_demand += weight_demand
                if volume_demand > 0:
                    total_volume_demand += volume_demand
        
        # Identificar nodos no asignados
        unassigned_nodes = cls._get_unassigned_nodes(raw_solution, locations)
        
        # Procesar rutas
        routes = []
        warnings = []
        capacity_warnings = []
        
        # Identificar nodos asignados
        assigned_nodes = set()
        # --- Validación pickup-delivery ---
        # Construir mapa de ubicación a (vehículo, posición en ruta)
        node_to_vehiclepos = {}
        for i, route_data in enumerate(raw_solution.get('routes', [])):
            vehicle_id = route_data.get('vehicle_id', '')
            route_nodes = route_data.get('route', [])
            for pos, node in enumerate(route_nodes):
                loc_id = node.get('location_id')
                # Solo agregar nodos que no sean depósito
                if loc_id and loc_id not in ['depot', 'deposit']:
                    node_to_vehiclepos[loc_id] = (vehicle_id, i, pos)
                    assigned_nodes.add(loc_id)
        # Validar cada par pickup-delivery
        for pd in pickups_deliveries:
            pickup_id = pd.get('pickup')
            delivery_id = pd.get('delivery')
            if not pickup_id or not delivery_id:
                warnings.append(f"Par pickup-delivery inválido: {pd}")
                continue
            pickup_info = node_to_vehiclepos.get(pickup_id)
            delivery_info = node_to_vehiclepos.get(delivery_id)
            if not pickup_info or not delivery_info:
                warnings.append(f"Par pickup-delivery no atendido: pickup={pickup_id}, delivery={delivery_id}")
                continue
            if pickup_info[0] != delivery_info[0]:
                warnings.append(f"Pickup y delivery de par {pickup_id}->{delivery_id} están en vehículos distintos: {pickup_info[0]} y {delivery_info[0]}")
            elif pickup_info[2] >= delivery_info[2]:
                warnings.append(f"Pickup {pickup_id} ocurre después de delivery {delivery_id} en la ruta del vehículo {pickup_info[0]}")
        
        for i, route_data in enumerate(raw_solution.get('routes', [])):
            # Obtener el ID del vehículo
            vehicle_id = route_data.get('vehicle_id', '')
            
            # Encontrar el vehículo en los datos de entrada
            vehicle = None
            for v in vehicles:
                if v.get('id') == vehicle_id:
                    vehicle = v
                    break
            
            if not vehicle:
                warnings.append(f"Vehículo no encontrado: {vehicle_id}")
                continue
                
            # Capacidades del vehículo
            weight_capacity = vehicle.get('weight_capacity', 0)
            volume_capacity = vehicle.get('volume_capacity', 0)
            vehicle_skills = set(vehicle.get('skills', []))
            
            # Extraer nodos de la ruta
            route_nodes = route_data.get('route', [])
            
            # Los nodos ya son objetos con información completa
            nodes = route_nodes
            if not nodes:
                continue
            
            # Calcular demandas acumuladas
            # Acumulación progresiva de demanda para verificar en cada nodo
            cumulative_weight = 0
            cumulative_volume = 0
            max_cumulative_weight = 0
            max_cumulative_volume = 0
            node_names = []
            missing_skills = []
            
            print(f"[DEBUG] Procesando ruta para vehículo {vehicle_id} con capacidad peso={weight_capacity}, volumen={volume_capacity}")
            print(f"[DEBUG] Nodos en ruta: {len(nodes)}")
            
            for node_data in nodes:
                # Los nodos ya son objetos con toda la información
                location_id = node_data.get('location_id')
                location_name = node_data.get('location_name')
                location_type = node_data.get('location_type')
                
                # Acumular demandas desde el objeto nodo
                weight_demand = 0
                volume_demand = 0
                
                # Buscar la ubicación original para obtener las demandas
                for loc in locations:
                    if loc.get('id') == location_id:
                        weight_demand = loc.get('weight_demand', 0)
                        volume_demand = loc.get('volume_demand', 0)
                        break
                
                cumulative_weight += weight_demand
                cumulative_volume += volume_demand
                
                # Seguir el máximo acumulado (para pickup-delivery puede variar)
                max_cumulative_weight = max(max_cumulative_weight, cumulative_weight)
                max_cumulative_volume = max(max_cumulative_volume, cumulative_volume)
                
                print(f"[DEBUG] Nodo {location_id} ({location_name}): Demanda peso={weight_demand}, Acumulado={cumulative_weight}/{weight_capacity}")
                print(f"[DEBUG] Nodo {location_id} ({location_name}): Demanda volumen={volume_demand}, Acumulado={cumulative_volume}/{volume_capacity}")
                
                # Verificar skills requeridos
                for loc in locations:
                    if loc.get('id') == location_id:
                        required_skills = set(loc.get('required_skills', []))
                        if required_skills and not required_skills.issubset(vehicle_skills):
                            missing = required_skills - vehicle_skills
                            missing_skills.append(f"Nodo {location_id} requiere: {missing}")
                        break
                
                # Guardar nombre del nodo
                node_names.append(location_name or location_id)
                
                # Verificar si se excede la capacidad en cualquier punto
                if cumulative_weight > weight_capacity:
                    msg = f"Exceso de peso en vehículo {vehicle_id} ({vehicle.get('name')}) después de nodo {location_id}: {cumulative_weight} > {weight_capacity}"
                    capacity_warnings.append(msg)
                    print(f"[ERROR] {msg}")
                        
                if cumulative_volume > volume_capacity:
                    msg = f"Exceso de volumen en vehículo {vehicle_id} ({vehicle.get('name')}) después de nodo {location_id}: {cumulative_volume} > {volume_capacity}"
                    capacity_warnings.append(msg)
                    print(f"[ERROR] {msg}")
            
            # Añadir advertencias de habilidades requeridas
            if missing_skills:
                warnings.append(f"Vehículo {vehicle_id} no tiene todas las habilidades requeridas: {missing_skills}")
            
            # Crear ruta procesada
            route_info = {
                "vehicle_id": vehicle_id,
                "vehicle_name": vehicle.get('name', f"Vehículo {i}"),
                "route": nodes,
                "node_names": node_names,
                "total_distance": route_data.get('total_distance', 0),
                "weight_capacity": weight_capacity,
                "max_weight": max_cumulative_weight,
                "volume_capacity": volume_capacity,
                "max_volume": max_cumulative_volume
            }
            

            routes.append(route_info)
        
        # Identificar nodos no asignados (solo clientes, no depósitos)
        unassigned_nodes = []
        for loc in locations:
            loc_id = loc.get('id')
            loc_type = loc.get('type', '').lower()
            
            # Excluir explícitamente depósitos
            if loc_type in ['depot', 'deposit']:
                continue
                
            # Solo considerar nodos que no estén ya asignados
            if (loc_id not in assigned_nodes and
                not any(pd.get('pickup') == loc_id or 
                       pd.get('delivery') == loc_id 
                       for pd in pickups_deliveries)):
                unassigned_nodes.append({
                    'id': loc_id,
                    'name': loc.get('name', loc_id),
                    'reason': 'No asignado',
                    'weight_demand': loc.get('weight_demand', 0),
                    'volume_demand': loc.get('volume_demand', 0),
                    'time_window': f"{cls.format_time(loc.get('time_window_start', 0))} - {cls.format_time(loc.get('time_window_end', 0))}"
                })
        
        # Analizar vehículos no utilizados
        unused_vehicles = []
        used_vehicle_ids = {route.get('vehicle_id') for route in raw_solution.get('routes', [])}
        
        for v in vehicles:
            v_id = v.get('id')
            if v_id not in used_vehicle_ids:
                unused_vehicles.append({
                    "id": v_id,
                    "name": v.get('name', v_id),
                    "weight_capacity": v.get('weight_capacity', 0),
                    "volume_capacity": v.get('volume_capacity', 0),
                    "skills": v.get('skills', [])
                })
        
        # La validación de capacidad ya se realizó previamente
        # No es necesario verificar vehículos alternativos ya que las rutas ya respetan las capacidades
        # Formatear tiempos a legible (HH:MM:SS)
        def format_seconds(seconds):
            if not isinstance(seconds, (int, float)):
                return str(seconds)
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        # Procesar rutas para el nuevo formato
        formatted_routes = []
        total_duration = 0
        total_stops = 0

        for route in routes:
            stops = []
            current_weight = 0
            current_volume = 0
            route_distance = 0
            
            for node in route.get('route', []):
                # Calcular tiempo de salida (llegada + tiempo de servicio)
                arrival = node.get('arrival_time', 0)
                service_time = node.get('service_time', 0)
                departure = arrival + service_time
                
                # Actualizar cargas y registrar nodo asignado
                location_id = node.get('location_id')
                if location_id and location_id not in ['depot', 'deposit']:
                    assigned_nodes.add(location_id)
                
                for loc in locations:
                    if loc.get('id') == location_id:
                        current_weight += loc.get('weight_demand', 0)
                        current_volume += loc.get('volume_demand', 0)
                        break
                
                stops.append({
                    "location": {
                        "id": location_id,
                        "name": node.get('location_name', ''),
                        "type": node.get('location_type', ''),
                        "coords": node.get('coords', [])
                    },
                    "arrival": format_seconds(arrival),
                    "departure": format_seconds(departure),
                    "load": {
                        "weight": current_weight,
                        "volume": current_volume
                    }
                })
                
                # Acumular estadísticas
                total_stops += 1
                route_distance = max(route_distance, node.get('distance', 0))
            
            # Calcular duración y distancia total de la ruta
            route_duration = 0
            route_total_distance = 0
            
            # Si hay paradas, calcular duración y distancia
            if stops:
                # Convertir HH:MM:SS a segundos para calcular la duración
                start_time = sum(int(x) * 60 ** (2 - i) for i, x in enumerate(stops[0]['arrival'].split(":")))
                end_time = sum(int(x) * 60 ** (2 - i) for i, x in enumerate(stops[-1]['departure'].split(":")))
                route_duration = end_time - start_time
                
                # Calcular distancia total sumando distancias entre nodos consecutivos
                route_nodes = route.get('route', [])
                route_total_distance = 0
                
                # Si la ruta tiene nodos con coordenadas, calcular distancias euclidianas
                if len(route_nodes) > 1 and 'coords' in route_nodes[0]:
                    import math
                    for i in range(len(route_nodes) - 1):
                        current = route_nodes[i]
                        next_node = route_nodes[i + 1]
                        
                        if 'coords' in current and 'coords' in next_node:
                            from_lat, from_lon = current['coords']
                            to_lat, to_lon = next_node['coords']
                            
                            # Fórmula de distancia euclidiana simplificada (aproximación en metros)
                            dx = (to_lon - from_lon) * 111.32 * 1000 * math.cos(math.radians((from_lat + to_lat) / 2))
                            dy = (to_lat - from_lat) * 111.32 * 1000
                            route_total_distance += math.sqrt(dx*dx + dy*dy)
                
                # Si no se pudo calcular la distancia, usar el valor del nodo
                if route_total_distance <= 0:
                    route_total_distance = max(node.get('distance', 0) for node in route_nodes if 'distance' in node)
            
            # Calcular métricas de capacidad
            max_weight = max((stop['load']['weight'] for stop in stops), default=0)
            max_volume = max((stop['load']['volume'] for stop in stops), default=0)
            
            # Agregar ruta formateada
            formatted_routes.append({
                "vehicle": {
                    "id": route.get('vehicle_id', ''),
                    "name": route.get('vehicle_name', '')
                },
                "stops": stops,
                "metrics": {
                    "total_distance": route_total_distance,  # en metros
                    "total_duration": route_duration,        # en segundos
                    "total_stops": len(stops),
                    "max_weight": max_weight,
                    "max_volume": max_volume
                }
                })
        
        # Respuesta final mejorada
        return {
            "status": "success",
            "message": "Optimización completada",
            "warnings": warnings + capacity_warnings,
            "excluded_nodes": [],
            "used_cache": raw_solution.get('used_cache', False),
            "statistics": {
                "vehicles_used": len([r for r in formatted_routes if r.get('stops', [])]),
                "vehicles_available": len(vehicles),
                "total_demand_weight": total_weight_demand,
                "total_demand_volume": total_volume_demand,
                "nodes_assigned": len(assigned_nodes),
                "nodes_unassigned": len(unassigned_nodes)
            },
            "routes": formatted_routes,
            "unused_vehicles": unused_vehicles,
            "unassigned_nodes": unassigned_nodes
        }


# Para mantener compatibilidad con el código existente
def present_solution(raw_solution, request_data):
    return JsonSolutionPresenter.present(raw_solution, request_data)