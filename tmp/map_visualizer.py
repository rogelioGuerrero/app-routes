"""
map_visualizer.py

Módulo para visualizar las rutas del VRP en un mapa interactivo usando folium.
"""
import os
import json
import webbrowser
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import folium
from folium import plugins
import polyline
from branca.element import Figure

class MapVisualizer:
    """Clase para visualizar rutas de VRP en un mapa interactivo usando folium."""
    
    # Colores para diferentes tipos de ubicaciones
    LOCATION_COLORS = {
        'depot': 'red',
        'pickup': 'blue',
        'delivery': 'green',
        'default': 'gray'
    }
    
    # Colores para diferentes vehículos
    VEHICLE_COLORS = [
        'blue', 'green', 'red', 'purple', 'orange', 'darkred',
        'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
        'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray',
        'black', 'lightgray'
    ]
    
    def __init__(self, output_dir: str = "output"):
        """Inicializa el visualizador con el directorio de salida.
        
        Args:
            output_dir: Directorio donde se guardarán los mapas generados.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _get_coordinates_center(self, locations: List[Dict]) -> Tuple[float, float]:
        """Calcula el centro del mapa basado en las coordenadas de las ubicaciones."""
        if not locations:
            return 19.4326, -99.1332  # CDMX por defecto
            
        lats = [loc['coords'][1] for loc in locations if 'coords' in loc and len(loc['coords']) >= 2]
        lons = [loc['coords'][0] for loc in locations if 'coords' in loc and len(loc['coords']) >= 2]
        
        if not lats or not lons:
            return 19.4326, -99.1332  # CDMX por defecto
            
        return sum(lats)/len(lats), sum(lons)/len(lons)
    
    def _format_time(self, seconds: int) -> str:
        """Formatea segundos a formato HH:MM:SS."""
        if seconds is None:
            return "N/A"
        return str(timedelta(seconds=seconds)).split('.')[0]  # Eliminar microsegundos
    
    def _normalize_route(self, route: Dict) -> Dict:
        """Normaliza la estructura de una ruta para asegurar consistencia."""
        normalized = route.copy()
        
        # Asegurar que exista la clave 'stops'
        if 'stops' not in normalized:
            if 'route' in normalized:
                normalized['stops'] = normalized.pop('route')
            else:
                normalized['stops'] = []
        
        # Normalizar cada parada
        for stop in normalized['stops']:
            # Asegurar que exista location_id o location
            if 'location_id' not in stop and 'location' in stop:
                stop['location_id'] = stop['location']
            
            # Asegurar que existan los campos básicos
            stop.setdefault('arrival_time', 0)
            stop.setdefault('departure_time', 0)
            stop.setdefault('service_time', 0)
        
        return normalized
    
    def _add_route_to_map(self, m: folium.Map, route: Dict, locations_dict: Dict, 
                         color: str, vehicle_id: str) -> None:
        """Agrega una ruta al mapa.
        
        Args:
            m: Mapa de folium donde se agregará la ruta
            route: Diccionario con la información de la ruta
            locations_dict: Diccionario con la información de las ubicaciones
            color: Color a usar para la ruta
            vehicle_id: Identificador del vehículo
        """
        # Normalizar la ruta
        route = self._normalize_route(route)
        
        if not route.get('stops'):
            print(f"[ADVERTENCIA] La ruta del vehículo {vehicle_id} no tiene paradas")
            return
            
        route_coords = []
        
        # Procesar cada parada en la ruta
        for i, stop in enumerate(route['stops']):
            location_id = stop.get('location_id')
            
            # Si no hay location_id, intentar con location
            if not location_id:
                location_id = stop.get('location')
            
            # Si aún no hay location_id, saltar esta parada
            if not location_id:
                print(f"[ADVERTENCIA] Parada {i} en la ruta {vehicle_id} no tiene identificador de ubicación")
                continue
                
            # Buscar la ubicación en el diccionario
            loc = locations_dict.get(location_id)
            if not loc:
                print(f"[ADVERTENCIA] Ubicación no encontrada: {location_id}")
                continue
                
            if 'coords' not in loc or len(loc['coords']) < 2:
                print(f"[ADVERTENCIA] Coordenadas no válidas para la ubicación {location_id}")
                continue
                
            try:
                # Obtener coordenadas [lat, lng] para folium
                lat, lng = float(loc['coords'][1]), float(loc['coords'][0])
                route_coords.append((lat, lng))
                
                # Determinar el tipo de marcador
                location_type = loc.get('type', 'default')
                icon_color = self.LOCATION_COLORS.get(location_type, self.LOCATION_COLORS['default'])
                
                # Crear popup con información
                popup_html = f"""
                <div style="width: 250px">
                    <h4>{loc.get('name', 'Ubicación')}</h4>
                    <p><b>ID:</b> {location_id}</p>
                    <p><b>Tipo:</b> {location_type.capitalize()}</p>
                    <p><b>Vehículo:</b> {vehicle_id}</p>
                    <p><b>Parada:</b> {i+1} de {len(route['stops'])}</p>
                    <p><b>Llegada:</b> {self._format_time(stop.get('arrival_time'))}</p>
                    <p><b>Salida:</b> {self._format_time(stop.get('departure_time'))}</p>
                    <p><b>Servicio:</b> {self._format_time(stop.get('service_time'))}</p>
                </div>
                """
                
                # Agregar marcador
                folium.Marker(
                    location=[lat, lng],
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color=icon_color, icon='info-sign' if location_type == 'depot' else 'map-marker'),
                    tooltip=f"{loc.get('name', 'Ubicación')} ({location_type})"
                ).add_to(m)
                
            except (ValueError, IndexError, TypeError) as e:
                print(f"[ERROR] Error procesando la ubicación {location_id}: {str(e)}")
                continue
        
        # Agregar la línea de la ruta si hay al menos 2 puntos
        if len(route_coords) > 1:
            try:
                folium.PolyLine(
                    route_coords,
                    color=color,
                    weight=3,
                    opacity=0.8,
                    popup=f"Ruta del vehículo {vehicle_id}",
                    tooltip=f"Ruta {vehicle_id}"
                ).add_to(m)
            except Exception as e:
                print(f"[ERROR] Error dibujando la ruta del vehículo {vehicle_id}: {str(e)}")
    
    def _normalize_solution(self, solution: Dict) -> Dict:
        """Normaliza la estructura de la solución para asegurar consistencia."""
        normalized = solution.copy()
        
        # Asegurar que exista la clave 'routes'
        if 'routes' not in normalized:
            if 'vehicles' in normalized:
                normalized['routes'] = normalized.pop('vehicles')
            else:
                normalized['routes'] = []
        
        # Normalizar cada ruta
        normalized['routes'] = [self._normalize_route(route) for route in normalized.get('routes', [])]
        
        return normalized
    
    def _normalize_vrp_data(self, vrp_data: Dict) -> Dict:
        """Normaliza los datos de entrada del VRP."""
        normalized = vrp_data.copy()
        
        # Asegurar que exista la clave 'locations'
        if 'locations' not in normalized:
            normalized['locations'] = []
        
        # Normalizar cada ubicación
        for loc in normalized['locations']:
            # Asegurar que exista 'id'
            if 'id' not in loc and 'location_id' in loc:
                loc['id'] = loc['location_id']
            
            # Asegurar que exista 'coords' con el formato correcto
            if 'coords' not in loc:
                if 'lat' in loc and 'lng' in loc:
                    loc['coords'] = [loc['lng'], loc['lat']]
                elif 'latitude' in loc and 'longitude' in loc:
                    loc['coords'] = [loc['longitude'], loc['latitude']]
                else:
                    print(f"[ADVERTENCIA] Ubicación sin coordenadas: {loc.get('id', 'desconocido')}")
        
        return normalized
    
    def plot_routes(self, vrp_data: Dict[str, Any], solution: Dict[str, Any], 
                   output_file: Optional[str] = None) -> str:
        """Genera un mapa interactivo con las rutas.
        
        Args:
            vrp_data: Datos originales del VRP.
            solution: Solución generada por el solver.
            output_file: Nombre del archivo de salida. Si es None, se genera uno automáticamente.
            
        Returns:
            Ruta al archivo HTML generado.
            
        Raises:
            ValueError: Si los datos de entrada no son válidos.
        """
        if not vrp_data or not solution:
            raise ValueError("Se requieren tanto los datos del VRP como la solución")
        
        try:
            # Normalizar los datos de entrada
            vrp_data = self._normalize_vrp_data(vrp_data)
            solution = self._normalize_solution(solution)
            
            # Generar nombre de archivo si no se proporciona
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(self.output_dir, f"vrp_routes_{timestamp}.html")
            
            # Crear un diccionario de ubicaciones para acceso rápido
            locations_dict = {}
            for loc in vrp_data.get('locations', []):
                if 'id' in loc:
                    locations_dict[loc['id']] = loc
            
            if not locations_dict:
                print("[ADVERTENCIA] No se encontraron ubicaciones válidas en los datos de entrada")
            
            # Calcular el centro del mapa
            center_lat, center_lng = self._get_coordinates_center(list(locations_dict.values()))
            
            # Crear el mapa con folium
            m = folium.Map(
                location=[center_lat, center_lng],
                zoom_start=12,
                tiles='OpenStreetMap',
                control_scale=True
            )
            
            # Agregar control de capas
            folium.LayerControl().add_to(m)
            
            # Agregar cada ruta al mapa
            routes = solution.get('routes', [])
            if not routes:
                print("[ADVERTENCIA] La solución no contiene rutas")
            
            for i, route in enumerate(routes):
                vehicle_id = route.get('vehicle_id', f'veh_{i}')
                color_idx = i % len(self.VEHICLE_COLORS)
                self._add_route_to_map(m, route, locations_dict, self.VEHICLE_COLORS[color_idx], vehicle_id)
            
            # Agregar título y leyenda
            title_html = """
                <h3 align="center" style="font-size:16px">
                    <b>Rutas de Entrega - Solución VRP</b><br>
                    <small>Generado el {}</small>
                </h3>
            """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Agregar mini mapa
            minimap = plugins.MiniMap()
            m.add_child(minimap)
            
            # Agregar control de pantalla completa
            plugins.Fullscreen().add_to(m)
            
            # Crear directorio de salida si no existe
            os.makedirs(os.path.dirname(os.path.abspath(output_file)) or '.', exist_ok=True)
            
            # Guardar el mapa como archivo HTML
            m.save(output_file)
            
            return output_file
            
        except Exception as e:
            error_msg = f"Error al generar el mapa: {str(e)}"
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg) from e

def visualize_solution(vrp_data: Dict[str, Any], solution: Dict[str, Any], 
                      output_dir: str = "output") -> str:
    """Función de conveniencia para visualizar una solución VRP.
    
    Args:
        vrp_data: Datos originales del VRP.
        solution: Solución generada por el solver.
        output_dir: Directorio donde se guardará el mapa.
        
    Returns:
        Ruta al archivo HTML generado.
    """
    visualizer = MapVisualizer(output_dir)
    return visualizer.plot_routes(vrp_data, solution)
