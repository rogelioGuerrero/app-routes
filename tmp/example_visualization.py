"""
example_visualization.py

Ejemplo de cómo usar el visualizador de rutas VRP con folium.
"""
import json
import os
from map_visualizer import visualize_solution
import webbrowser

def load_json_file(filepath):
    """Carga un archivo JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # Cargar datos de ejemplo
    vrp_data = load_json_file('sample.json')
    
    # Crear una solución de ejemplo (en un caso real, esto vendría del solver)
    # Aquí usamos una solución de ejemplo basada en la estructura esperada
    solution = {
        "status": "success",
        "total_distance": 10085,
        "total_time": 12605,
        "vehicles_used": 1,
        "routes": [
            {
                "vehicle_id": "veh_0",
                "distance": 10085,
                "duration": 12605,
                "stops": [
                    {
                        "location_id": "depot",
                        "arrival_time": 0,
                        "departure_time": 0,
                        "service_time": 0
                    },
                    {
                        "location_id": "c1",
                        "arrival_time": 3600,  # 1 hora después
                        "departure_time": 4200,  # +10 minutos de servicio
                        "service_time": 600
                    },
                    {
                        "location_id": "c2",
                        "arrival_time": 5400,  # 30 minutos después
                        "departure_time": 6000,  # +10 minutos de servicio
                        "service_time": 600
                    },
                    {
                        "location_id": "depot",
                        "arrival_time": 7200,  # 2 horas después
                        "departure_time": 7200,
                        "service_time": 0
                    }
                ]
            }
        ]
    }
    
    # Visualizar la solución
    output_file = visualize_solution(vrp_data, solution)
    output_path = os.path.abspath(output_file)
    print(f"Mapa generado exitosamente: {output_path}")
    print("Abriendo en el navegador...")
    
    # Abrir en el navegador predeterminado
    webbrowser.open(f'file://{output_path}')

if __name__ == "__main__":
    main()
