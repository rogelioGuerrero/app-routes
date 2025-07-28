"""
integrate_with_solver.py

Ejemplo de cómo integrar el visualizador con el solver VRP existente.
"""
import json
import os
import sys
from map_visualizer import MapVisualizer

def load_json_file(filepath):
    """Carga un archivo JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # Cargar datos de ejemplo (esto vendría de tu aplicación real)
    try:
        vrp_data = load_json_file('sample.json')
    except FileNotFoundError:
        print("Error: No se encontró el archivo sample.json")
        sys.exit(1)
    
    # En una aplicación real, aquí llamarías a tu solver
    # Por ejemplo:
    # from vrp_solver import VRPSolver
    # solver = VRPSolver()
    # solution = solver.solve(vrp_data)
    
    # Para este ejemplo, usaremos una solución de ejemplo similar a la salida real del solver
    solution = {
        "status": "success",
        "total_distance": 12500,
        "total_time": 15000,
        "vehicles_used": 2,
        "routes": [
            {
                "vehicle_id": "veh_0",
                "distance": 7500,
                "duration": 9000,
                "stops": [
                    {
                        "location_id": "depot",
                        "arrival_time": 0,
                        "departure_time": 300,
                        "service_time": 300
                    },
                    {
                        "location_id": "c1",
                        "arrival_time": 3600,
                        "departure_time": 3900,
                        "service_time": 300
                    },
                    {
                        "location_id": "c2",
                        "arrival_time": 7200,
                        "departure_time": 7500,
                        "service_time": 300
                    },
                    {
                        "location_id": "depot",
                        "arrival_time": 9000,
                        "departure_time": 9000,
                        "service_time": 0
                    }
                ]
            },
            {
                "vehicle_id": "veh_1",
                "distance": 5000,
                "duration": 6000,
                "stops": [
                    {
                        "location_id": "depot_norte",
                        "arrival_time": 0,
                        "departure_time": 300,
                        "service_time": 300
                    },
                    {
                        "location_id": "c3",
                        "arrival_time": 1800,
                        "departure_time": 2100,
                        "service_time": 300
                    },
                    {
                        "location_id": "c4",
                        "arrival_time": 3600,
                        "departure_time": 3900,
                        "service_time": 300
                    },
                    {
                        "location_id": "depot_norte",
                        "arrival_time": 6000,
                        "departure_time": 6000,
                        "service_time": 0
                    }
                ]
            }
        ]
    }
    
    # Crear el visualizador
    visualizer = MapVisualizer(output_dir="output")
    
    try:
        # Generar el mapa
        output_file = visualizer.plot_routes(vrp_data, solution)
        
        # Mostrar información
        print(f"Visualización generada exitosamente: {os.path.abspath(output_file)}")
        print("Abriendo en el navegador...")
        
        # Abrir en el navegador predeterminado
        import webbrowser
        webbrowser.open(f'file://{os.path.abspath(output_file)}')
        
    except Exception as e:
        print(f"Error al generar la visualización: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
