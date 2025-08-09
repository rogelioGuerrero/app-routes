"""
Módulo para convertir archivos XLSX al formato JSON esperado por el optimizador VRP.
"""
import pandas as pd
import json
from typing import Dict, Any, List, Optional

def safe_convert(value, default=None):
    """Convierte valores de pandas a tipos nativos de Python."""
    if pd.isna(value):
        return default
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if hasattr(value, 'item'):  # Para numpy types
        return value.item()
    return value

def xlsx_to_json(file_path: str) -> Dict[str, Any]:
    """
    Convierte un archivo XLSX al formato JSON esperado por el optimizador VRP.
    
    Hojas obligatorias:
    - Locations: Información de ubicaciones (depósitos, clientes, etc.)
    - Vehicles: Información de vehículos
    - PickupDeliveries: Pares de recogida-entrega
    
    Hojas adicionales mapeadas a la raíz del JSON:
    - TimeWindows: Ventanas de tiempo para optimización
    - Congestion: Factores de congestión por horario
    - OptimizationProfile: Configuración del optimizador
    
    Args:
        file_path: Ruta al archivo XLSX
        
    Returns:
        Dict con la estructura de datos para el optimizador VRP
    """
    # Validar hojas obligatorias
    xls = pd.ExcelFile(file_path)
    required_sheets = ['Locations', 'Vehicles']
    missing_sheets = [sheet for sheet in required_sheets if sheet not in xls.sheet_names]
    
    if missing_sheets:
        raise ValueError(f'Faltan hojas obligatorias en el archivo Excel: {missing_sheets}')
    
    # Verificar si existe la hoja PickupDeliveries
    has_pickup_deliveries = 'PickupDeliveries' in xls.sheet_names
    
    result = {}
    
    # 1. Procesar Locations (obligatoria)
    df_locations = pd.read_excel(xls, sheet_name='Locations')
    locations = []
    
    for _, row in df_locations.iterrows():
        location = {
            'id': safe_convert(row.get('id', '')),
            'name': safe_convert(row.get('name', '')),
            'coords': eval(safe_convert(row.get('coords', '[]'), '[]')),
            'type': safe_convert(row.get('type', 'delivery')),
            'service_time': safe_convert(row.get('service_time', 0), 0),
            'time_window_start': int(float(safe_convert(row.get('time_window_start', 0), 0))),
            'time_window_end': int(float(safe_convert(row.get('time_window_end', 0), 0))),
            'weight_demand': safe_convert(row.get('weight_demand', 0), 0),
            'volume_demand': safe_convert(row.get('volume_demand', 0.0), 0.0),
            'required_skills': eval(safe_convert(row.get('required_skills', '[]'), '[]'))
        }
        locations.append(location)
    
    result['locations'] = locations
    
    # 2. Procesar Vehicles (obligatoria)
    df_vehicles = pd.read_excel(xls, sheet_name='Vehicles')
    vehicles = []
    
    for _, row in df_vehicles.iterrows():
        # Mapeo de columnas según el formato estándar
        weight_capacity = safe_convert(row.get('weight_capacity', 0), 0)
        volume_capacity = safe_convert(row.get('volume_capacity', 0.0), 0.0)
        
        vehicle = {
            'id': safe_convert(row.get('id', '')),
            'name': safe_convert(row.get('name', f"Vehículo {len(vehicles) + 1}")),
            'start_location_id': safe_convert(row.get('start_location_id', row.get('start_location', ''))),
            'end_location_id': safe_convert(row.get('end_location_id', row.get('end_location', ''))),
            'start_time': int(float(safe_convert(row.get('start_time', 0), 0))),
            'end_time': int(float(safe_convert(row.get('end_time', 0), 0))),
            'weight_capacity': weight_capacity,
            'volume_capacity': volume_capacity,
            'skills': eval(safe_convert(row.get('skills', '[]'), '[]')),
            'breaks': eval(safe_convert(row.get('breaks', '[]'), '[]')),  # Formato: [{"duration": segundos, "time_windows": [[inicio1, fin1], [inicio2, fin2]]}]
            'cost_per_km': safe_convert(row.get('cost_per_km', 0.0), 0.0),
            'cost_per_hour': safe_convert(row.get('cost_per_hour', 0.0), 0.0),
            'fixed_cost': safe_convert(row.get('fixed_cost', 0.0), 0.0)
        }
        vehicles.append(vehicle)
    
    result['vehicles'] = vehicles
    
    # 3. Procesar Pickup-Deliveries (opcional)
    if has_pickup_deliveries:
        try:
            df_pd = pd.read_excel(xls, sheet_name='PickupDeliveries')
            pickup_deliveries = []
            
            for _, row in df_pd.iterrows():
                pd_pair = {
                    'pickup_id': safe_convert(row.get('pickup_id', '')),
                    'delivery_id': safe_convert(row.get('delivery_id', '')),
                    'amount': safe_convert(row.get('amount', 0), 0)
                }
                if pd_pair['pickup_id'] and pd_pair['delivery_id']:  # Solo agregar si tiene ambos IDs
                    pickup_deliveries.append(pd_pair)
            
            if pickup_deliveries:  # Solo agregar si hay pares válidos
                result['pickups_deliveries'] = pickup_deliveries
        except Exception as e:
            print(f'Advertencia: No se pudo procesar la hoja PickupDeliveries: {str(e)}')
    
    # 4. Procesar hojas adicionales con estructuras conocidas
    processed_sheets = set(required_sheets)
    if has_pickup_deliveries:
        processed_sheets.add('PickupDeliveries')
    
    # 4.1 TimeWindows
    if 'TimeWindows' in xls.sheet_names and 'TimeWindows' not in processed_sheets:
        try:
            df_time_windows = pd.read_excel(xls, sheet_name='TimeWindows')
            time_windows = []
            for _, row in df_time_windows.iterrows():
                time_window = {
                    'time_window': [
                        safe_convert(row.get('start_time', '00:00')),
                        safe_convert(row.get('end_time', '23:59'))
                    ],
                    'factor': safe_convert(row.get('factor', 1.0), 1.0),
                    'description': safe_convert(row.get('description', ''))
                }
                time_windows.append(time_window)
            result['time_windows'] = time_windows
            processed_sheets.add('TimeWindows')
        except Exception as e:
            print(f'Advertencia: No se pudo procesar la hoja TimeWindows: {str(e)}')
    
    # 4.2 Congestion
    if 'Congestion' in xls.sheet_names and 'Congestion' not in processed_sheets:
        try:
            df_congestion = pd.read_excel(xls, sheet_name='Congestion')
            congestion = []
            for _, row in df_congestion.iterrows():
                congestion_item = {
                    'time_window': [
                        safe_convert(row.get('start_time', '00:00')),
                        safe_convert(row.get('end_time', '23:59'))
                    ],
                    'factor': safe_convert(row.get('factor', 1.0), 1.0),
                    'description': safe_convert(row.get('description', ''))
                }
                congestion.append(congestion_item)
            result['congestion'] = congestion
            result['congestion_enabled'] = True
            processed_sheets.add('Congestion')
        except Exception as e:
            print(f'Advertencia: No se pudo procesar la hoja Congestion: {str(e)}')
    
    # 4.3 OptimizationProfile
    if 'OptimizationProfile' in xls.sheet_names and 'OptimizationProfile' not in processed_sheets:
        try:
            df_opt = pd.read_excel(xls, sheet_name='OptimizationProfile')
            if not df_opt.empty:
                profile = {
                    'first_solution_strategy': safe_convert(
                        df_opt.iloc[0].get('first_solution_strategy', 'PATH_CHEAPEST_ARC'),
                        'PATH_CHEAPEST_ARC'
                    ),
                    'local_search_metaheuristic': safe_convert(
                        df_opt.iloc[0].get('local_search_metaheuristic', 'GUIDED_LOCAL_SEARCH'),
                        'GUIDED_LOCAL_SEARCH'
                    ),
                    'time_limit_seconds': safe_convert(df_opt.iloc[0].get('time_limit_seconds', 30), 30)
                }
                result['optimization_profile'] = profile
                processed_sheets.add('OptimizationProfile')
        except Exception as e:
            print(f'Advertencia: No se pudo procesar la hoja OptimizationProfile: {str(e)}')
    
    # 5. Procesar dinámicamente cualquier otra hoja no procesada
    for sheet_name in xls.sheet_names:
        if sheet_name not in processed_sheets:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                # Convertir el DataFrame a una lista de diccionarios
                sheet_data = []
                for _, row in df.iterrows():
                    row_data = {}
                    for col in df.columns:
                        val = row[col]
                        if pd.notna(val):  # Solo incluir valores no nulos
                            row_data[col] = safe_convert(val)
                    if row_data:  # Solo agregar filas con datos
                        sheet_data.append(row_data)
                
                if sheet_data:  # Solo agregar hojas con datos
                    # Usar el nombre de la hoja en snake_case como clave
                    key = sheet_name.lower().replace(' ', '_')
                    result[key] = sheet_data
                
            except Exception as e:
                print(f'Advertencia: No se pudo procesar la hoja {sheet_name}: {str(e)}')
    
    return result
