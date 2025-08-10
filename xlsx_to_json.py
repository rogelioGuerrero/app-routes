"""
Módulo para convertir archivos XLSX al formato JSON esperado por el optimizador VRP.
"""
import pandas as pd
import json
import re
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

# --- Utilidades de parsing amigables para el usuario ---
def _parse_time_to_seconds(val: Any) -> Optional[int]:
    """Convierte una hora a segundos desde medianoche.

    Acepta:
    - Cadenas 'HH:MM' o 'HH:MM:SS'
    - Números/strings numéricos como segundos (p.ej. 39600)
    """
    if val is None:
        return None
    # Si viene como número (o string numérico), interpretarlo como segundos directos
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip()
    if not s:
        return None
    # Si es numérico puro, tratar como segundos
    if re.fullmatch(r"\d+", s):
        return int(s)
    # Formatos con ':' -> HH:MM(:SS)?
    if ":" in s:
        parts = s.split(":")
        try:
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            sec = int(parts[2]) if len(parts) > 2 else 0
            return h * 3600 + m * 60 + sec
        except Exception:
            return None
    return None

def _parse_break_windows(cell_val: Any) -> List[List[int]]:
    """Parsea ventanas de tiempo de descanso a [[ini, fin], ...] (segundos).

    Acepta:
    - Lista ya en segundos: [[39600,50400], ...]
    - Texto JSON-like de lista: "[[39600,50400],[54000,57600]]"
    - Texto de rangos legibles: "11:00-14:00; 15:00-16:00" (separador ';' o ',')
    """
    if cell_val is None:
        return []
    # Si es lista ya estructurada
    if isinstance(cell_val, list):
        out: List[List[int]] = []
        for rng in cell_val:
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                try:
                    start = int(float(rng[0])) if rng[0] is not None else None
                    end = int(float(rng[1])) if rng[1] is not None else None
                    if start is not None and end is not None:
                        out.append([start, end])
                except Exception:
                    continue
        return out
    s = str(cell_val).strip()
    if not s:
        return []
    # Si parece lista JSON, eval rápida y recursiva
    if s.startswith("[") and s.endswith("]"):
        try:
            val = eval(s)
            return _parse_break_windows(val)
        except Exception:
            return []
    # Normalizar separador de rango y múltiplos
    s = s.replace("–", "-")  # en-dash a guion
    # Separar por ';' o ',' como múltiplos
    tokens = re.split(r"[;,]", s)
    windows: List[List[int]] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if "-" not in tok:
            continue
        start_str, end_str = [p.strip() for p in tok.split("-", 1)]
        start = _parse_time_to_seconds(start_str)
        end = _parse_time_to_seconds(end_str)
        if start is not None and end is not None:
            windows.append([start, end])
    return windows

def _parse_minutes_to_seconds(val: Any) -> Optional[int]:
    """Interpreta el valor como minutos y retorna segundos.

    Soporta números y strings numéricos. Admite separador decimal con coma o punto.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(round(float(val) * 60))
    s = str(val).strip()
    if not s:
        return None
    # Reemplazar coma decimal por punto
    s = s.replace(",", ".")
    try:
        return int(round(float(s) * 60))
    except Exception:
        return None

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
        
        # Construcción de 'breaks' exclusivamente desde dos columnas del XLSX (amigables):
        #  - 'break_duration' o 'breaks_duration'  (EN MINUTOS, se convierte a segundos)
        #  - 'break_windows'  o 'breaks_time_windows' (rango(s) horario(s) legibles p.ej. "11:00-14:00; 15:00-16:00")
        # Nota: Ya NO se usa ninguna columna 'breaks' en el XLSX.
        breaks_value = []
        duration_cell = safe_convert(row.get('break_duration', row.get('breaks_duration')))
        tw_cell = safe_convert(row.get('break_windows', row.get('breaks_time_windows')))
        if duration_cell not in (None, '') and tw_cell not in (None, '', '[]'):
            duration_seconds = _parse_minutes_to_seconds(duration_cell)
            time_windows = _parse_break_windows(tw_cell)
            if duration_seconds is not None and time_windows:
                breaks_value = [{
                    'duration': duration_seconds,
                    'time_windows': time_windows
                }]
        
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
            'breaks': breaks_value,  # Formato: [{"duration": segundos, "time_windows": [[inicio1, fin1], [inicio2, fin2]]}]
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
