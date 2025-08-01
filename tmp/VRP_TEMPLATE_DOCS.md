# Plantilla Excel para Problemas VRP

Este documento describe la estructura del archivo Excel requerido por el solver VRP. Copia y pega cada tabla en una hoja diferente de Excel.

## 1. Hoja: Locations

| Campo | Tipo | Requerido | Descripción | Ejemplo |
|-------|------|-----------|-------------|----------|
| id | string | Sí | Identificador único de la ubicación | "depot1" |
| name | string | No | Nombre descriptivo | "Depósito Principal" |
| coords | string | Sí | Coordenadas [longitud, latitud] | "[-99.1332, 19.4326]" |
| type | string | Sí | Tipo: 'depot', 'pickup', 'delivery' | "depot" |
| service_time | int | No | Tiempo de servicio en segundos | 300 |
| time_window_start | int | No | Hora inicio en segundos (0-86399) | 28800 (8:00 AM) |
| time_window_end | int | No | Hora fin en segundos (0-86399) | 72000 (8:00 PM) |
| weight_demand | float | No | Demanda de peso (positivo para pickup) | 10.5 |
| volume_demand | float | No | Demanda de volumen (positivo para pickup) | 1.2 |
| required_skills | string | No | Lista de habilidades requeridas | '["refrigerado"]' |

## 2. Hoja: Vehicles

| Campo | Tipo | Requerido | Descripción | Ejemplo |
|-------|------|-----------|-------------|----------|
| id | string | Sí | Identificador único del vehículo | "camion1" |
| start_location_id | string | Sí | ID de ubicación de inicio | "depot1" |
| end_location_id | string | Sí | ID de ubicación de fin | "depot1" |
| start_time | int | No | Hora de inicio en segundos | 28800 |
| end_time | int | No | Hora de fin en segundos | 72000 |
| capacity | float | No | Capacidad de peso | 1000.0 |
| volume_capacity | float | No | Capacidad de volumen | 10.0 |
| skills | string | No | Lista de habilidades | '["refrigerado"]' |
| cost_per_km | float | No | Costo por kilómetro | 10.5 |
| cost_per_hour | float | No | Costo por hora | 200.0 |
| fixed_cost | float | No | Costo fijo por ruta | 500.0 |

## 3. Hoja: PickupDeliveries

| Campo | Tipo | Requerido | Descripción | Ejemplo |
|-------|------|-----------|-------------|----------|
| pickup_id | string | Sí | ID de ubicación de recogida | "cliente1" |
| delivery_id | string | Sí | ID de ubicación de entrega | "cliente2" |
| amount | float | No | Cantidad a transportar | 5.0 |

## 4. Hoja: TimeWindows (Opcional)

| Campo | Tipo | Requerido | Descripción | Ejemplo |
|-------|------|-----------|-------------|----------|
| start_time | string | Sí | Hora de inicio (HH:MM) | "07:00" |
| end_time | string | Sí | Hora de fin (HH:MM) | "09:00" |
| factor | float | No | Factor de tiempo | 1.5 |
| description | string | No | Descripción | "Hora pico" |

## 5. Hoja: Congestion (Opcional)

| Campo | Tipo | Requerido | Descripción | Ejemplo |
|-------|------|-----------|-------------|----------|
| start_time | string | Sí | Hora de inicio (HH:MM) | "17:00" |
| end_time | string | Sí | Hora de fin (HH:MM) | "20:00" |
| factor | float | No | Factor de congestión | 1.7 |
| description | string | No | Descripción | "Tráfico tarde" |

## 6. Hoja: OptimizationProfile (Opcional)

| Campo | Tipo | Requerido | Valores posibles | Ejemplo |
|-------|------|-----------|-------------------|----------|
| first_solution_strategy | string | No | PATH_CHEAPEST_ARC, SAVINGS, etc. | "PATH_CHEAPEST_ARC" |
| local_search_metaheuristic | string | No | GUIDED_LOCAL_SEARCH, TABU_SEARCH, etc. | "GUIDED_LOCAL_SEARCH" |
| time_limit_seconds | int | No | Tiempo límite en segundos | 30 |

## Notas Importantes

1. Las hojas obligatorias son: Locations, Vehicles y PickupDeliveries
2. Las horas deben estar en formato de 24 horas (ej. "14:30" para las 2:30 PM)
3. Los tiempos en segundos van desde 0 (medianoche) hasta 86399 (23:59:59)
4. Las listas (como skills) deben estar en formato JSON válido
5. Las coordenadas deben estar en el formato [longitud, latitud]
