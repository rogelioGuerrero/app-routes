import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_route_polyline_and_geojson(latlons, api_key=None, rdp_tolerance=0.0001):
    """
    Dado un listado de (lat, lon), retorna el polyline y el GeoJSON de la ruta realista usando Google Directions.
    Aplica compresión Ramer-Douglas-Peucker (rdp) a los puntos decodificados (tolerancia por defecto: 0.0001 ≈ 10m).
    """
    import numpy as np
    try:
        from rdp import rdp
    except ImportError:
        rdp = None
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY no encontrada")
    if len(latlons) < 2:
        return None, None
    origin = f"{latlons[0][0]},{latlons[0][1]}"
    destination = f"{latlons[-1][0]},{latlons[-1][1]}"
    waypoints = [f"{lat},{lon}" for lat, lon in latlons[1:-1]]
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": destination,
        "key": api_key,
        "mode": "driving"
    }
    if waypoints:
        params["waypoints"] = "|".join(waypoints)
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "OK":
        return None, None
    # Polyline global (overview_polyline)
    polyline = data["routes"][0]["overview_polyline"]["points"]
    # GeoJSON (decodificado)
    points = decode_google_polyline(polyline)
    # Aplica RDP si está disponible
    if rdp is not None and len(points) > 2:
        try:
            arr = np.array(points, dtype=float)
            points_rdp = rdp(arr, epsilon=rdp_tolerance)
            points = [tuple(pt) for pt in points_rdp]
        except Exception:
            pass  # Si falla, usa los puntos originales
    geojson = {
        "type": "LineString",
        "coordinates": [[lon, lat] for lat, lon in points]
    }
    return polyline, geojson

def decode_google_polyline(polyline_str):
    # Decodifica polyline de Google a lista de (lat, lon)
    # Fuente: https://gist.github.com/signed0/2031157
    import math
    points = []
    index = lat = lng = 0
    changes = {'lat': 0, 'lng': 0}
    while index < len(polyline_str):
        for key in ['lat', 'lng']:
            shift = result = 0
            while True:
                b = ord(polyline_str[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            if (result & 1):
                changes[key] = ~(result >> 1)
            else:
                changes[key] = (result >> 1)
        lat += changes['lat']
        lng += changes['lng']
        points.append((lat / 1e5, lng / 1e5))
    return points
