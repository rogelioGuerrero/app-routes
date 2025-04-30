import os
from dotenv import load_dotenv
import requests
from typing import List, Tuple, Optional

load_dotenv()

# Lee la API Key de OpenRouteService o Google Directions desde el .env
def get_api_key(service: str = "ORS") -> Optional[str]:
    if service.upper() == "ORS":
        return os.getenv("ORS_API_KEY")
    elif service.upper() == "GOOGLE":
        return os.getenv("GOOGLE_API_KEY")
    return None

# Construye y ejecuta la petición a Google Directions API
# https://developers.google.com/maps/documentation/directions/start

def optimize_route_google(
    origin: str,
    destination: str,
    waypoints: List[str],
    api_key: Optional[str] = None,
    optimize_waypoints: bool = False
) -> dict:
    """
    origin: dirección o "lat,lon"
    destination: dirección o "lat,lon"
    waypoints: lista de direcciones o "lat,lon"
    optimize_waypoints: si True, Google optimiza el orden de los waypoints; si False, respeta el orden dado.
    """
    if not api_key:
        api_key = get_api_key("GOOGLE")
    if not api_key or not isinstance(api_key, str) or len(api_key.strip()) < 10:
        print("[ERROR] GOOGLE_API_KEY no encontrada o inválida. Verifica tu archivo .env. Debe tener una línea GOOGLE_API_KEY=tu_clave_aqui")
        raise ValueError("GOOGLE_API_KEY no encontrada o inválida. Verifica tu archivo .env. Debe tener una línea GOOGLE_API_KEY=tu_clave_aqui")
    print(f"[DEBUG] Usando GOOGLE_API_KEY que inicia con: {api_key[:6]}... (oculto por seguridad)")
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": destination,
        "key": api_key,
        "mode": "driving",
    }
    if waypoints:
        if optimize_waypoints:
            params["waypoints"] = "optimize:true|" + "|".join(waypoints)
        else:
            params["waypoints"] = "|".join(waypoints)
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()
