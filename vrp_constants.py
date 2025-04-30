# Constantes centralizadas para VRP

# --- Velocidades y distancias ---
DEFAULT_SPEED_KMH = 40            # Velocidad promedio para fallback
DEFAULT_DISTANCE_FALLBACK = 1e6   # Valor grande para distancias imposibles

# --- Tiempo y solver ---
MAX_TIME_MINUTES = 1440           # Minutos en un día
DEFAULT_SOLVER_TIMEOUT = 30       # Segundos para el solver de OR-Tools
DEFAULT_SLACK_MINUTES = 1440      # Slack máximo permitido en minutos
DEFAULT_ROUTE_TIME_LIMIT = 1440   # Tiempo máximo por ruta en minutos

# --- Penalizaciones y costos ---
SKILL_PENALTY = 10000             # Penalización por skills no compatibles
DEFAULT_SKILL_PENALTY = 10000     # Alias para compatibilidad
DEFAULT_SOFT_TIME_PENALTY = 1000  # Penalización por minuto fuera de ventana blanda

# --- Buffers y ventanas ---
DEFAULT_BUFFER_MINUTES = 10       # Buffer de tiempo estándar
DEFAULT_SERVICE_TIME = 5          # Tiempo de servicio estándar por cliente
DEFAULT_TIME_WINDOW = [420, 1080] # Ventana estándar (07:00 a 18:00)

# --- Congestión y horas pico ---
DEFAULT_PEAK_HOURS = []           # Ejemplo: [450, 570] para 07:30 a 09:30
DEFAULT_PEAK_MULTIPLIER = 1.3     # 30% más lento en horas pico

# --- Límites y thresholds ---
MAX_POLYLINE_ROUTES = 10          # Máximo de rutas con polylines
MAX_CLIENTS = 100                 # Máximo de clientes por request
MAX_VEHICLES = 20                 # Máximo de vehículos por request

# --- Otros parámetros útiles ---
DEFAULT_UNITS = "metric"
DEFAULT_MODE = "driving"
