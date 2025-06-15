# VRP Solver API

API para resolver problemas de Enrutamiento de Vehículos (VRP) con restricciones de tiempo y capacidad.

## Características Principales

- Resolución de problemas VRPTW (VRP con ventanas de tiempo)
- Soporte para múltiples vehículos con diferentes capacidades
- Optimización de rutas considerando:
  - Restricciones de tiempo (time windows)
  - Capacidad de vehículos
  - Habilidades requeridas
  - Tiempos de servicio

## Endpoint Principal

`POST /api/v1/vrptw/solve-unified`

### Ejemplo de Uso

```bash
curl -X POST http://localhost:8000/api/v1/vrptw/solve-unified \
  -H "Content-Type: application/json" \
  -d @test_payload_unified.json
```

## Requisitos

- Python 3.8+
- OR-Tools
- FastAPI
- Uvicorn

## Instalación

1. Clonar el repositorio
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Configurar variables de entorno (ver `.env.example`)
4. Iniciar el servidor:
   ```bash
   uvicorn main:app --reload
   ```

## Despliegue

El proyecto está configurado para ser desplegado en Render. Consulta el archivo `render.yaml` para más detalles.

## Licencia

MIT
  - Cálculo euclidiano (sin conexión)
- API RESTful con documentación interactiva
- Fácil de extender con nuevos proveedores y restricciones

## Requisitos

- Python 3.8+
- pip (gestor de paquetes de Python)
- Claves de API para los proveedores que desees usar (ORS o Google Maps)

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/vrp-solver.git
   cd vrp-solver
   ```

2. Crea un entorno virtual y actívalo:
   ```bash
   python -m venv venv
   # En Windows:
   .\\venv\\Scripts\\activate
   # En Unix/macOS:
   # source venv/bin/activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Configura las variables de entorno:
   - Copia el archivo `.env.example` a `.env`
   - Edita `.env` y agrega tus claves de API

## Uso

### Iniciar el servidor de desarrollo:

```bash
uvicorn api.main:app --reload
```

La API estará disponible en `http://localhost:8000`

### Documentación de la API:

- Documentación interactiva (Swagger UI): http://localhost:8000/api/v1/docs
- Documentación alternativa (ReDoc): http://localhost:8000/api/v1/redoc

### Ejemplo de solicitud CURL:

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/cvrp/solve' \
  -H 'Content-Type: application/json' \
  -d '{
    "locations": [
      {"id": "depot", "lat": 40.7128, "lng": -74.0060, "demand": 0},
      {"id": "loc1", "lat": 34.0522, "lng": -118.2437, "demand": 10},
      {"id": "loc2", "lat": 41.8781, "lng": -87.6298, "demand": 15}
    ],
    "vehicles": [
      {"id": "veh1", "capacity": 50}
    ]
  }'
```

## Estructura del Proyecto

```
vrp-solver/
├── api/                    # Código de la API
│   ├── endpoints/          # Definición de endpoints
│   ├── models.py           # Modelos Pydantic
│   └── main.py             # Aplicación FastAPI
├── core/                   # Lógica principal
│   └── cvrp/               # Implementación CVRP
│       ├── solver.py       # Solucionador CVRP
│       └── solver_adapter.py # Adaptador para el sistema de matrices
├── services/               # Servicios externos
│   └── distance_matrix/    # Proveedores de matrices
├── tests/                  # Pruebas unitarias
├── .env.example            # Plantilla de variables de entorno
├── requirements.txt        # Dependencias
└── README.md              # Este archivo
```

## Variables de Entorno

| Variable | Descripción | Valor por defecto |
|----------|-------------|------------------|
| `DEBUG` | Modo depuración | `false` |
| `API_PREFIX` | Prefijo para rutas de la API | `/api/v1` |
| `PORT` | Puerto para el servidor | `8000` |
| `HOST` | Host para el servidor | `0.0.0.0` |
| `ORS_API_KEY` | Clave de API para OpenRouteService | - |
| `GOOGLE_MAPS_API_KEY` | Clave de API para Google Maps | - |
| `CORS_ORIGINS` | Orígenes permitidos para CORS | `*` |

## Despliegue

### Con Docker (recomendado para producción):

1. Construye la imagen:
   ```bash
   docker build -t vrp-solver .
   ```

2. Ejecuta el contenedor:
   ```bash
   docker run -d --name vrp-solver -p 8000:8000 --env-file .env vrp-solver
   ```

### Sin Docker:

1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Configura las variables de entorno en `.env`

3. Inicia el servidor con Gunicorn (para producción):
   ```bash
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.main:app
   ```

## Pruebas

Para ejecutar las pruebas:

```bash
pytest tests/
```

## Licencia

MIT License - ver el archivo [LICENSE](LICENSE) para más detalles.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, envía un Pull Request o abre un Issue para discutir los cambios propuestos.

## Créditos

- [Google OR-Tools](https://developers.google.com/optimization) - Para el núcleo del solucionador
- [FastAPI](https://fastapi.tiangolo.com/) - Para la API web
- [OpenRouteService](https://openrouteservice.org/) - Para datos de ruteo de alta calidad
