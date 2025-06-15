"""API principal para el solucionador VRP."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import os
import sys
from typing import Dict, Any
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# # Deshabilitar logs de Uvicorn - Comentado para permitir la depuración de errores de inicio
# logging.getLogger("uvicorn").setLevel(logging.WARNING)
# logging.getLogger("uvicorn.access").disabled = True

# Importar endpoints
from api.endpoints import cvrp, vrptw, pdp

# Variables de entorno
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Maneja eventos de inicio y cierre de la aplicación."""
    # Código de inicio
    logging.info("Iniciando API de Optimización de Rutas")
    
    # Verificar variables de entorno requeridas
    if not os.getenv("ORS_API_KEY") and not os.getenv("GOOGLE_MAPS_API_KEY"):
        logging.warning(
            "No se encontraron claves de API para ORS ni Google Maps. "
            "Solo estará disponible el cálculo de distancia euclidiana."
        )
    
    yield
    
    # Código de cierre
    logging.info("Deteniendo API de Optimización de Rutas")

# Crear aplicación FastAPI
app = FastAPI(
    title="VRP Solver API",
    description="API para resolver problemas de enrutamiento de vehículos (VRP)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,  # Deshabilitar ReDoc
    openapi_url="/openapi.json"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, restringir a dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Manejo de errores de validación
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Maneja errores de validación de Pydantic."""
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Error de validación",
            "errors": errors
        },
    )

# Incluir routers
app.include_router(cvrp.router, prefix=API_PREFIX)
app.include_router(vrptw.router, prefix=API_PREFIX)
app.include_router(pdp.router, prefix=API_PREFIX)

# Ruta de verificación de salud
@app.get(f"{API_PREFIX}/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    """Verifica el estado de la API."""
    return {"status": "ok"}

# Ruta raíz
@app.get("/")
async def root():
    """Redirige a la documentación de la API."""
    return {
        "message": "Bienvenido a la API de Optimización de Rutas (VRP)",
        "documentation": f"{API_PREFIX}/docs",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # Configuración para desarrollo
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=DEBUG,
        log_level="info" if DEBUG else "warning"
    )
