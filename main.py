from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
import uvicorn
from pydantic import BaseModel
import sys
import os
import logging # Import logging module

# Add project root to Python path to allow absolute imports
# This ensures that modules like 'models' and 'core' can be found
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv

# Configurar logging básico para la aplicación
logging.basicConfig(
    level=logging.DEBUG, 
    stream=sys.stdout, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Cargar variables de entorno desde .env al inicio
load_dotenv()

# Añadir el directorio actual al path para importaciones
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importaciones locales
app = None
try:
    # Importaciones locales que podrían fallar
    from api.endpoints import vrptw, cvrp

    app = FastAPI(title="VRP Solver API", version="0.1.0")

    # Incluir los routers de los endpoints
    app.include_router(cvrp.router, prefix="/api/v1", tags=["CVRP"])
    app.include_router(vrptw.router, prefix="/api/v1", tags=["VRPTW"])

    @app.get("/")
    async def root():
        return {"name": "VRP Solver API", "status": "running"}

    logging.info("FastAPI application initialized successfully.")

except Exception as e:
    logging.critical("CRITICAL ERROR: Failed to initialize FastAPI application.", exc_info=True)
    # Salir explícitamente si la app no se puede inicializar.
    # Esto previene que Uvicorn intente correr una app 'None'.
    sys.exit(1)


if __name__ == "__main__":
    # Asegurarse de que la app no es None antes de intentar correrla.
    if app:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        logging.error("Could not start server because FastAPI app object is None.")
