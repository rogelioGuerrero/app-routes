from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import router as optimize_router
from distance_matrix import router as matrix_router
from vrp_solver import router as vrp_router
from vrp_advanced import router as vrp_advanced_router
from vrp_capacity import router as vrp_capacity_router
from vrp_skills import router as vrp_skills_router
from vrp_skills_check import router as vrp_skills_check_router

app = FastAPI(
    title="Optimizador de Rutas para PYMES",
    description="API para optimización logística con OR-Tools y OpenRouteService",
    version="0.1.0"
)

# Configuración básica de CORS (puedes ajustar los orígenes según el frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(optimize_router)
app.include_router(matrix_router)
app.include_router(vrp_router)
app.include_router(vrp_advanced_router)
app.include_router(vrp_capacity_router)
app.include_router(vrp_skills_router)
app.include_router(vrp_skills_check_router)

@app.get("/")
def root():
    return {"message": "Optimizador de rutas activo. Listo para recibir instrucciones."}
