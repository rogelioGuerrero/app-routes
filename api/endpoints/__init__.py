"""Paquete de endpoints de la API."""

# Importar los routers para que estén disponibles al importar el paquete
from .cvrp import router as cvrp_router

__all__ = ['cvrp_router']
