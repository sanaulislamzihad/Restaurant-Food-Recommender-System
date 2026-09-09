"""Aggregate router mounted under /api."""

from fastapi import APIRouter

from app.api.routes import admin, auth, interactions, menu, recommendations

api_router = APIRouter(prefix="/api")
api_router.include_router(auth.router)
api_router.include_router(menu.router)
api_router.include_router(interactions.router)
api_router.include_router(recommendations.router)
api_router.include_router(admin.router)
