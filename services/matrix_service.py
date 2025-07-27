"""Backward-compatibility wrapper around the new asynchronous distance_matrix module.

This module exposes a **synchronous** `OrsMatrixProvider` class with the legacy
`get_matrices()` method expected by some older parts of the codebase
(`main.py`, for instance).  Internally it re-uses the modern asynchronous
implementation in `services.distance_matrix.providers.ors.ORSMatrix` via the
`DistanceMatrixFactory` helper.  This prevents us from having to refactor the
whole FastAPI layer right now while still leveraging the new provider stack.
"""
from __future__ import annotations

from typing import List, Dict, Any, Literal
import asyncio
import logging
import inspect

from .distance_matrix import DistanceMatrixFactory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legacy public API ----------------------------------------------------------
# ---------------------------------------------------------------------------
class OrsMatrixProvider:  # pylint: disable=too-few-public-methods
    """Synchronous wrapper that mimics the old behaviour.

    Usage::

        provider = OrsMatrixProvider()
        matrices = provider.get_matrices(locations, ["duration", "distance"])

    The *locations* argument may be either::

        [
            {"id": "loc1", "latitude": 20.0, "longitude": -103.0},
            ...
        ]

    or already the simplified form consumed by the new provider::

        [
            {"lat": 20.0, "lng": -103.0},
            ...
        ]
    """

    _METRIC_MAP: dict[str, Literal["durations", "distances"]] = {
        "duration": "durations",
        "distance": "distances",
    }

    def __init__(self, api_key: str | None = None, profile: str = "driving-car") -> None:
        self._profile = profile
        # We create the *async* provider once and reuse it.
        self._async_provider = DistanceMatrixFactory.create_provider("ors", api_key=api_key)
        logger.debug("OrsMatrixProvider initialised (sync wrapper around async ORS provider)")

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def get_matrices(self, locations: List[Dict[str, Any]], metrics: List[str] | None = None) -> Dict[str, Any]:
        """Legacy synchronous helper that blocks until the matrices are fetched."""
        if not metrics:
            metrics = ["duration", "distance"]
        # Validate & adapt locations -------------------------------------------------
        formatted_locations = [
            {"lat": loc.get("lat") or loc.get("latitude"), "lng": loc.get("lng") or loc.get("longitude")}
            for loc in locations
        ]

        # Translate legacy metric names to the ones expected by the async provider.
        async_metrics = [self._METRIC_MAP[m] for m in metrics if m in self._METRIC_MAP]

        async def _fetch():
            return await self._async_provider.get_matrix(
                formatted_locations, metrics=async_metrics, profile=self._profile
            )

        # If there's an existing running loop (unlikely inside FastAPI sync endpoint),
        # we have to create a task differently.  For safety we detect this.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            logger.debug("Running inside existing asyncio loop – creating task and waiting")
            task = loop.create_task(_fetch())
            result = loop.run_until_complete(task) if not inspect.iscoroutine(task) else loop.run_until_complete(task)
        else:
            logger.debug("Creating new asyncio event loop to fetch ORS matrices synchronously")
            result = asyncio.run(_fetch())

            # Cerrar sesión interna para evitar warnings de "Unclosed client session".
            try:
                if hasattr(self._async_provider, "_session") and self._async_provider._session:  # type: ignore
                    asyncio.run(self._async_provider._session.close())  # type: ignore[attr-defined]
            except Exception as exc:
                logger.debug("No se pudo cerrar sesión ORS: %s", exc)

        # Build legacy output structure -------------------------------------------
        output: Dict[str, Any] = {}
        if "duration" in metrics:
            output["duration"] = result.durations
        if "distance" in metrics:
            output["distance"] = result.distances
        return output

# Convenience alias expected by some code (mirrors previous naming exactly)
__all__ = ["OrsMatrixProvider"]
