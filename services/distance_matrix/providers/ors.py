import logging
import time
import os
import aiohttp
import asyncio
from typing import List, Dict, Any, Literal, Optional

from ..base import MatrixProvider, MatrixResult

# --- Configuration ---
ORS_BASE_URL = "https://api.openrouteservice.org/v2/matrix/"
ORS_MAX_LOCATIONS = 50
ORS_REQUEST_INTERVAL_SECONDS = 1.5  # Corresponds to 40 requests/minute

logger = logging.getLogger(__name__)

class ORSMatrix(MatrixProvider):
    """
    Provides distance and duration matrices using the OpenRouteService (ORS) API.

    This provider handles API authentication, asynchronous requests, automatic
    chunking for large matrices, and rate limiting to comply with ORS usage policies.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = ORS_BASE_URL):
        """
        Initializes the provider.
        If api_key is provided, it's used directly.
        Otherwise, it attempts to load the ORS API key from the 'ORS_API_KEY' environment variable.

        Args:
            api_key: The ORS API key. If None, it will be loaded from 'ORS_API_KEY' env var.
            base_url: The base URL for the ORS API.

        Raises:
            ValueError: If no API key is provided and 'ORS_API_KEY' environment variable is not set.
        """
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('ORS_API_KEY')
        
        logger.debug(f"ORSMatrix: Initializing with API key: {'******' if self.api_key else 'NOT FOUND'}")
        
        if not self.api_key:
            logger.critical("ORS API key was not provided directly and 'ORS_API_KEY' environment variable is not set.")
            raise ValueError("ORS API key not found (neither provided nor in env var ORS_API_KEY).")
        
        self.base_url = base_url
        self._last_request_time = 0
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _rate_limit(self):
        current_time = time.monotonic()
        time_since_last = current_time - self._last_request_time
        if time_since_last < ORS_REQUEST_INTERVAL_SECONDS:
            sleep_time = ORS_REQUEST_INTERVAL_SECONDS - time_since_last
            logger.debug(f"Rate limiting: waiting for {sleep_time:.2f} seconds.")
            await asyncio.sleep(sleep_time)
        self._last_request_time = time.monotonic()

    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        profile: str,
        locations: List[List[float]],
        metrics: List[str],
        sources: List[int],
        destinations: List[int]
    ) -> Dict[str, Any]:
        await self._rate_limit()
        url = f"{ORS_BASE_URL}{profile}"
        headers = {
            'Authorization': self.api_key,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        }
        json_payload = {
            "locations": locations,
            "metrics": metrics,
            "sources": sources,
            "destinations": destinations,
            "units": "m"
        }
        logger.debug(f"ORSMatrix._make_request: URL: {url}")
        logger.debug(f"ORSMatrix._make_request: Payload: {json_payload}")

        logger.debug(f"Requesting ORS chunk. Sources: {len(sources)}, Dests: {len(destinations)}")
        try:
            async with session.post(url, json=json_payload, headers=headers, timeout=40) as response:
                logger.debug(f"ORSMatrix._make_request: Response status: {response.status}")
                if not response.ok:  # Check if status is 4xx or 5xx
                    error_body_text = "<Could not read error body>"
                    try:
                        error_body_text = await response.text()
                    except Exception as read_ex:
                        logger.error(f"ORSMatrix._make_request: Could not read error response body on non-ok response: {read_ex}")
                    logger.error(f"ORS API Error (pre-raise): Status {response.status}. URL: {response.url}. Response Body: {error_body_text}")
                    response.raise_for_status() # Now raise, will be caught by ClientResponseError
                
                response_json = await response.json()
                logger.debug(f"ORSMatrix._make_request: Response JSON (first 200 chars): {str(response_json)[:200]}")
                return response_json
        except aiohttp.ClientResponseError as e: # This will catch the error raised by response.raise_for_status()
            # The detailed body should have been logged above if reading was successful
            # Log a simpler message here, or re-log e.message if it contains useful info from aiohttp
            logger.error(f"ORS API ClientResponseError: Status {e.status}, Message: '{e.message}'. URL: {e.request_info.url}. Check previous logs for response body.")
            raise Exception(f"ORS API request failed with status {e.status}. Message: '{e.message}'. See logs for details.") from e
        except asyncio.TimeoutError as e:
            logger.error("ORS API request timed out.")
            raise Exception("Timeout connecting to ORS API.") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during ORS request: {e}", exc_info=True)
            raise

    def _process_chunk_response(
        self,
        response_data: Dict[str, Any],
        source_indices: List[int],
        dest_indices: List[int],
        distance_matrix: List[List[float]],
        duration_matrix: List[List[float]]
    ):
        logger.debug(f"ORSMatrix._process_chunk_response: Processing chunk for sources {source_indices} and destinations {dest_indices}.")
        distances = response_data.get('distances')
        durations = response_data.get('durations')
        logger.debug(f"ORSMatrix._process_chunk_response: Distances in chunk: {distances is not None}, Durations in chunk: {durations is not None}")

        if distances:
            for i, original_source_idx in enumerate(source_indices):
                for j, original_dest_idx in enumerate(dest_indices):
                    if i < len(distances) and j < len(distances[i]):
                        distance_matrix[original_source_idx][original_dest_idx] = distances[i][j]

        if durations:
            for i, original_source_idx in enumerate(source_indices):
                for j, original_dest_idx in enumerate(dest_indices):
                     if i < len(durations) and j < len(durations[i]):
                        duration_matrix[original_source_idx][original_dest_idx] = durations[i][j]

    async def get_matrix(
        self,
        locations: List[Dict[str, float]],
        profile: str = 'driving-car',
        metrics: Optional[List[Literal['distances', 'durations']]] = None,
    ) -> MatrixResult:
        # 'metrics' initially refers to the input parameter of the get_matrix method.
        # We'll use 'metrics_input_param' to hold its value to avoid confusion.
        metrics_input_param = metrics 
        logger.info(f"ORSMatrix.get_matrix: Requesting matrix for {len(locations)} locations, profile '{profile}', original metrics parameter: {metrics_input_param}.")
        
        # Determine effective metrics (plural form initially for user input compatibility)
        effective_metrics_plural = metrics_input_param if metrics_input_param else ['distances', 'durations']
        
        # Convert to singular form for ORS API and internal logic.
        # The result will be stored back into the 'metrics' variable, which is assumed
        # to be used by _make_request later in this method.
        processed_singular_metrics = [] 
        for m in effective_metrics_plural:
            if m == "distances":
                processed_singular_metrics.append("distance")
            elif m == "durations":
                processed_singular_metrics.append("duration")
            else:
                # If user provides already singular or other valid ORS metrics
                processed_singular_metrics.append(m) 
        
        metrics = processed_singular_metrics # CRITICAL: 'metrics' now holds the singular list for _make_request
        logger.debug(f"ORSMatrix.get_matrix: Processed metrics for ORS API (now in 'metrics' var): {metrics}")

        session = await self._get_session()
        num_locations = len(locations)
        coords_for_ors = [[loc['lng'], loc['lat']] for loc in locations]
        logger.debug(f"ORSMatrix.get_matrix: Coords for ORS (first 5): {coords_for_ors[:5]}")

        # Initialize matrices based on the 'metrics' (which now holds processed singular values) list
        distance_matrix = [[0.0] * num_locations for _ in range(num_locations)] if 'distance' in metrics else []
        duration_matrix = [[0.0] * num_locations for _ in range(num_locations)] if 'duration' in metrics else []

        tasks = []
        for i in range(0, num_locations, ORS_MAX_LOCATIONS):
            source_chunk_indices = list(range(i, min(i + ORS_MAX_LOCATIONS, num_locations)))
            for j in range(0, num_locations, ORS_MAX_LOCATIONS):
                dest_chunk_indices = list(range(j, min(j + ORS_MAX_LOCATIONS, num_locations)))

                if not source_chunk_indices or not dest_chunk_indices:
                    continue
                
                task = asyncio.create_task(self._make_request(
                    session=session,
                    profile=profile,
                    locations=coords_for_ors,
                    metrics=metrics,
                    sources=source_chunk_indices,
                    destinations=dest_chunk_indices
                ))
                tasks.append((task, source_chunk_indices, dest_chunk_indices))

        logger.debug(f"ORSMatrix.get_matrix: Created {len(tasks)} tasks for ORS requests.")
        all_results = await asyncio.gather(*(t[0] for t in tasks), return_exceptions=True)
        logger.debug(f"ORSMatrix.get_matrix: asyncio.gather completed. Number of results: {len(all_results)}.")

        for result, source_indices, dest_indices in zip(all_results, (t[1] for t in tasks), (t[2] for t in tasks)):
            if isinstance(result, Exception):
                logger.error(f"ORSMatrix.get_matrix: A chunk request (sources: {source_indices}, dests: {dest_indices}) failed: {result}", exc_info=result)
                # Decide if you want to raise immediately or try to continue with partial data / fallback
                # For now, let's re-raise to see the error clearly.
                raise result
            
            self._process_chunk_response(
                result, source_indices, dest_indices, distance_matrix, duration_matrix
            )

        logger.info(f"ORSMatrix.get_matrix: Successfully processed all chunks. Returning MatrixResult.")
        logger.debug(f"ORSMatrix.get_matrix: Final distance_matrix (sample): {distance_matrix[0][:5] if distance_matrix and distance_matrix[0] else 'N/A'}")
        logger.debug(f"ORSMatrix.get_matrix: Final duration_matrix (sample): {duration_matrix[0][:5] if duration_matrix and duration_matrix[0] else 'N/A'}")
        return MatrixResult(
            distances=distance_matrix,
            durations=duration_matrix,
            provider='ors'
        )

    async def get_distance_matrix(
        self,
        locations: List[Dict[str, float]],
        **kwargs
    ) -> List[List[float]]:
        profile = kwargs.get('profile', 'driving-car')
        result = await self.get_matrix(locations, profile=profile, metrics=['distances'])
        return result.distances

    async def get_duration_matrix(
        self,
        locations: List[Dict[str, float]],
        **kwargs
    ) -> List[List[float]]:
        profile = kwargs.get('profile', 'driving-car')
        result = await self.get_matrix(locations, profile=profile, metrics=['durations'])
        return result.durations

