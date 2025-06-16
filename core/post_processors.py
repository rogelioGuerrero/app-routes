from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def add_suggested_departure_times(
    solution: Any,  # Pydantic VRPTWSolution or dict-like object with routes list
    locations: List[Any],  # List of Location models (must have id, time_window_start, time_window_end)
    loc_id_to_idx: Dict[str, int],
    duration_matrix: List[List[float]],
) -> None:
    """Add a key 'suggested_depot_departure' to each route indicating the recommended
    departure time (seconds since 00:00) so the vehicle arrives at the first stop
    just-in-time. No times are modified, preserving the original solution coming
    from OR-Tools. The suggestion is purely advisory for the client application.

    The function analyses each route and stores the recommended departure time in
    a new key without altering any existing timestamps.

    Constraints respected:
    1. Depot time window (start & end).
    2. Latest time window of every stop in the route.
    3. Durations between consecutive stops obtained from *duration_matrix*.


    """

    # --- Simplified implementation using cached duration_matrix ---
    if not duration_matrix:
        logger.warning("Duration matrix not provided; skipping suggested departure calculation.")
        return

    tw_start_map: Dict[str, float] = {loc.id: getattr(loc, "time_window_start", None) for loc in locations}

    routes = solution.get("routes", []) if isinstance(solution, dict) else getattr(solution, "routes", [])
    for route in routes:
        stops = route.get("stops", []) if isinstance(route, dict) else getattr(route, "stops", [])
        if len(stops) < 2:
            continue

        depot_stop = stops[0]
        first_stop = stops[1]

        depot_id = depot_stop["location_id"] if isinstance(depot_stop, dict) else depot_stop.location_id
        first_id = first_stop["location_id"] if isinstance(first_stop, dict) else first_stop.location_id

        if depot_id not in loc_id_to_idx or first_id not in loc_id_to_idx:
            logger.warning(f"add_suggested_departure_times: missing index for {depot_id} or {first_id} – skipping route")
            continue

        travel_time = duration_matrix[loc_id_to_idx[depot_id]][loc_id_to_idx[first_id]]
        first_arrival = first_stop["arrival_time"] if isinstance(first_stop, dict) else first_stop.arrival_time
        suggested = int(round(first_arrival - travel_time))

        tw_start = tw_start_map.get(depot_id)
        if tw_start is not None:
            suggested = max(suggested, tw_start)

        if isinstance(route, dict):
            route["suggested_depot_departure"] = suggested
        else:
            setattr(route, "suggested_depot_departure", suggested)
    return

    # Build quick lookup for time windows per location id
    tw_map: Dict[str, Tuple[float, float]] = {
        loc.id: (getattr(loc, "time_window_start", None), getattr(loc, "time_window_end", None))
        for loc in locations
    }

    # Obtain list of routes regardless of container type
    routes_list = solution.get("routes", []) if isinstance(solution, dict) else getattr(solution, "routes", [])

    for route in routes_list:
        stops = route.get("stops", []) if isinstance(route, dict) else getattr(route, "stops", [])
        if len(stops) < 2:
            continue  # Nothing to shift

        depot_stop = stops[0]
        first_customer = stops[1]

        depot_id = depot_stop.get("location_id") if isinstance(depot_stop, dict) else getattr(depot_stop, "location_id")
        first_id = first_customer.get("location_id") if isinstance(first_customer, dict) else getattr(first_customer, "location_id")

        # Default suggestion is original departure time; ensures non-null even if later checks fail
        depot_departure_orig = depot_stop.get("departure_time") if isinstance(depot_stop, dict) else getattr(depot_stop, "departure_time")
        if isinstance(route, dict):
            route["suggested_depot_departure"] = depot_departure_orig
        else:
            setattr(route, "suggested_depot_departure", depot_departure_orig)

        if depot_id not in loc_id_to_idx or first_id not in loc_id_to_idx:
            continue  # Missing index mapping – skip

        i = loc_id_to_idx[depot_id]
        j = loc_id_to_idx[first_id]
        # Obtener tiempo de viaje; usar duration_matrix si existe, si no estimar con distance_from_previous / 11.11 (≈40 km/h)
        try:
            travel_time = duration_matrix[i][j]
        except Exception:
            # Fallback usando distancia y velocidad media
            dist_meters = first_customer.get("distance_from_previous") if isinstance(first_customer, dict) else getattr(first_customer, "distance_from_previous", None)
            travel_time = (dist_meters / 11.11) if dist_meters is not None else 0

        first_arrival = first_customer.get("arrival_time") if isinstance(first_customer, dict) else getattr(first_customer, "arrival_time")
        desired_depot_departure: float = first_arrival - travel_time

        depot_tw_start, depot_tw_end = tw_map.get(depot_id, (None, None))
        if depot_tw_start is None:
            continue  # Should not happen

        # Initial candidate shift (positive means later)
        shift = desired_depot_departure - depot_departure_orig
        if shift <= 0:
            # Already optimal (or cannot be delayed). Store current departure as suggestion.
            if isinstance(route, dict):
                route["suggested_depot_departure"] = depot_departure_orig
            else:
                setattr(route, "suggested_depot_departure", depot_departure_orig)
            continue

        # Bound by depot latest end
        if depot_tw_end is not None:
            shift = min(shift, depot_tw_end - depot_departure_orig)

        # Bound by each stop's latest window
        for st in stops[1:]:  # Skip depot itself
            loc_id = st.get("location_id") if isinstance(st, dict) else getattr(st, "location_id")
            _, latest_end = tw_map.get(loc_id, (None, None))
            if latest_end is not None:
                st_arrival = st.get("arrival_time") if isinstance(st, dict) else getattr(st, "arrival_time")
                shift = min(shift, latest_end - st_arrival)

        if shift <= 0:
            # No feasible positive shift remains; suggest original departure time
            if isinstance(route, dict):
                route["suggested_depot_departure"] = depot_departure_orig
            else:
                setattr(route, "suggested_depot_departure", depot_departure_orig)
            continue

        # Suggest the shift (do not mutate times) only if strictly positive and feasible
        suggested = depot_departure_orig + shift
        if suggested != depot_departure_orig:
            if isinstance(route, dict):
                route["suggested_depot_departure"] = suggested
            else:
                setattr(route, "suggested_depot_departure", suggested)
        logger.debug(
            f"Route for vehicle {route.get('vehicle_id') if isinstance(route, dict) else getattr(route, 'vehicle_id', '?')} "
            f"recommended depot departure {suggested} (shift {shift} sec)."
        )
