import pytest
from fastapi import HTTPException
from vrp_utils import (
    validate_full_request, add_warning, warn_no_viable_clients, VRPConstants
)

class DummyLocation:
    def __init__(self, lat=0, lon=0, name="A", time_window=None, required_skills=None, weight=0, volume=0, demand=0, id=None):
        self.lat = lat
        self.lon = lon
        self.name = name
        self.time_window = time_window
        self.required_skills = required_skills or []
        self.weight = weight
        self.volume = volume
        self.demand = demand
        self.service_time = 5
        self.id = id if id is not None else name

class DummyVehicle:
    def __init__(self, provided_skills=None, capacity_weight=100, capacity_volume=100, capacity_quantity=100, start_time=420, end_time=1080):
        self.provided_skills = provided_skills or []
        self.capacity_weight = capacity_weight
        self.capacity_volume = capacity_volume
        self.capacity_quantity = capacity_quantity
        self.start_time = start_time
        self.end_time = end_time

def make_request(num_locations=2, num_vehicles=1, **kwargs):
    class DummyRequest:
        pass
    req = DummyRequest()
    req.locations = [DummyLocation(name=f"Loc{i}", id=i) for i in range(num_locations)]
    req.vehicles = [DummyVehicle() for _ in range(num_vehicles)]
    req.num_vehicles = num_vehicles
    req.depot = 0
    req.strict_mode = kwargs.get("strict_mode", False)
    return req

def test_validate_full_request_ok():
    req = make_request()
    is_valid, warnings, diagnostics = validate_full_request(req)
    assert is_valid
    assert warnings == []
    assert diagnostics == {}

def test_validate_full_request_too_many_locations():
    req = make_request(num_locations=VRPConstants.MAX_LOCATIONS + 1)
    is_valid, warnings, diagnostics = validate_full_request(req)
    assert not is_valid
    assert any(w["code"] == "TOO_MANY_LOCATIONS" for w in warnings)
    assert diagnostics["max_locations"] == VRPConstants.MAX_LOCATIONS

def test_validate_full_request_invalid_depot():
    req = make_request()
    req.depot = 99
    is_valid, warnings, diagnostics = validate_full_request(req)
    assert not is_valid
    assert any(w["code"] == "INVALID_DEPOT_INDEX" for w in warnings)

def test_add_warning_and_warn_no_viable_clients():
    warnings = []
    add_warning(warnings, code="CUSTOM", message="Mensaje de prueba", context={"foo": 1})
    assert warnings[0]["code"] == "CUSTOM"
    excl = ["c1", "c2"]
    warn_no_viable_clients(warnings, excl)
    found = any(w["code"] == "NO_VIABLE_CLIENTS" for w in warnings)
    assert found

def test_validate_full_request_only_depot():
    req = make_request(num_locations=1)
    is_valid, warnings, diagnostics = validate_full_request(req)
    assert not is_valid
    assert any(w["code"] == "ONLY_DEPOT" for w in warnings)
    assert diagnostics["routes"] == []
    assert diagnostics["total_distance"] == 0

def test_validate_full_request_strict_mode_skills():
    req = make_request()
    req.strict_mode = True
    # Hacer que la ubicación requiera un skill que ningún vehículo tiene
    req.locations[1].required_skills = ["foo"]
    is_valid, warnings, diagnostics = validate_full_request(req)
    assert not is_valid
    assert any(w["code"] == "SKILLS_NOT_COVERED" for w in warnings)

def test_validate_full_request_invalid_time_window():
    req = make_request()
    req.locations[1].time_window = [1100, 1000]  # fin < inicio
    is_valid, warnings, diagnostics = validate_full_request(req)
    assert not is_valid
    assert any(w["code"] == "VALIDATION_ERROR" for w in warnings)


def test_validate_full_request_capacity_exceeded():
    req = make_request()
    req.strict_mode = True
    req.locations[1].weight = 1000  # Excede la capacidad del vehículo
    is_valid, warnings, diagnostics = validate_full_request(req)
    assert not is_valid
    assert any(w["code"] == "CAPACITY_NOT_COVERED" for w in warnings)


def test_validate_full_request_skills_mixed():
    req = make_request(num_vehicles=2)
    req.strict_mode = True
    req.vehicles[0].provided_skills = ["foo"]
    req.vehicles[1].provided_skills = ["bar"]
    req.locations[1].required_skills = ["bar"]
    is_valid, warnings, diagnostics = validate_full_request(req)
    assert is_valid  # Al menos un vehículo cubre el skill


def test_validate_full_request_vehicle_missing():
    req = make_request(num_vehicles=0)
    is_valid, warnings, diagnostics = validate_full_request(req)
    assert not is_valid
    assert any(w["code"] == "VALIDATION_ERROR" for w in warnings)


def test_validate_full_request_demand_exceeded():
    req = make_request()
    req.strict_mode = True
    req.locations[1].demand = 9999  # Excede la capacidad_quantity
    is_valid, warnings, diagnostics = validate_full_request(req)
    assert not is_valid
    assert any(w["code"] == "CAPACITY_NOT_COVERED" for w in warnings)

# Puedes agregar más tests para capacidades, ventanas, etc.
