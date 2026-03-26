"""
Simulation backends.

``StubBackend`` is always available (pure numpy, no extra dependencies).
``PySpiceBackend`` requires ``pip install diagnosable-systems-simulation[spice]``.
"""
from diagnosable_systems_simulation.electrical_simulation.backend.stub import StubBackend
from diagnosable_systems_simulation.electrical_simulation.backend.base import SimulationBackend

__all__ = ["SimulationBackend", "StubBackend", "PySpiceBackend"]


def __getattr__(name: str):
    if name == "PySpiceBackend":
        from diagnosable_systems_simulation.electrical_simulation.backend.spice import PySpiceBackend
        return PySpiceBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
