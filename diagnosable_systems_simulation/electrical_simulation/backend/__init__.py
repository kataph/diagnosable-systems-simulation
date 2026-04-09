"""
Simulation backends.

``PySpiceBackend`` requires ``pip install diagnosable-systems-simulation[spice]``.
"""
from diagnosable_systems_simulation.electrical_simulation.backend.base import SimulationBackend

__all__ = ["SimulationBackend", "PySpiceBackend"]


def __getattr__(name: str):
    if name == "PySpiceBackend":
        from diagnosable_systems_simulation.electrical_simulation.backend.spice import PySpiceBackend
        return PySpiceBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
