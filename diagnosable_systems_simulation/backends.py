"""
Convenience re-export of simulation backends.

Usage::

    from diagnosable_systems_simulation.backends import PySpiceBackend  # needs [spice]
"""
from diagnosable_systems_simulation.electrical_simulation.backend import (
    SimulationBackend,
    PySpiceBackend,  # lazy-loaded; ImportError raised only when actually used
)

__all__ = ["SimulationBackend", "PySpiceBackend"]
