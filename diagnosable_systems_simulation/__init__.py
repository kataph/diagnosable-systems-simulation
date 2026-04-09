"""
diagnosable_systems_simulation
==============================

Framework for simulating diagnosable physical systems with fault injection
and diagnostic actions.

Quick start (requires ``pip install diagnosable-systems-simulation[spice]``)::

    from diagnosable_systems_simulation import build_three_cubes_system
    from diagnosable_systems_simulation.actions import MeasureVoltage, DisconnectCable
    from diagnosable_systems_simulation.backends import PySpiceBackend

    system = build_three_cubes_system(backend=PySpiceBackend(), extra_tools={"multimeter"})
    result = system.simulate()
    print(result.emitting_light)

    r = system.apply_action(MeasureVoltage(), {"subject": system.component("main_bulb")})
    print(r.message)
"""
from diagnosable_systems_simulation.systems.three_cubes import build_three_cubes_system
from diagnosable_systems_simulation.systems.ten_cubes import build_ten_cubes_system
from diagnosable_systems_simulation.systems.base_system import DiagnosableSystem
from diagnosable_systems_simulation.electrical_simulation.results import SimulationResult

__all__ = [
    "build_three_cubes_system",
    "build_ten_cubes_system",
    "DiagnosableSystem",
    "SimulationResult",
]
