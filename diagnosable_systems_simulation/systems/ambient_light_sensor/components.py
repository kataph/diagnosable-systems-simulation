"""
Component definitions for the ambient light sensor lamp system.

Physical layout (z-axis stacking, bottom → top):
  Power supply cube   z=0.05 m
  Control cube        z=0.20 m
  Load cube           z=0.35 m

Each cube is 0.10 m on a side.

Call ``create_components()`` to get a fresh, independent set of component
instances.  Never share instances across ``DiagnosableSystem`` objects.
"""
from __future__ import annotations

from types import SimpleNamespace

from diagnosable_systems_simulation.systems.shared.module_builders import (
    create_ambient_ctrl_module,
    create_load_module,
    create_psu_module,
)


def create_components() -> SimpleNamespace:
    """
    Build and return a fresh, fully independent set of component instances.

    Every call returns new objects with no shared mutable state, so multiple
    ``DiagnosableSystem`` instances can coexist without interfering.

    Returns a ``SimpleNamespace`` whose attributes are the individual
    component objects; ``.ALL`` is a ``{component_id: component}`` dict.
    """
    psu  = create_psu_module(x_left=0.00)
    ctrl = create_ambient_ctrl_module(prefix="ctrl")
    load = create_load_module(x_left=0.00)

    ns = SimpleNamespace(
        module_psu=psu.module,
        module_ctrl=ctrl.module,
        module_load=load.module,
        cube_psu=psu.cube,
        cube_ctrl=ctrl.cube,
        cube_load=load.cube,
        battery=psu.source,
        battery_internal_resistor=psu.battery_internal_resistor,
        psu_green_led=psu.green_led,
        psu_green_resistor=psu.green_resistor,
        psu_cable_pos=psu.cable_pos,
        psu_cable_neg=psu.cable_neg,
        ctrl_panel=ctrl.panel,
        ctrl_light_sensor=ctrl.light_sensor,
        ctrl_relay=ctrl.relay,
        ctrl_sensitivity_pot=ctrl.sensitivity_pot,
        ctrl_timing_pot=ctrl.timing_pot,
        ctrl_sensor_bias=ctrl.sensor_bias,
        ctrl_cable_in_pos=ctrl.cable_in_pos,
        ctrl_cable_in_neg=ctrl.cable_in_neg,
        ctrl_cable_out_pos=ctrl.cable_out_pos,
        ctrl_cable_out_neg=ctrl.cable_out_neg,
        main_bulb=load.main_bulb,
        internal_bulb=load.internal_bulb,
        load_diode=load.diode,
        load_cable_pos=load.cable_pos,
        load_cable_neg=load.cable_neg,
        load_peephole=load.peephole,
    )
    ns.ALL = {**psu.ALL, **ctrl.ALL, **load.ALL}
    return ns
