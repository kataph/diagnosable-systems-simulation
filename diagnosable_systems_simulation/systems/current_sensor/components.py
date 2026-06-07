"""
Component definitions for the current sensor lamp system.

Physical layout (x-axis, left to right):
  Power supply cube   x=0.00
  Control cube        x=0.15
  Load cube           x=0.30

Call ``create_components()`` to get a fresh, independent set of component
instances.  Never share instances across ``DiagnosableSystem`` objects.
"""
from __future__ import annotations

from types import SimpleNamespace

from diagnosable_systems_simulation.systems.shared.module_builders import (
    create_current_sensor_ctrl_module,
    create_ambient_load_module,
    create_psu_module,
)


def create_components() -> SimpleNamespace:
    """
    Build and return a fresh, fully independent set of component instances.

    Returns a ``SimpleNamespace`` whose attributes are the individual
    component objects; ``.ALL`` is a ``{component_id: component}`` dict.
    """
    psu  = create_psu_module(x_left=0.00)
    ctrl = create_current_sensor_ctrl_module(prefix="ctrl", x_left=0.15)
    load = create_ambient_load_module(x_left=0.30)

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
        ctrl_relay=ctrl.relay,
        ctrl_green_led=ctrl.green_led,
        ctrl_green_resistor=ctrl.green_resistor,
        ctrl_cable_in_pos=ctrl.cable_in_pos,
        ctrl_cable_in_neg=ctrl.cable_in_neg,
        ctrl_cable_out_pos=ctrl.cable_out_pos,
        ctrl_cable_out_neg=ctrl.cable_out_neg,
        main_bulb=load.main_bulb,
        internal_bulb=load.internal_bulb,
        load_diode=load.diode,
        load_cable_pos=load.cable_pos,
        load_cable_neg=load.cable_neg,
        load_panel=load.panel,
    )
    ns.ALL = {**psu.ALL, **ctrl.ALL, **load.ALL}
    return ns
