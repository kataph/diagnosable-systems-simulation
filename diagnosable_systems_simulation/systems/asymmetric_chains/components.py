"""
Component definitions for the asymmetric chains lamp system.

Physical layout (x-axis, two parallel chains):
  Chain 1 (y=0.00):  PSU1 x=0.00 → CTRL1 x=0.15 → LOAD1 x=0.30
  Chain 2 (y=0.20):  PSU2 x=0.00 → CTRL2 x=0.15 → CTRL3 x=0.30 → LOAD2 x=0.45

The chains are cross-linked:
  PSU2  output also feeds CTRL1 input  (via diode_psu2_ctrl1)
  CTRL1 output also feeds CTRL3 input  (via diode_ctrl1_ctrl3)

Four inter-module diodes enforce unidirectional current flow on the positive
lines between modules (see factory.py for exact wiring).

Call ``create_components()`` to get a fresh, independent set of component
instances.  Never share instances across ``DiagnosableSystem`` objects.
"""
from __future__ import annotations

from types import SimpleNamespace

from diagnosable_systems_simulation.world.affordances import Affordance, AffordanceSet
from diagnosable_systems_simulation.world.components import Diode
from diagnosable_systems_simulation.world.spatial import Position
from diagnosable_systems_simulation.systems.shared.module_builders import (
    create_10cubes_control_module,
    create_load_module,
    create_psu_module,
)


def _make_intermodule_diode(component_id: str, display_name: str, x: float, y: float) -> Diode:
    d = Diode(
        component_id=component_id,
        display_name=display_name,
        forward_voltage=0.7,
        position=Position(x, y, 0.05),
    )
    d.affordances = AffordanceSet(
        static={Affordance.MEASURABLE, Affordance.REPLACEABLE, Affordance.OBSERVABLE},
    )
    return d


def create_components() -> SimpleNamespace:
    """
    Build and return a fresh, fully independent set of component instances.

    Returns a ``SimpleNamespace`` whose attributes are the individual
    component objects; ``.ALL`` is a ``{component_id: component}`` dict.
    """
    psu1  = create_psu_module(x_left=0.00, prefix="psu1")
    psu2  = create_psu_module(x_left=0.00, prefix="psu2")
    ctrl1 = create_10cubes_control_module(prefix="ctrl1", x_left=0.15, label="1")
    ctrl2 = create_10cubes_control_module(prefix="ctrl2", x_left=0.15, label="2")
    ctrl3 = create_10cubes_control_module(prefix="ctrl3", x_left=0.30, label="3")
    load1 = create_load_module(x_left=0.30, prefix="load1")
    load2 = create_load_module(x_left=0.45, prefix="load2")

    # Inter-module protection diodes (on the positive lines)
    diode_psu1_ctrl1  = _make_intermodule_diode("diode_psu1_ctrl1",  "Diode PSU1→CTRL1",  x=0.12, y=0.00)
    diode_psu2_ctrl1  = _make_intermodule_diode("diode_psu2_ctrl1",  "Diode PSU2→CTRL1",  x=0.12, y=0.10)
    diode_ctrl1_ctrl3 = _make_intermodule_diode("diode_ctrl1_ctrl3", "Diode CTRL1→CTRL3", x=0.27, y=0.00)
    diode_ctrl2_ctrl3 = _make_intermodule_diode("diode_ctrl2_ctrl3", "Diode CTRL2→CTRL3", x=0.27, y=0.10)

    extra_diodes = {
        "diode_psu1_ctrl1":  diode_psu1_ctrl1,
        "diode_psu2_ctrl1":  diode_psu2_ctrl1,
        "diode_ctrl1_ctrl3": diode_ctrl1_ctrl3,
        "diode_ctrl2_ctrl3": diode_ctrl2_ctrl3,
    }

    ns = SimpleNamespace(
        # Modules
        module_psu1=psu1.module,   module_psu2=psu2.module,
        module_ctrl1=ctrl1.module, module_ctrl2=ctrl2.module, module_ctrl3=ctrl3.module,
        module_load1=load1.module, module_load2=load2.module,
        # Cubes
        cube_psu1=psu1.cube,   cube_psu2=psu2.cube,
        cube_ctrl1=ctrl1.cube, cube_ctrl2=ctrl2.cube, cube_ctrl3=ctrl3.cube,
        cube_load1=load1.cube, cube_load2=load2.cube,
        # PSU1
        psu1_battery=psu1.source,
        psu1_battery_internal_resistor=psu1.battery_internal_resistor,
        psu1_green_led=psu1.green_led,
        psu1_green_resistor=psu1.green_resistor,
        psu1_cable_pos=psu1.cable_pos,
        psu1_cable_neg=psu1.cable_neg,
        # PSU2
        psu2_battery=psu2.source,
        psu2_battery_internal_resistor=psu2.battery_internal_resistor,
        psu2_green_led=psu2.green_led,
        psu2_green_resistor=psu2.green_resistor,
        psu2_cable_pos=psu2.cable_pos,
        psu2_cable_neg=psu2.cable_neg,
        # CTRL1
        ctrl1_switch=ctrl1.switch,
        ctrl1_green_led=ctrl1.green_led,
        ctrl1_green_resistor=ctrl1.green_resistor,
        ctrl1_cable_in_pos=ctrl1.cable_in_pos,
        ctrl1_cable_in_neg=ctrl1.cable_in_neg,
        ctrl1_cable_out_pos=ctrl1.cable_out_pos,
        ctrl1_cable_out_neg=ctrl1.cable_out_neg,
        # CTRL2
        ctrl2_switch=ctrl2.switch,
        ctrl2_green_led=ctrl2.green_led,
        ctrl2_green_resistor=ctrl2.green_resistor,
        ctrl2_cable_in_pos=ctrl2.cable_in_pos,
        ctrl2_cable_in_neg=ctrl2.cable_in_neg,
        ctrl2_cable_out_pos=ctrl2.cable_out_pos,
        ctrl2_cable_out_neg=ctrl2.cable_out_neg,
        # CTRL3
        ctrl3_switch=ctrl3.switch,
        ctrl3_green_led=ctrl3.green_led,
        ctrl3_green_resistor=ctrl3.green_resistor,
        ctrl3_cable_in_pos=ctrl3.cable_in_pos,
        ctrl3_cable_in_neg=ctrl3.cable_in_neg,
        ctrl3_cable_out_pos=ctrl3.cable_out_pos,
        ctrl3_cable_out_neg=ctrl3.cable_out_neg,
        # LOAD1
        load1_main_bulb=load1.main_bulb,
        load1_internal_bulb=load1.internal_bulb,
        load1_diode=load1.diode,
        load1_cable_pos=load1.cable_pos,
        load1_cable_neg=load1.cable_neg,
        load1_peephole=load1.peephole,
        # LOAD2
        load2_main_bulb=load2.main_bulb,
        load2_internal_bulb=load2.internal_bulb,
        load2_diode=load2.diode,
        load2_cable_pos=load2.cable_pos,
        load2_cable_neg=load2.cable_neg,
        load2_peephole=load2.peephole,
        # Inter-module diodes
        diode_psu1_ctrl1=diode_psu1_ctrl1,
        diode_psu2_ctrl1=diode_psu2_ctrl1,
        diode_ctrl1_ctrl3=diode_ctrl1_ctrl3,
        diode_ctrl2_ctrl3=diode_ctrl2_ctrl3,
    )
    ns.ALL = {
        **psu1.ALL, **psu2.ALL,
        **ctrl1.ALL, **ctrl2.ALL, **ctrl3.ALL,
        **load1.ALL, **load2.ALL,
        **extra_diodes,
    }
    return ns
