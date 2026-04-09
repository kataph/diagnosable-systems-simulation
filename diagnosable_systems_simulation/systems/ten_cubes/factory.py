"""
Factory that assembles the complete 10-cubes lamp system.

Usage::

    from diagnosable_systems_simulation.systems.ten_cubes.factory import build_ten_cubes_system

    system = build_ten_cubes_system()
    result = system.simulate()

Topology summary
----------------
PSU → ctrl1 → ctrl2 → … → ctrl8 → Load  (series chain on the +12 V line)

Each control module (ctrl1 … ctrl8) contains:
  • switch  (in series on the +12 V path)
  • green LED + resistor  (power-flow indicator)
    anode → resistor → same net as switch.n and ctrl_cable_out_pos
    cathode → ground net

  Unlike the 3-cubes control module there is NO red inverted-cables LED and
  NO protection diode inside the control cube.

The load module contains main bulb and internal bulb (parallel) plus the
protection diode on the +12 V input line — same position as in 3-cubes.
"""
from __future__ import annotations

from diagnosable_systems_simulation.electrical_simulation.backend.spice import PySpiceBackend
from diagnosable_systems_simulation.electrical_simulation.solver import SimulationRunner
from diagnosable_systems_simulation.systems.base_system import DiagnosableSystem
from diagnosable_systems_simulation.world.context import WorldContext
from diagnosable_systems_simulation.world.knowledge_graph import (
    EntityType, RelationType, SystemGraph,
)

from .components import create_components


def _build_kg() -> SystemGraph:
    kg = SystemGraph()
    c = create_components()

    psu       = c.psu
    ctrl_mods = c.ctrl_mods   # list[SimpleNamespace], index 0 = ctrl module 1
    load      = c.load

    # ── Component entities ─────────────────────────────────────────────
    for cid, comp in c.ALL.items():
        kg.add_entity(cid, EntityType.COMPONENT, comp)

    def part_of(*comps, module):
        for comp in comps:
            kg.add_edge(comp.component_id, module.component_id, RelationType.PART_OF)

    def contained_in(*comps, enclosure):
        for comp in comps:
            kg.add_edge(comp.component_id, enclosure.component_id, RelationType.CONTAINED_IN)

    EC = RelationType.ELECTRICALLY_CONNECTED

    def wire(port_a, port_b, **kw):
        kg.add_edge(
            port_a.component.component_id, port_b.component.component_id, EC,
            from_port=port_a.port_name, to_port=port_b.port_name, **kw,
        )

    # ── PART_OF ────────────────────────────────────────────────────────
    part_of(psu.source, psu.green_led, psu.green_resistor,
            psu.cable_pos, psu.cable_neg, psu.battery_internal_resistor, module=psu.cube)

    for ctrl in ctrl_mods:
        part_of(ctrl.switch, ctrl.green_led, ctrl.green_resistor,
                ctrl.cable_in_pos, ctrl.cable_in_neg,
                ctrl.cable_out_pos, ctrl.cable_out_neg,            module=ctrl.cube)

    part_of(load.main_bulb, load.internal_bulb, load.peephole,
            load.diode, load.cable_pos, load.cable_neg,            module=load.cube)

    # ── CONTAINED_IN ──────────────────────────────────────────────────
    contained_in(psu.source, psu.green_led, psu.green_resistor, psu.battery_internal_resistor,  enclosure=psu.cube)

    for ctrl in ctrl_mods:
        contained_in(ctrl.green_led, ctrl.green_resistor,          enclosure=ctrl.cube)

    contained_in(load.main_bulb, load.internal_bulb,
                 load.diode, load.peephole,                        enclosure=load.cube)

    # ── ELECTRICALLY_CONNECTED ─────────────────────────────────────────
    #
    # Nominal topology (all switches closed, correct polarity):
    #   PSU green LED:   lit  (in parallel with 12V source)
    #   All ctrl green LEDs: lit  (switch.n side is energised)
    #   Main bulb + internal bulb: lit  (in parallel after load diode)

    # Ground net (power_source.neg side)
    wire(psu.source.neg,       psu.green_led.cathode, is_ground=True)
    wire(psu.source.neg,       psu.cable_neg.p)
    # psu_pos net
    wire(psu.source.pos,       psu.battery_internal_resistor.n)
    wire(psu.battery_internal_resistor.p,       psu.green_resistor.p)
    wire(psu.battery_internal_resistor.p,       psu.cable_pos.p)
    # psu_green_mid
    wire(psu.green_resistor.n, psu.green_led.anode)

    # Series chain: PSU → ctrl1 → ctrl2 → … → ctrl8 → load
    wire(psu.cable_pos.n, ctrl_mods[0].cable_in_pos.p)
    wire(psu.cable_neg.n, ctrl_mods[0].cable_in_neg.p)
    for i in range(len(ctrl_mods) - 1):
        wire(ctrl_mods[i].cable_out_pos.n, ctrl_mods[i + 1].cable_in_pos.p)
        wire(ctrl_mods[i].cable_out_neg.n, ctrl_mods[i + 1].cable_in_neg.p)
    wire(ctrl_mods[-1].cable_out_pos.n, load.cable_pos.p)
    wire(ctrl_mods[-1].cable_out_neg.n, load.cable_neg.p)

    # Each control module (green LED on the OUTPUT / switch.n side):
    #
    #   ctrl_in_p  →  switch.p
    #   switch.n   =  ctrl_cable_out_pos.p  =  green_resistor.p  (output +12V net)
    #   green_resistor.n  →  green_led.anode
    #   green_led.cathode  =  ctrl_in_n  =  ctrl_cable_out_neg.p  (ground net)
    for ctrl in ctrl_mods:
        wire(ctrl.cable_in_pos.n,    ctrl.switch.p)
        wire(ctrl.switch.n,          ctrl.cable_out_pos.p)
        wire(ctrl.switch.n,          ctrl.green_resistor.p)
        wire(ctrl.green_resistor.n,  ctrl.green_led.anode)
        wire(ctrl.cable_in_neg.n,    ctrl.green_led.cathode)
        wire(ctrl.cable_in_neg.n,    ctrl.cable_out_neg.p)

    # Load module (protection diode on +12V input line):
    #   load_in_p  →  load_diode.anode
    #   load_diode.cathode  →  main_bulb.p  +  internal_bulb.p
    #   load_in_n  →  main_bulb.n  +  internal_bulb.n
    wire(load.cable_pos.n,     load.diode.anode)
    wire(load.diode.cathode,   load.main_bulb.p)
    wire(load.diode.cathode,   load.internal_bulb.p)
    wire(load.cable_neg.n,     load.main_bulb.n)
    wire(load.cable_neg.n,     load.internal_bulb.n)

    return kg


def build_ten_cubes_system(
    backend=None,
    extra_tools: set[str] | None = None,
) -> DiagnosableSystem:
    """
    Build and return a fresh ``DiagnosableSystem`` for the 10-cubes lamp.

    Each call creates fully independent component instances, so multiple
    systems (e.g. for different fault scenarios) can coexist safely.

    Parameters
    ----------
    backend
        A ``SimulationBackend``.  Defaults to ``PySpiceBackend()``.
        Defaults to ``PySpiceBackend()``.
    extra_tools
        Tool identifiers to pre-load into ``WorldContext.tools_in_hand``.
    """
    if backend is None:
        backend = PySpiceBackend()

    kg = _build_kg()
    return DiagnosableSystem(
        name="ten_cubes",
        kg=kg,
        context=WorldContext(tools_in_hand=set(extra_tools or [])),
        runner=SimulationRunner(backend=backend, couplings=[]),
    )
