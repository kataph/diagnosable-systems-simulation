"""
Factory that assembles the complete cube lamp system.

Usage::

    from diagnosable_systems_simulation.systems.three_cubes.factory import build_three_cubes_system

    system = build_three_cubes_system()
    result = system.simulate()
"""
from __future__ import annotations

from diagnosable_systems_simulation.electrical_simulation.backend.spice import PySpiceBackend
from diagnosable_systems_simulation.electrical_simulation.solver import SimulationRunner
from diagnosable_systems_simulation.systems.base_system import DiagnosableSystem
from diagnosable_systems_simulation.systems.three_cubes.components import create_components
from diagnosable_systems_simulation.world.context import WorldContext
from diagnosable_systems_simulation.world.knowledge_graph import (
    EntityType, RelationType, SystemGraph,
)


def _build_kg() -> SystemGraph:
    kg = SystemGraph()

    # Fresh component instances — independent of any previous system
    c = create_components()

    # Unpack for readable wiring below
    (cube_psu, cube_ctrl, cube_load,
     power_source, psu_green_led, psu_green_resistor, psu_cable_pos, psu_cable_neg,
     ctrl_switch, ctrl_red_led, ctrl_red_resistor,
     ctrl_cable_in_pos, ctrl_cable_in_neg, ctrl_cable_out_pos, ctrl_cable_out_neg,
     main_bulb, internal_bulb, load_diode, load_cable_pos, load_cable_neg,
     load_peephole) = (
        c.cube_psu, c.cube_ctrl, c.cube_load,
        c.battery, c.psu_green_led, c.psu_green_resistor, c.psu_cable_pos, c.psu_cable_neg,
        c.ctrl_switch, c.ctrl_red_led, c.ctrl_red_resistor,
        c.ctrl_cable_in_pos, c.ctrl_cable_in_neg, c.ctrl_cable_out_pos, c.ctrl_cable_out_neg,
        c.main_bulb, c.internal_bulb, c.load_diode, c.load_cable_pos, c.load_cable_neg,
        c.load_peephole,
    )

    # ── Component entities ─────────────────────────────────────────────
    for cid, comp in c.ALL.items():
        kg.add_entity(cid, EntityType.COMPONENT, comp)

    def part_of(*comps, module):
        for comp in comps:
            kg.add_edge(comp.component_id, module.component_id, RelationType.PART_OF)

    def contained_in(*comps, enclosure):
        for comp in comps:
            kg.add_edge(comp.component_id, enclosure.component_id, RelationType.CONTAINED_IN)

    # ── PART_OF  (enclosures act as module anchors) ────────────────────
    part_of(power_source, psu_green_led, psu_green_resistor,
            psu_cable_pos, psu_cable_neg,                       module=cube_psu)

    part_of(ctrl_switch, ctrl_red_led, ctrl_red_resistor,
            ctrl_cable_in_pos, ctrl_cable_in_neg,
            ctrl_cable_out_pos, ctrl_cable_out_neg,             module=cube_ctrl)

    part_of(main_bulb, internal_bulb, load_diode, load_peephole,
            load_cable_pos, load_cable_neg,                     module=cube_load)

    # ── CONTAINED_IN ──────────────────────────────────────────────────
    contained_in(power_source, psu_green_led, psu_green_resistor,    enclosure=cube_psu)
    contained_in(ctrl_red_led, ctrl_red_resistor,                     enclosure=cube_ctrl)
    contained_in(main_bulb, internal_bulb, load_diode, load_peephole, enclosure=cube_load)

    # ── ELECTRICALLY_CONNECTED: port-to-port wiring ────────────────────
    #
    # Each edge says: port X of component A is wired to port Y of component B.
    # Nodes are NOT stored here; build_circuit_from_kg derives them via
    # union-find over these edges.
    #
    # For nets with 3+ ports, N-1 edges suffice (spanning tree per net).
    # One edge per net carries is_ground=True to mark the 0 V reference.
    #
    # Nominal topology (switch closed, correct polarity):
    #   Green LED: lit (in parallel with 12V source)
    #   Red LED:   OFF (reverse-biased: cathode toward ctrl_in_p = 12V)
    #   Main bulb + internal bulb: lit (in parallel after protection diode)

    EC = RelationType.ELECTRICALLY_CONNECTED

    def wire(port_a, port_b, **kw):
        kg.add_edge(port_a.component.component_id, port_b.component.component_id, EC,
                    from_port=port_a.port_name, to_port=port_b.port_name, **kw)

    # Ground net
    wire(power_source.neg,       psu_green_led.cathode,     is_ground=True)
    wire(power_source.neg,       psu_cable_neg.p)
    # psu_pos net
    wire(power_source.pos,       psu_green_resistor.p)
    wire(power_source.pos,       psu_cable_pos.p)
    # psu_green_mid
    wire(psu_green_resistor.n,   psu_green_led.anode)
    # PSU → Control junction
    wire(psu_cable_pos.n,        ctrl_cable_in_pos.p)
    wire(psu_cable_neg.n,        ctrl_cable_in_neg.p)
    # ctrl_in_p net
    wire(ctrl_cable_in_pos.n,    ctrl_switch.p)
    wire(ctrl_cable_in_pos.n,    ctrl_red_resistor.p)
    # ctrl_mid
    wire(ctrl_switch.n,          ctrl_cable_out_pos.p)
    # ctrl_red_mid: cathode → indicator resistor (no cable on this net)
    wire(ctrl_red_resistor.n,    ctrl_red_led.cathode)
    # ctrl_in_n net (ground): anode, ctrl_cable_in_neg.n, ctrl_cable_out_neg.p all land here.
    # NOMINAL inspect_connections result: anode shows TWO cables (both negative cables),
    # cathode shows NO cable.  This is by design — not a fault.
    wire(ctrl_cable_in_neg.n,    ctrl_red_led.anode)
    wire(ctrl_cable_in_neg.n,    ctrl_cable_out_neg.p)
    # Control → Load junction
    wire(ctrl_cable_out_pos.n,   load_cable_pos.p)
    wire(ctrl_cable_out_neg.n,   load_cable_neg.p)
    # load_in_p
    wire(load_cable_pos.n,       load_diode.anode)
    # load_in_n net
    wire(load_cable_neg.n,       main_bulb.n)
    wire(load_cable_neg.n,       internal_bulb.n)
    # load_post net
    wire(load_diode.cathode,     main_bulb.p)
    wire(load_diode.cathode,     internal_bulb.p)

    return kg


def build_three_cubes_system(
    backend=None,
    extra_tools: set[str] | None = None,
) -> DiagnosableSystem:
    """
    Build and return a fresh ``DiagnosableSystem`` for the cube lamp.

    Each call creates fully independent component instances, so multiple
    systems (e.g. for different fault scenarios) can coexist safely.

    Parameters
    ----------
    backend
        A ``SimulationBackend``. Defaults to ``PySpiceBackend()``.
        Defaults to ``PySpiceBackend()``.
    extra_tools
        Tool identifiers to pre-load into ``WorldContext.tools_in_hand``.
    """
    if backend is None:
        backend = PySpiceBackend()

    kg = _build_kg()
    return DiagnosableSystem(
        name="three_cubes",
        kg=kg,
        context=WorldContext(tools_in_hand=set(extra_tools or [])),
        runner=SimulationRunner(backend=backend, couplings=[]),
    )
