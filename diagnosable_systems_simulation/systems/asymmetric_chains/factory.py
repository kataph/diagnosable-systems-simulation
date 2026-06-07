"""
Factory that assembles the asymmetric chains lamp system.

Usage::

    from diagnosable_systems_simulation.systems.asymmetric_chains.factory import (
        build_asymmetric_chains_system,
    )

    system = build_asymmetric_chains_system()
    result = system.simulate()

Topology overview
-----------------
Two interconnected chains operate two independent loads.

Chain 1:  PSU1 ─diode─► CTRL1 ──► LOAD1
                 ↑
Chain 2:  PSU2 ─diode─► CTRL1   (PSU2 also feeds CTRL1 via diode_psu2_ctrl1)
          PSU2 ──────► CTRL2 ─diode─► CTRL1  (via diode_ctrl2_ctrl1, back into CTRL1)
          CTRL1 ─diode─► CTRL3 ──► LOAD2
          CTRL2 ──────► CTRL3

Four diodes on positive lines prevent reverse current:
  diode_psu1_ctrl1  : PSU1 output → CTRL1 input
  diode_psu2_ctrl1  : PSU2 output → CTRL1 input  (cross-link)
  diode_ctrl2_ctrl1 : CTRL2 output → CTRL1 input  (back-feed protection)
  diode_ctrl1_ctrl3 : CTRL1 output → CTRL3 input  (cross-link)

CTRL2 → CTRL3 positive line is a direct connection (no diode per spec).

Negative (ground) lines are merged without diodes:
  PSU1.neg and PSU2.neg share a common ground.
  All module negative lines connect to ground.

Nominal state (all switches closed, correct polarity):
  Both PSU green LEDs lit.
  CTRL1, CTRL2, CTRL3 green LEDs lit.
  main_bulb of LOAD1 and LOAD2 both lit.
"""
from __future__ import annotations

from diagnosable_systems_simulation.electrical_simulation.backend.spice import PySpiceBackend
from diagnosable_systems_simulation.electrical_simulation.solver import SimulationRunner
from diagnosable_systems_simulation.systems.asymmetric_chains.components import create_components
from diagnosable_systems_simulation.systems.base_system import DiagnosableSystem
from diagnosable_systems_simulation.world.context import WorldContext
from diagnosable_systems_simulation.world.knowledge_graph import (
    EntityType, RelationType, SystemGraph,
)


def _build_kg() -> SystemGraph:
    kg = SystemGraph()
    c = create_components()

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

    # ── PART_OF ───────────────────────────────────────────────────────
    part_of(
        c.psu1_battery, c.psu1_battery_internal_resistor,
        c.psu1_green_led, c.psu1_green_resistor,
        c.psu1_cable_pos, c.psu1_cable_neg,
        c.diode_psu1_ctrl1,
        module=c.module_psu1,
    )
    part_of(
        c.psu2_battery, c.psu2_battery_internal_resistor,
        c.psu2_green_led, c.psu2_green_resistor,
        c.psu2_cable_pos, c.psu2_cable_neg,
        c.diode_psu2_ctrl1,
        module=c.module_psu2,
    )
    part_of(
        c.ctrl1_switch, c.ctrl1_green_led, c.ctrl1_green_resistor,
        c.ctrl1_cable_in_pos, c.ctrl1_cable_in_neg,
        c.ctrl1_cable_out_pos, c.ctrl1_cable_out_neg,
        module=c.module_ctrl1,
    )
    part_of(
        c.ctrl2_switch, c.ctrl2_green_led, c.ctrl2_green_resistor,
        c.ctrl2_cable_in_pos, c.ctrl2_cable_in_neg,
        c.ctrl2_cable_out_pos, c.ctrl2_cable_out_neg,
        module=c.module_ctrl2,
    )
    part_of(
        c.ctrl3_switch, c.ctrl3_green_led, c.ctrl3_green_resistor,
        c.ctrl3_cable_in_pos, c.ctrl3_cable_in_neg,
        c.ctrl3_cable_out_pos, c.ctrl3_cable_out_neg,
        c.diode_ctrl1_ctrl3, c.diode_ctrl2_ctrl3,
        module=c.module_ctrl3,
    )
    part_of(
        c.load1_main_bulb, c.load1_internal_bulb, c.load1_diode,
        c.load1_peephole, c.load1_cable_pos, c.load1_cable_neg,
        module=c.module_load1,
    )
    part_of(
        c.load2_main_bulb, c.load2_internal_bulb, c.load2_diode,
        c.load2_peephole, c.load2_cable_pos, c.load2_cable_neg,
        module=c.module_load2,
    )

    # ── CONTAINED_IN ──────────────────────────────────────────────────
    contained_in(
        c.psu1_battery, c.psu1_battery_internal_resistor,
        c.psu1_green_led, c.psu1_green_resistor, c.diode_psu1_ctrl1,
        enclosure=c.cube_psu1,
    )
    contained_in(
        c.psu2_battery, c.psu2_battery_internal_resistor,
        c.psu2_green_led, c.psu2_green_resistor, c.diode_psu2_ctrl1,
        enclosure=c.cube_psu2,
    )
    contained_in(
        c.ctrl1_switch, c.ctrl1_green_led, c.ctrl1_green_resistor,
        enclosure=c.cube_ctrl1,
    )
    contained_in(
        c.ctrl2_switch, c.ctrl2_green_led, c.ctrl2_green_resistor,
        enclosure=c.cube_ctrl2,
    )
    contained_in(
        c.ctrl3_switch, c.ctrl3_green_led, c.ctrl3_green_resistor,
        c.diode_ctrl1_ctrl3, c.diode_ctrl2_ctrl3,
        enclosure=c.cube_ctrl3,
    )
    contained_in(
        c.load1_main_bulb, c.load1_internal_bulb, c.load1_diode, c.load1_peephole,
        enclosure=c.cube_load1,
    )
    contained_in(
        c.load2_main_bulb, c.load2_internal_bulb, c.load2_diode, c.load2_peephole,
        enclosure=c.cube_load2,
    )

    # ── ELECTRICALLY_CONNECTED ────────────────────────────────────────
    #
    # Ground net: PSU1.neg and PSU2.neg share a common reference.
    # All module negative return lines merge to this ground.
    #
    # Positive path (diode-guarded junctions):
    #
    #  PSU1.pos ─► diode_psu1_ctrl1.anode
    #              diode_psu1_ctrl1.cathode ─┐
    #  PSU2.pos ─► diode_psu2_ctrl1.anode    ├─► ctrl1_in_p net → CTRL1 → CTRL1.out → LOAD1
    #              diode_psu2_ctrl1.cathode ─┘
    #
    #  PSU2.pos ─► ctrl2_cable_in_pos.p → CTRL2 → ctrl2_cable_out_pos.n
    #              ctrl2_cable_out_pos.n ─► diode_ctrl2_ctrl3.anode
    #                                       diode_ctrl2_ctrl3.cathode ─► ctrl3_in_p net
    #
    #  ctrl1_cable_out_pos.n ─► diode_ctrl1_ctrl3.anode
    #                            diode_ctrl1_ctrl3.cathode ─► ctrl3_in_p net → CTRL3 → LOAD2

    # --- Ground net (is_ground=True on first edge) ---
    wire(c.psu1_battery.neg,  c.psu1_green_led.cathode,             is_ground=True)
    wire(c.psu1_battery.neg,  c.psu1_cable_neg.p)
    wire(c.psu2_battery.neg,  c.psu2_green_led.cathode)
    wire(c.psu2_battery.neg,  c.psu2_cable_neg.p)
    # PSU negative cables merge to a shared GND rail at module junctions
    wire(c.psu1_cable_neg.n,  c.ctrl1_cable_in_neg.p)
    wire(c.psu2_cable_neg.n,  c.ctrl2_cable_in_neg.p)
    # ctrl1 and ctrl2 negative return paths merge at ctrl1 ground junction
    wire(c.ctrl1_cable_in_neg.n, c.ctrl1_cable_out_neg.p)
    wire(c.ctrl1_cable_in_neg.n, c.ctrl1_green_led.cathode)
    # ctrl2 negative feeds ctrl3 (and also feeds back to ctrl1 via diode_ctrl2_ctrl1)
    wire(c.ctrl2_cable_in_neg.n, c.ctrl2_cable_out_neg.p)
    wire(c.ctrl2_cable_in_neg.n, c.ctrl2_green_led.cathode)
    # ctrl3 negative
    wire(c.ctrl3_cable_in_neg.n, c.ctrl3_cable_out_neg.p)
    wire(c.ctrl3_cable_in_neg.n, c.ctrl3_green_led.cathode)
    # CTRL2 output negative → CTRL3 input negative (CTRL2 feeds CTRL3 directly)
    wire(c.ctrl2_cable_out_neg.n, c.ctrl3_cable_in_neg.p)
    # Merge all ctrl negative input nets to ground (spanning tree for ground net)
    wire(c.ctrl2_cable_in_neg.n, c.ctrl1_cable_in_neg.n)
    wire(c.ctrl3_cable_in_neg.n, c.ctrl1_cable_in_neg.n)
    # Load negative rails
    wire(c.ctrl1_cable_out_neg.n, c.load1_cable_neg.p)
    wire(c.ctrl3_cable_out_neg.n, c.load2_cable_neg.p)
    wire(c.load1_cable_neg.n, c.load1_main_bulb.n)
    wire(c.load1_cable_neg.n, c.load1_internal_bulb.n)
    wire(c.load2_cable_neg.n, c.load2_main_bulb.n)
    wire(c.load2_cable_neg.n, c.load2_internal_bulb.n)

    # --- PSU1 positive ---
    wire(c.psu1_battery.pos,  c.psu1_battery_internal_resistor.n)
    wire(c.psu1_battery_internal_resistor.p, c.psu1_green_resistor.p)
    wire(c.psu1_battery_internal_resistor.p, c.psu1_cable_pos.p)
    wire(c.psu1_green_resistor.n, c.psu1_green_led.anode)
    # PSU1 output → diode_psu1_ctrl1
    wire(c.psu1_cable_pos.n,  c.diode_psu1_ctrl1.anode)

    # --- PSU2 positive ---
    wire(c.psu2_battery.pos,  c.psu2_battery_internal_resistor.n)
    wire(c.psu2_battery_internal_resistor.p, c.psu2_green_resistor.p)
    wire(c.psu2_battery_internal_resistor.p, c.psu2_cable_pos.p)
    wire(c.psu2_green_resistor.n, c.psu2_green_led.anode)
    # PSU2 output splits: → diode_psu2_ctrl1 AND → ctrl2 directly
    wire(c.psu2_cable_pos.n,  c.diode_psu2_ctrl1.anode)
    wire(c.psu2_cable_pos.n,  c.ctrl2_cable_in_pos.p)

    # --- ctrl1_in_p net: diode_psu1_ctrl1.cathode + diode_psu2_ctrl1.cathode ---
    wire(c.diode_psu1_ctrl1.cathode,  c.ctrl1_cable_in_pos.p)
    wire(c.diode_psu2_ctrl1.cathode,  c.ctrl1_cable_in_pos.p)

    # --- CTRL1 internal (10-cubes style: green LED anode→resistor→switch.n) ---
    wire(c.ctrl1_cable_in_pos.n,  c.ctrl1_switch.p)
    wire(c.ctrl1_cable_in_pos.n,  c.ctrl1_green_resistor.p)
    wire(c.ctrl1_green_resistor.n, c.ctrl1_green_led.anode)
    wire(c.ctrl1_switch.n,         c.ctrl1_cable_out_pos.p)

    # --- CTRL1 output → diode_ctrl1_ctrl3 AND → LOAD1 ---
    wire(c.ctrl1_cable_out_pos.n,  c.diode_ctrl1_ctrl3.anode)
    wire(c.ctrl1_cable_out_pos.n,  c.load1_cable_pos.p)

    # --- CTRL2 internal ---
    wire(c.ctrl2_cable_in_pos.n,  c.ctrl2_switch.p)
    wire(c.ctrl2_cable_in_pos.n,  c.ctrl2_green_resistor.p)
    wire(c.ctrl2_green_resistor.n, c.ctrl2_green_led.anode)
    wire(c.ctrl2_switch.n,         c.ctrl2_cable_out_pos.p)

    # --- CTRL2 output → diode_ctrl2_ctrl3 ---
    wire(c.ctrl2_cable_out_pos.n,  c.diode_ctrl2_ctrl3.anode)

    # --- ctrl3_in_p net: diode_ctrl1_ctrl3.cathode + diode_ctrl2_ctrl3.cathode ---
    wire(c.diode_ctrl1_ctrl3.cathode, c.ctrl3_cable_in_pos.p)
    wire(c.diode_ctrl2_ctrl3.cathode, c.ctrl3_cable_in_pos.p)

    # --- CTRL3 internal ---
    wire(c.ctrl3_cable_in_pos.n,  c.ctrl3_switch.p)
    wire(c.ctrl3_cable_in_pos.n,  c.ctrl3_green_resistor.p)
    wire(c.ctrl3_green_resistor.n, c.ctrl3_green_led.anode)
    wire(c.ctrl3_switch.n,         c.ctrl3_cable_out_pos.p)

    # --- CTRL3 output → LOAD2 ---
    wire(c.ctrl3_cable_out_pos.n,  c.load2_cable_pos.p)

    # --- LOAD1 ---
    wire(c.load1_cable_pos.n,  c.load1_diode.anode)
    wire(c.load1_diode.cathode, c.load1_main_bulb.p)
    wire(c.load1_diode.cathode, c.load1_internal_bulb.p)

    # --- LOAD2 ---
    wire(c.load2_cable_pos.n,  c.load2_diode.anode)
    wire(c.load2_diode.cathode, c.load2_main_bulb.p)
    wire(c.load2_diode.cathode, c.load2_internal_bulb.p)

    return kg


def build_asymmetric_chains_system(
    backend=None,
    extra_tools: "set[str] | None" = None,
) -> DiagnosableSystem:
    """
    Build and return a fresh ``DiagnosableSystem`` for the asymmetric chains
    system.

    Parameters
    ----------
    backend
        A ``SimulationBackend``.  Defaults to ``PySpiceBackend()``.
    extra_tools
        Tool identifiers to pre-load into ``WorldContext.tools_in_hand``.
    """
    if backend is None:
        backend = PySpiceBackend()

    kg = _build_kg()

    return DiagnosableSystem(
        name="asymmetric_chains",
        kg=kg,
        context=WorldContext(tools_in_hand=set(extra_tools or [])),
        runner=SimulationRunner(backend=backend, couplings=[]),
    )
