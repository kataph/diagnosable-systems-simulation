"""
Factory that assembles the current sensor lamp system.

Usage::

    from diagnosable_systems_simulation.systems.current_sensor.factory import (
        build_current_sensor_system,
    )

    system = build_current_sensor_system()
    result = system.simulate()

System overview
---------------
Three modules in series: Power Supply → Current Sensor Control → Load.

The control module contains a relay on the **0V/negative line**.
A ``CurrentSensorRelayCoupling`` reads the current through the relay after
each simulation step.  If the current exceeds ``CURRENT_THRESHOLD`` the
relay opens, disconnecting the return path and stopping all current flow.

In nominal operation the lamp draws ~0.1 A (12 V / 120 Ω), well below the
default threshold of 0.5 A, so the relay stays closed and the lamp is lit.

To model an overcurrent fault, degrade the main bulb resistance::

    system.inject_fault(
        DegradeComponent({"resistance": 5.0}),
        {"subject": system.component("main_bulb")},
    )

With R=5 Ω the current rises to ~2.4 A (> 0.5 A threshold) → relay opens →
lamp goes off.
"""
from __future__ import annotations

from diagnosable_systems_simulation.electrical_simulation.backend.spice import PySpiceBackend
from diagnosable_systems_simulation.electrical_simulation.circuit import CircuitGraph
from diagnosable_systems_simulation.electrical_simulation.results import SimulationResult
from diagnosable_systems_simulation.electrical_simulation.solver import (
    PhysicalCoupling, SimulationRunner,
)
from diagnosable_systems_simulation.systems.base_system import DiagnosableSystem
from diagnosable_systems_simulation.systems.current_sensor.components import create_components
from diagnosable_systems_simulation.world.components import Switch
from diagnosable_systems_simulation.world.context import WorldContext
from diagnosable_systems_simulation.world.knowledge_graph import (
    EntityType, RelationType, SystemGraph,
)

# Relay opens when current exceeds this threshold (Amps).
# Nominal lamp current ≈ 0.1 A; fault threshold well above nominal.
CURRENT_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Current sensor coupling
# ---------------------------------------------------------------------------

class CurrentSensorRelayCoupling(PhysicalCoupling):
    """
    Opens or closes the relay based on circuit current after each simulation.

    The current is read from ``result.branch_currents[sensor_component_id]``,
    which is the current entering the positive port of that component
    (the relay itself, placed on the 0V line).

    If ``|current| > current_threshold``: relay opens (overcurrent protection).
    If ``|current| <= current_threshold``: relay closes (normal operation).

    The relay's ``_fault_overlay`` is respected: if ``"is_closed"`` is forced
    by a fault injection the coupling does not fight it.
    """

    def __init__(
        self,
        sensor_component_id: str,
        relay_id: str,
        current_threshold: float = CURRENT_THRESHOLD,
    ) -> None:
        self.sensor_component_id = sensor_component_id
        self.relay_id = relay_id
        self.current_threshold = current_threshold

    def apply(
        self,
        result: SimulationResult,
        graph: CircuitGraph,
        context: WorldContext,
    ) -> bool:
        if not graph.has_component(self.relay_id):
            return False

        relay: Switch = graph.get_component(self.relay_id)  # type: ignore[assignment]
        if not isinstance(relay, Switch):
            return False

        if "is_closed" in relay._fault_overlay:
            return False

        current = abs(result.branch_currents.get(self.sensor_component_id, 0.0))
        should_close = current <= self.current_threshold

        if relay.is_closed != should_close:
            relay.is_closed = should_close
            return True
        return False


# ---------------------------------------------------------------------------
# Knowledge graph builder
# ---------------------------------------------------------------------------

def _build_kg() -> SystemGraph:
    kg = SystemGraph()
    c = create_components()

    (
        module_psu, module_ctrl, module_load,
        cube_psu, cube_ctrl, cube_load,
        power_source, battery_internal_resistor,
        psu_green_led, psu_green_resistor, psu_cable_pos, psu_cable_neg,
        ctrl_panel, ctrl_relay, ctrl_green_led, ctrl_green_resistor,
        ctrl_cable_in_pos, ctrl_cable_in_neg, ctrl_cable_out_pos, ctrl_cable_out_neg,
        main_bulb, internal_bulb, load_diode, load_cable_pos, load_cable_neg,
        load_panel,
    ) = (
        c.module_psu, c.module_ctrl, c.module_load,
        c.cube_psu, c.cube_ctrl, c.cube_load,
        c.battery, c.battery_internal_resistor,
        c.psu_green_led, c.psu_green_resistor, c.psu_cable_pos, c.psu_cable_neg,
        c.ctrl_panel, c.ctrl_relay, c.ctrl_green_led, c.ctrl_green_resistor,
        c.ctrl_cable_in_pos, c.ctrl_cable_in_neg, c.ctrl_cable_out_pos, c.ctrl_cable_out_neg,
        c.main_bulb, c.internal_bulb, c.load_diode, c.load_cable_pos, c.load_cable_neg,
        c.load_panel,
    )

    for cid, comp in c.ALL.items():
        kg.add_entity(cid, EntityType.COMPONENT, comp)

    def part_of(*comps, module):
        for comp in comps:
            kg.add_edge(comp.component_id, module.component_id, RelationType.PART_OF)

    def contained_in(*comps, enclosure):
        for comp in comps:
            kg.add_edge(comp.component_id, enclosure.component_id, RelationType.CONTAINED_IN)

    # ── PART_OF ───────────────────────────────────────────────────────
    part_of(
        power_source, psu_green_led, psu_green_resistor,
        psu_cable_pos, psu_cable_neg, battery_internal_resistor,
        module=module_psu,
    )
    part_of(
        ctrl_panel, ctrl_relay, ctrl_green_led, ctrl_green_resistor,
        ctrl_cable_in_pos, ctrl_cable_in_neg,
        ctrl_cable_out_pos, ctrl_cable_out_neg,
        module=module_ctrl,
    )
    part_of(
        main_bulb, internal_bulb, load_diode, load_panel,
        load_cable_pos, load_cable_neg,
        module=module_load,
    )

    # ── CONTAINED_IN ──────────────────────────────────────────────────
    contained_in(
        power_source, psu_green_led, psu_green_resistor, battery_internal_resistor,
        enclosure=cube_psu,
    )
    contained_in(
        ctrl_panel, ctrl_relay, ctrl_green_led, ctrl_green_resistor,
        enclosure=cube_ctrl,
    )
    contained_in(
        main_bulb, internal_bulb, load_diode, load_panel,
        enclosure=cube_load,
    )

    # ── ELECTRICALLY_CONNECTED ────────────────────────────────────────
    EC = RelationType.ELECTRICALLY_CONNECTED

    def wire(port_a, port_b, **kw):
        kg.add_edge(
            port_a.component.component_id, port_b.component.component_id, EC,
            from_port=port_a.port_name, to_port=port_b.port_name, **kw,
        )

    # Ground net (relay.n side is the true return path to battery neg)
    wire(power_source.neg,              psu_green_led.cathode,     is_ground=True)
    wire(power_source.neg,              psu_cable_neg.p)

    # psu_pos net
    wire(power_source.pos,              battery_internal_resistor.n)
    wire(battery_internal_resistor.p,   psu_green_resistor.p)
    wire(battery_internal_resistor.p,   psu_cable_pos.p)

    # psu_green_mid net
    wire(psu_green_resistor.n,          psu_green_led.anode)

    # PSU → Control junction
    wire(psu_cable_pos.n,               ctrl_cable_in_pos.p)
    wire(psu_cable_neg.n,               ctrl_cable_in_neg.p)

    # ctrl_in_p net: 12V passes straight through to output positive
    wire(ctrl_cable_in_pos.n,           ctrl_cable_out_pos.p)
    wire(ctrl_cable_in_pos.n,           ctrl_green_resistor.p)

    # green_mid net
    wire(ctrl_green_resistor.n,         ctrl_green_led.anode)

    # Relay on 0V/negative line:
    #   cable_in_neg.n → relay.p → relay.n → cable_out_neg.p
    wire(ctrl_cable_in_neg.n,           ctrl_relay.p)
    wire(ctrl_relay.n,                  ctrl_cable_out_neg.p)
    wire(ctrl_relay.n,                  ctrl_green_led.cathode)

    # Control → Load junction
    wire(ctrl_cable_out_pos.n,          load_cable_pos.p)
    wire(ctrl_cable_out_neg.n,          load_cable_neg.p)

    # load_in_p net
    wire(load_cable_pos.n,              load_diode.anode)

    # load_in_n net
    wire(load_cable_neg.n,              main_bulb.n)
    wire(load_cable_neg.n,              internal_bulb.n)

    # load_post net
    wire(load_diode.cathode,            main_bulb.p)
    wire(load_diode.cathode,            internal_bulb.p)

    return kg


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_current_sensor_system(
    backend=None,
    extra_tools: "set[str] | None" = None,
    current_threshold: float = CURRENT_THRESHOLD,
) -> DiagnosableSystem:
    """
    Build and return a fresh ``DiagnosableSystem`` for the current sensor system.

    Parameters
    ----------
    backend
        A ``SimulationBackend``.  Defaults to ``PySpiceBackend()``.
    extra_tools
        Tool identifiers to pre-load into ``WorldContext.tools_in_hand``.
    current_threshold
        Current (Amps) above which the relay opens.  Default 0.5 A.
    """
    if backend is None:
        backend = PySpiceBackend()

    kg = _build_kg()

    coupling = CurrentSensorRelayCoupling(
        sensor_component_id="ctrl_relay",
        relay_id="ctrl_relay",
        current_threshold=current_threshold,
    )

    return DiagnosableSystem(
        name="current_sensor",
        kg=kg,
        context=WorldContext(tools_in_hand=set(extra_tools or [])),
        runner=SimulationRunner(backend=backend, couplings=[coupling]),
    )
