"""
Factory that assembles the ambient light sensor lamp system.

Usage::

    from diagnosable_systems_simulation.systems.ambient_light_sensor.factory import (
        build_ambient_light_system,
    )

    system = build_ambient_light_system()
    result = system.simulate()

Scenario 15 — feedback loop
----------------------------
In nominal operation the three modules are separated and the sensor sees no
lamp light: relay closed → lamp ON stably.

When the modules are stacked on top of each other (fault: ``als_feedback``
flag set in ``context.extra``), the lamp (in the load cube at z≈0.35 m) sits
directly above the light sensor (in the ctrl cube at z≈0.20 m).  The
``AmbientFeedbackCoupling`` then drives:

    lamp ON  → sensor lit  → relay opens → lamp OFF
    lamp OFF → sensor dark → relay closes → lamp ON → …

The coupling loop never stabilises (``converged=False``), modelling the
observed flickering/intermittent behaviour.

Fix
---
Rotating or moving either the load cube or the control cube so that the lamp
no longer faces the sensor breaks the optical path.  In the simulation this
corresponds to calling ``InvertEnclosure`` on ``cube_load`` or ``cube_ctrl``,
which sets ``is_inverted=True`` on the enclosure object.  The coupling checks
this flag and, when either shielding enclosure is inverted, skips the
lamp→sensor illumination.  The relay stays closed, the lamp stays on.

Potentiometer bait
------------------
The sensitivity and timing potentiometers are wired across the supply rail
(p→ctrl_in_p, n→ctrl_in_n) with the wiper left unconnected.  Adjusting them
changes the wiper_position parameter but never affects whether the relay
opens or closes — that is entirely determined by the coupling above.
"""
from __future__ import annotations

from diagnosable_systems_simulation.electrical_simulation.backend.spice import PySpiceBackend
from diagnosable_systems_simulation.electrical_simulation.results import SimulationResult
from diagnosable_systems_simulation.electrical_simulation.circuit import CircuitGraph
from diagnosable_systems_simulation.electrical_simulation.solver import (
    PhysicalCoupling, SimulationRunner,
)
from diagnosable_systems_simulation.systems.base_system import DiagnosableSystem
from diagnosable_systems_simulation.systems.ambient_light_sensor.components import create_components
from diagnosable_systems_simulation.world.components import LightSensor, PhysicalEnclosure, Switch
from diagnosable_systems_simulation.world.context import WorldContext
from diagnosable_systems_simulation.world.knowledge_graph import (
    EntityType, RelationType, SystemGraph,
)
from diagnosable_systems_simulation.world.spatial import Position


# ---------------------------------------------------------------------------
# Feedback coupling
# ---------------------------------------------------------------------------

class AmbientFeedbackCoupling(PhysicalCoupling):
    """
    Models the optical feedback between the lamp and the ambient light sensor
    when the three modules are stacked on top of each other.

    Two effects are applied in one ``apply()`` call:

    1. **Lamp → sensor illumination**: if ``als_feedback`` is active in the
       context, neither shielding enclosure is inverted/rotated, the lamp is
       lit, and the lamp is within coupling range of the sensor, then the
       sensor is set to its lit (low-resistance) state.  Otherwise the sensor
       stays dark.

    2. **Sensor → relay**: a dark sensor (high resistance) keeps the relay
       closed (lamp ON path); a lit sensor (low resistance) opens the relay
       (lamp OFF path).  The relay state is set deterministically from the
       sensor state — no hysteresis.

    The coupling loop in ``SimulationRunner`` iterates until stable.  Because
    the lamp ON → sensor lit → relay open → lamp OFF → sensor dark → relay
    closed → lamp ON path never stabilises, the runner hits ``MAX_ITERATIONS``
    and returns ``converged=False``, modelling the observed flickering.

    Parameters
    ----------
    lamp_id
        Component ID of the light-emitting bulb (``"main_bulb"``).
    sensor_id
        Component ID of the LightSensor (``"ctrl_light_sensor"``).
    relay_id
        Component ID of the Switch acting as relay (``"ctrl_relay"``).
    lamp_pos, sensor_pos
        Positions used for the distance check.
    coupling_radius
        Maximum distance (metres) at which the lamp illuminates the sensor.
    shielding_enclosures
        List of ``PhysicalEnclosure`` objects (``cube_load``, ``cube_ctrl``).
        If **any** of them has ``is_inverted=True``, the optical path is
        considered broken and the lamp cannot illuminate the sensor.
    """

    def __init__(
        self,
        lamp_id: str,
        sensor_id: str,
        relay_id: str,
        lamp_pos: Position,
        sensor_pos: Position,
        coupling_radius: float,
        shielding_enclosures: list[PhysicalEnclosure],
    ) -> None:
        self.lamp_id = lamp_id
        self.sensor_id = sensor_id
        self.relay_id = relay_id
        self.lamp_pos = lamp_pos
        self.sensor_pos = sensor_pos
        self.coupling_radius = coupling_radius
        self.shielding_enclosures = shielding_enclosures

    def _feedback_blocked(self) -> bool:
        """Return True if any shielding enclosure has been rotated/moved."""
        return any(enc.is_inverted for enc in self.shielding_enclosures)

    def apply(
        self,
        result: SimulationResult,
        graph: CircuitGraph,
        context: WorldContext,
    ) -> bool:
        feedback_active = context.extra.get("als_feedback", False)

        if not graph.has_component(self.sensor_id) or not graph.has_component(self.relay_id):
            return False

        sensor: LightSensor = graph.get_component(self.sensor_id)  # type: ignore[assignment]
        relay: Switch = graph.get_component(self.relay_id)          # type: ignore[assignment]

        if not isinstance(sensor, LightSensor) or not isinstance(relay, Switch):
            return False

        changed = False

        # 1. Lamp → sensor: does the lamp illuminate the sensor?
        lamp_illuminates = (
            feedback_active
            and not self._feedback_blocked()
            and result.is_lit(self.lamp_id)
            and self.lamp_pos.is_within(self.sensor_pos, self.coupling_radius)
        )
        old_r = sensor._current_resistance
        sensor.set_illuminated(lamp_illuminates)
        if sensor._current_resistance != old_r:
            changed = True

        # 2. Sensor → relay: dark sensor → relay closed; lit sensor → relay open.
        # Skip if the relay has a fault overlay forcing its state — the overlay
        # already controls what SPICE sees, so the coupling must not fight it.
        if "is_closed" not in relay._fault_overlay:
            should_close = (sensor._current_resistance == sensor.resistance_dark)
            if relay.is_closed != should_close:
                relay.is_closed = should_close
                changed = True

        return changed


# ---------------------------------------------------------------------------
# Knowledge graph builder
# ---------------------------------------------------------------------------

def _build_kg() -> tuple[SystemGraph, PhysicalEnclosure, PhysicalEnclosure]:
    """
    Build the system knowledge graph and return it together with the two
    shielding enclosure objects (needed by ``AmbientFeedbackCoupling``).
    """
    kg = SystemGraph()
    c = create_components()

    (
        module_psu, module_ctrl, module_load,
        cube_psu, cube_ctrl, cube_load,
        power_source, battery_internal_resistor,
        psu_green_led, psu_green_resistor, psu_cable_pos, psu_cable_neg,
        ctrl_panel, ctrl_light_sensor, ctrl_relay,
        ctrl_sensitivity_pot, ctrl_timing_pot, ctrl_sensor_bias,
        ctrl_cable_in_pos, ctrl_cable_in_neg, ctrl_cable_out_pos, ctrl_cable_out_neg,
        main_bulb, internal_bulb, load_diode, load_cable_pos, load_cable_neg,
        load_panel,
    ) = (
        c.module_psu, c.module_ctrl, c.module_load,
        c.cube_psu, c.cube_ctrl, c.cube_load,
        c.battery, c.battery_internal_resistor,
        c.psu_green_led, c.psu_green_resistor, c.psu_cable_pos, c.psu_cable_neg,
        c.ctrl_panel, c.ctrl_light_sensor, c.ctrl_relay,
        c.ctrl_sensitivity_pot, c.ctrl_timing_pot, c.ctrl_sensor_bias,
        c.ctrl_cable_in_pos, c.ctrl_cable_in_neg, c.ctrl_cable_out_pos, c.ctrl_cable_out_neg,
        c.main_bulb, c.internal_bulb, c.load_diode, c.load_cable_pos, c.load_cable_neg,
        c.load_panel,
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

    # ── PART_OF ───────────────────────────────────────────────────────
    part_of(
        power_source, psu_green_led, psu_green_resistor,
        psu_cable_pos, psu_cable_neg, battery_internal_resistor,
        module=module_psu,
    )
    part_of(
        ctrl_panel, ctrl_light_sensor, ctrl_relay,
        ctrl_sensitivity_pot, ctrl_timing_pot, ctrl_sensor_bias,
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
        ctrl_panel, ctrl_light_sensor, ctrl_relay,
        ctrl_sensitivity_pot, ctrl_timing_pot, ctrl_sensor_bias,
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

    # Ground net
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

    # ctrl_in_p net (= ctrl_cable_in_pos.n)
    wire(ctrl_cable_in_pos.n,           ctrl_relay.p)
    wire(ctrl_cable_in_pos.n,           ctrl_sensor_bias.p)
    wire(ctrl_cable_in_pos.n,           ctrl_sensitivity_pot.p)
    wire(ctrl_cable_in_pos.n,           ctrl_timing_pot.p)

    # sensor_mid net
    wire(ctrl_sensor_bias.n,            ctrl_light_sensor.p)

    # relay_out net
    wire(ctrl_relay.n,                  ctrl_cable_out_pos.p)

    # ctrl_in_n / GND net (= ctrl_cable_in_neg.n)
    # Wiper ports of potentiometers are intentionally left unconnected;
    # the SPICE backend creates an internal floating node for each wiper,
    # which is harmless (the two resistor halves form a complete p→n path).
    wire(ctrl_cable_in_neg.n,           ctrl_cable_out_neg.p)
    wire(ctrl_cable_in_neg.n,           ctrl_light_sensor.n)
    wire(ctrl_cable_in_neg.n,           ctrl_sensitivity_pot.n)
    wire(ctrl_cable_in_neg.n,           ctrl_timing_pot.n)

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

    return kg, cube_ctrl, cube_load


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_ambient_light_system(
    backend=None,
    extra_tools: set[str] | None = None,
) -> DiagnosableSystem:
    """
    Build and return a fresh ``DiagnosableSystem`` for the ambient light sensor
    lamp system.

    The system starts with ``als_feedback=False`` in ``context.extra``, which
    means the feedback coupling is inactive and the lamp is stably ON.

    To activate the feedback loop (Scenario 15 fault), set::

        system.context.extra["als_feedback"] = True

    Parameters
    ----------
    backend
        A ``SimulationBackend``.  Defaults to ``PySpiceBackend()``.
    extra_tools
        Tool identifiers to pre-load into ``WorldContext.tools_in_hand``.
    """
    if backend is None:
        backend = PySpiceBackend()

    kg, cube_ctrl, cube_load = _build_kg()

    # Positions for the coupling distance check (z-axis stacking scenario).
    # lamp (main_bulb) is in the load cube at z≈0.35 m;
    # sensor is in the ctrl cube at z≈0.20 m.
    # Distance = 0.15 m < coupling_radius = 0.25 m → within range when stacked.
    lamp_pos   = Position(0.05, 0.05, 0.35)
    sensor_pos = Position(0.05, 0.05, 0.20)

    coupling = AmbientFeedbackCoupling(
        lamp_id="main_bulb",
        sensor_id="ctrl_light_sensor",
        relay_id="ctrl_relay",
        lamp_pos=lamp_pos,
        sensor_pos=sensor_pos,
        coupling_radius=0.25,
        shielding_enclosures=[cube_ctrl, cube_load],
    )

    return DiagnosableSystem(
        name="ambient_light_sensor",
        kg=kg,
        context=WorldContext(tools_in_hand=set(extra_tools or [])),
        runner=SimulationRunner(backend=backend, couplings=[coupling]),
    )
