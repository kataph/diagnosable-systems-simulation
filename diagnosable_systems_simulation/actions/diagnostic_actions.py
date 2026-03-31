from __future__ import annotations

from diagnosable_systems_simulation.actions.base import Action, ActionCost, ActionResult
from diagnosable_systems_simulation.actions.observation import ObservationRecord, observe_component
from diagnosable_systems_simulation.actions.preconditions import (
    AffordanceRequirement, ToolRequirement, PreconditionChecker
)
from diagnosable_systems_simulation.world.affordances import Affordance


def _nearby_anomalies(comp, graph) -> list[str]:
    """
    Return human-readable anomaly strings for components directly adjacent
    (sharing a circuit node) to *comp* that are in an abnormal state a
    technician at this location would physically observe.

    Currently detects:
      • Cables with a floating port whose ``_orig_connections`` pointed to a
        node of *comp* — i.e. a cable end that was plugged into this
        component's terminal but has since been disconnected.

    Designed to be extended with further anomaly checks as needed.
    """
    from diagnosable_systems_simulation.world.components import Cable

    comp_nodes = {p.node_id for p in comp.ports if p.is_connected()}
    anomalies: list[str] = []
    seen: set[str] = set()

    for edge in graph.get_netlist():
        neighbor = edge.component
        if neighbor.component_id == comp.component_id or neighbor.component_id in seen:
            continue
        seen.add(neighbor.component_id)

        # --- Check: disconnected cable end adjacent to comp ---
        # Triggers only if the cable's disconnected port was originally attached
        # to one of comp's own terminals (direct adjacency). One-hop adjacency
        # (cable's live end shares a node with comp) is intentionally excluded:
        # a cable whose disconnected end is inside a closed enclosure is not
        # physically visible to a technician working at comp's location.
        if isinstance(neighbor, Cable):
            orig = getattr(neighbor, "_orig_connections", {})
            for port in neighbor.ports:
                if port.is_connected():
                    continue
                if orig.get(port.name) in comp_nodes:
                    orig_node = orig.get(port.name)
                    connected_names = [
                        e.component.display_name
                        for e in graph.get_netlist()
                        if orig_node in e.port_nodes.values()
                        and e.component.component_id != neighbor.component_id
                    ]
                    if connected_names:
                        conn_str = ", ".join(f"'{n}'" for n in connected_names)
                        msg = (
                            f"cable '{neighbor.display_name}' has a disconnected end "
                            f"(port '{port.name}' is floating; "
                            f"originally electrically connected to {conn_str})"
                        )
                    else:
                        msg = (
                            f"cable '{neighbor.display_name}' has a disconnected end "
                            f"(port '{port.name}' is floating)"
                        )
                    anomalies.append(msg)

        # --- Future checks can be added here ---
        # e.g. blown fuse on adjacent node, degraded component visible at terminal, …

    return anomalies


class ObserveComponent(Action):
    """
    Visually inspect a component.

    Requires OBSERVABLE affordance.
    Returns an ObservationRecord with all visible properties.

    targets: {"subject": <any Component>}
    """

    action_id = "observe_component"
    description = "Visually inspect a component."
    cost = ActionCost(time=10.0)

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [AffordanceRequirement("subject", Affordance.OBSERVABLE)],
            targets, context,
        )
        return ok, "; ".join(failures)

    def execute(self, targets, graph, context, last_result):
        comp = targets["subject"]
        return ActionResult(
            observation=observe_component(comp, context, self.action_id, last_result),
            message=f"Observed {comp.display_name!r}.",
        )


class MeasureVoltage(Action):
    """
    Measure voltage at a component's ports (w.r.t. ground) using a multimeter.

    Requires REACHABLE and MEASURABLE affordance and "multimeter" in tools_in_hand.
    Returns an ObservationRecord with port voltages only.

    targets: {"subject": <any Component>}
    """

    action_id = "measure_voltage"
    description = "Measure voltage at component ports (w.r.t. ground) with a multimeter."
    cost = ActionCost(time=20.0, equipment=["multimeter"])

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [
                AffordanceRequirement("subject", Affordance.REACHABLE),
                AffordanceRequirement("subject", Affordance.MEASURABLE),
                ToolRequirement("multimeter"),
            ],
            targets, context,
        )
        return ok, "; ".join(failures)

    def execute(self, targets, graph, context, last_result):
        comp = targets["subject"]
        record = ObservationRecord(
            component_id=comp.component_id,
            action_id=self.action_id,
            simulation_snapshot=last_result,
        )
        # Report floating ports but still measure connected ones.
        floating = [p.name for p in comp.ports if not p.is_connected()]
        if floating:
            record.add("disconnected_ports", str(floating))

        if last_result is None:
            record.add("note", "No simulation result available.")
        else:
            for port in comp.ports:
                if port.node_id is not None:
                    v = last_result.voltage(port.node_id)
                    if v is not None:
                        record.add(f"voltage_{port.name}", round(v, 4), "V")
        anomalies = _nearby_anomalies(comp, graph)
        if anomalies:
            record.add("nearby_anomalies", "; ".join(anomalies))
        anomaly_suffix = (" NEARBY ANOMALY: " + "; ".join(anomalies)) if anomalies else ""
        return ActionResult(
            observation=record,
            message=f"Measured voltage at {comp.display_name!r}.{anomaly_suffix}",
        )


class MeasureCurrent(Action):
    """
    Measure the branch current through a component using a multimeter (ammeter mode).

    Requires REACHABLE and MEASURABLE affordance and "multimeter" in tools_in_hand.
    Returns an ObservationRecord with the branch current only.

    targets: {"subject": <any Component>}
    """

    action_id = "measure_current"
    description = "Measure the branch current through a component with a multimeter (ammeter mode)."
    cost = ActionCost(time=20.0, equipment=["multimeter"])

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [
                AffordanceRequirement("subject", Affordance.REACHABLE),
                AffordanceRequirement("subject", Affordance.MEASURABLE),
                ToolRequirement("multimeter"),
            ],
            targets, context,
        )
        return ok, "; ".join(failures)

    def execute(self, targets, graph, context, last_result):
        comp = targets["subject"]
        record = ObservationRecord(
            component_id=comp.component_id,
            action_id=self.action_id,
            simulation_snapshot=last_result,
        )
        if last_result is None:
            record.add("note", "No simulation result available.")
            return ActionResult(
                observation=record,
                message=f"Measured current through {comp.display_name!r}: no simulation result available.",
            )
        i = last_result.current(comp.component_id)
        if i is not None:
            record.add("current", round(i, 6), "A")
            return ActionResult(
                observation=record,
                message=f"Current through {comp.display_name!r}: {i:.6f} A.",
            )
        record.add("note", "Current not available for this component.")
        return ActionResult(
            observation=record,
            message=f"Current measurement not available for {comp.display_name!r}.",
        )


class OpenSwitch(Action):
    """
    Set a switch to the open (off) position.

    **Idempotent**: if the switch is already open, no mutation is made and
    the message reports the existing state.  The action never fails because
    the switch is already in the target position.

    Prefer this over a "toggle" action when calling through the NL
    interface: the agent has no access to the current simulation state, so
    a toggle would be ambiguous; a state-targeting action is always safe.

    Requires TOGGLABLE affordance (no REACHABLE needed — switch is flipped from outside the enclosure).
    targets: {"subject": <Switch component>}
    """

    action_id = "open_switch"
    description = "Open a switch (set it to the off/open position). Safe to call even if already open."
    cost = ActionCost(time=10.0)
    mutates_graph = True

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [AffordanceRequirement("subject", Affordance.TOGGLABLE)],
            targets, context,
        )
        return ok, "; ".join(failures)

    def execute(self, targets, graph, context, last_result):
        from diagnosable_systems_simulation.world.components import Switch
        sw: Switch = targets["subject"]  # type: ignore[assignment]
        if not sw.is_closed:
            return ActionResult(
                message=f"Switch {sw.display_name!r} is already open — no change made."
            )
        sw.is_closed = False
        return ActionResult(message=f"Switch {sw.display_name!r} is now open.")


class CloseSwitch(Action):
    """
    Set a switch to the closed (on) position.

    **Idempotent**: if the switch is already closed, no mutation is made and
    the message reports the existing state.  The action never fails because
    the switch is already in the target position.

    Prefer this over a "toggle" action when calling through the NL
    interface: the agent has no access to the current simulation state, so
    a toggle would be ambiguous; a state-targeting action is always safe.

    Requires TOGGLABLE affordance (no REACHABLE needed — switch is flipped from outside the enclosure).
    targets: {"subject": <Switch component>}
    """

    action_id = "close_switch"
    description = "Close a switch (set it to the on/closed position). Safe to call even if already closed."
    cost = ActionCost(time=10.0)
    mutates_graph = True

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [AffordanceRequirement("subject", Affordance.TOGGLABLE)],
            targets, context,
        )
        return ok, "; ".join(failures)

    def execute(self, targets, graph, context, last_result):
        from diagnosable_systems_simulation.world.components import Switch
        sw: Switch = targets["subject"]  # type: ignore[assignment]
        if sw.is_closed:
            return ActionResult(
                message=f"Switch {sw.display_name!r} is already closed — no change made."
            )
        sw.is_closed = True
        return ActionResult(message=f"Switch {sw.display_name!r} is now closed.")


class ReplaceComponent(Action):
    """
    Replace a faulty component with a fresh one (restores nominal parameters).

    Requires REACHABLE and REPLACEABLE affordance. Clears fault overlay. Consumes a replacement part.

    targets: {"subject": <any Component>}
    """

    action_id = "replace_component"
    description = "Replace a component with a new unit."
    mutates_graph = True

    def __init__(self, replacement_part_id: str, replacement_cost: float = 1.0):
        self.replacement_part_id = replacement_part_id
        self.cost = ActionCost(
            time=120.0,
            resources_consumed={replacement_part_id: replacement_cost},
        )

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [
                AffordanceRequirement("subject", Affordance.REACHABLE),
                AffordanceRequirement("subject", Affordance.REPLACEABLE),
            ],
            targets, context,
        )
        return ok, "; ".join(failures)

    def execute(self, targets, graph, context, last_result):
        comp = targets["subject"]
        comp.clear_fault()
        from diagnosable_systems_simulation.world.components import Fuse, Switch
        if isinstance(comp, Fuse):
            comp.is_blown = False
        if isinstance(comp, Switch):
            comp.is_closed = True
        return ActionResult(
            message=(
                f"Replaced {comp.display_name!r} with "
                f"{self.replacement_part_id!r}. Parameters restored to nominal."
            ),
        )


class InvertEnclosure(Action):
    """
    Lift and invert an enclosure (e.g. flip a cube upside-down).

    Makes components inside visible and reachable through the open bottom face.
    Requires REACHABLE and MOVABLE affordance on the enclosure.

    targets: {"enclosure": <Component representing the enclosure>}
    """

    action_id = "invert_enclosure"
    description = "Lift and invert an enclosure to look inside."
    cost = ActionCost(time=10.0)

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [
                AffordanceRequirement("subject", Affordance.REACHABLE),
                AffordanceRequirement("subject", Affordance.MOVABLE),
            ],
            targets, context,
        )
        return ok, "; ".join(failures)

    def execute(self, targets, graph, context, last_result):
        from diagnosable_systems_simulation.world.components import PhysicalEnclosure
        enc: PhysicalEnclosure = targets["subject"]
        enc.is_inverted = True
        return ActionResult(message=f"Enclosure {enc.display_name!r} is now inverted.")


class RestoreEnclosure(Action):
    """
    Put an inverted enclosure back in its normal orientation.

    Requires REACHABLE and MOVABLE affordance on the enclosure.
    targets: {"enclosure": <Component representing the enclosure>}
    """

    action_id = "restore_enclosure"
    description = "Return an inverted enclosure to its normal orientation."
    cost = ActionCost(time=10.0)

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [
                AffordanceRequirement("subject", Affordance.REACHABLE),
                AffordanceRequirement("subject", Affordance.MOVABLE),
            ],
            targets, context,
        )
        return ok, "; ".join(failures)

    def execute(self, targets, graph, context, last_result):
        from diagnosable_systems_simulation.world.components import PhysicalEnclosure
        enc: PhysicalEnclosure = targets["subject"]
        enc.is_inverted = False
        return ActionResult(message=f"Enclosure {enc.display_name!r} restored to normal orientation.")


class OpenPeephole(Action):
    """
    Open a peephole on an enclosure face.

    Sets ``peephole.is_open = True``; conditional affordances on internal
    components pick up the change automatically.
    Requires REACHABLE and OPENABLE affordance on the peephole component.

    targets: {"subject": <Peephole>}
    """

    action_id = "open_peephole"
    description = "Open a peephole to observe internal components."
    cost = ActionCost(time=5.0)

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [
                AffordanceRequirement("subject", Affordance.REACHABLE),
                AffordanceRequirement("subject", Affordance.OPENABLE),
            ],
            targets, context,
        )
        return ok, "; ".join(failures)

    def execute(self, targets, graph, context, last_result):
        from diagnosable_systems_simulation.world.components import Peephole
        peephole: Peephole = targets["subject"]
        peephole.is_open = True
        return ActionResult(message=f"Peephole {peephole.display_name!r} is now open.")


class ClosePeephole(Action):
    """
    Close an open peephole.

    Sets ``peephole.is_open = False``.
    Requires REACHABLE and CLOSEABLE affordance on the peephole component.

    targets: {"subject": <Peephole>}
    """

    action_id = "close_peephole"
    description = "Close an open peephole."
    cost = ActionCost(time=5.0)

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [
                AffordanceRequirement("subject", Affordance.REACHABLE),
                AffordanceRequirement("subject", Affordance.CLOSEABLE),
            ],
            targets, context,
        )
        return ok, "; ".join(failures)

    def execute(self, targets, graph, context, last_result):
        from diagnosable_systems_simulation.world.components import Peephole
        peephole: Peephole = targets["subject"]
        peephole.is_open = False
        return ActionResult(message=f"Peephole {peephole.display_name!r} is now closed.")


class AdjustPotentiometer(Action):
    """
    Adjust the wiper position of a potentiometer.

    Requires REACHABLE and ADJUSTABLE affordance.

    targets: {"subject": <Potentiometer component>}
    """

    action_id = "adjust_potentiometer"
    description = "Adjust the wiper position of a potentiometer."
    cost = ActionCost(time=20.0)
    mutates_graph = True

    def __init__(self, new_position: float):
        if not 0.0 <= new_position <= 1.0:
            raise ValueError("Wiper position must be in [0, 1].")
        self.new_position = new_position

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [
                AffordanceRequirement("subject", Affordance.REACHABLE),
                AffordanceRequirement("subject", Affordance.ADJUSTABLE),
            ],
            targets, context,
        )
        return ok, "; ".join(failures)

    def execute(self, targets, graph, context, last_result):
        from diagnosable_systems_simulation.world.components import Potentiometer
        pot: Potentiometer = targets["subject"]  # type: ignore[assignment]
        old = pot.wiper_position
        pot.wiper_position = self.new_position
        return ActionResult(
            message=f"Potentiometer {pot.display_name!r} wiper: {old:.2f} → {self.new_position:.2f}.",
        )


class TestContinuity(Action):
    """
    Test the internal resistance of a single component by placing one probe on
    each of its two terminals (ohmmeter mode).

    This measures **only the component itself** — not the circuit path around it.
    It is equivalent to desoldering the component and measuring it in isolation:
    parallel paths in the live circuit do not affect the reading, because the
    probes are placed directly across the component's own terminals.

    Requires REACHABLE and MEASURABLE affordance and "multimeter" in tools_in_hand.

    Compares ``current_parameters()["resistance"]`` against the nominal value and
    reports one of: nominal / open circuit (R > 1 MΩ) / short circuit (R < 0.01 Ω) /
    degraded.

    If a port is floating (cable end disconnected from the circuit), the action
    still reads the component's own resistance normally — the cable wire itself
    is still intact and conducting. The floating port is surfaced as a NEARBY
    ANOMALY, because a technician holding the probes would physically see the
    dangling end.

    For testing continuity **between two arbitrary points** in the circuit
    (a point-to-point path check), use ``TestPathContinuity`` instead.

    targets: {"subject": <any Component with a "resistance" parameter>}
    """

    action_id = "test_continuity"
    description = "Measure resistance / continuity with a multimeter (ohmmeter mode)."
    cost = ActionCost(time=20.0, equipment=["multimeter"])

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [
                AffordanceRequirement("subject", Affordance.REACHABLE),
                AffordanceRequirement("subject", Affordance.MEASURABLE),
                ToolRequirement("multimeter"),
            ],
            targets, context,
        )
        return ok, "; ".join(failures)

    def execute(self, targets, graph, context, last_result):
        comp = targets["subject"]
        params = comp.current_parameters()
        nominal = comp.nominal_parameters()

        record = ObservationRecord(
            component_id=comp.component_id,
            action_id=self.action_id,
        )

        if "resistance" not in params:
            record.add("note", "Component has no resistance parameter.")
            return ActionResult(
                observation=record,
                message=f"Cannot test continuity of {comp.display_name!r}: no resistance parameter.",
            )

        r_now = params["resistance"]
        r_nom = nominal.get("resistance", r_now)
        record.add("resistance_measured", round(r_now, 4), "Ω")
        record.add("resistance_nominal", round(r_nom, 4), "Ω")

        # Check nominal first so components with very low/high nominal R
        # (e.g. closed switch with Ron=1e-6) are not mis-classified.
        if abs(r_now - r_nom) / max(r_nom, 1.0) < 0.05:
            status = "nominal"
        elif r_now > 1e6:
            status = "open circuit"
        elif r_now < 0.01:
            status = "short circuit"
        else:
            status = "degraded"

        record.add("status", status)

        anomalies = _nearby_anomalies(comp, graph)
        # A floating port on the component itself means one of its own cable ends
        # is disconnected. A technician probing the component would physically
        # notice this (dangling wire end), so it is surfaced as an anomaly.
        floating = [p.name for p in comp.ports if not p.is_connected()]
        if floating:
            record.add("disconnected_ports", str(floating))
            anomalies = [
                f"port(s) {floating} are floating — this component's cable end is disconnected from the circuit"
            ] + anomalies
        if anomalies:
            record.add("nearby_anomalies", "; ".join(anomalies))
        anomaly_suffix = (" NEARBY ANOMALY: " + "; ".join(anomalies)) if anomalies else ""
        return ActionResult(
            observation=record,
            message=f"{comp.display_name!r} continuity: {status} (R={r_now:.2f} Ω, nominal={r_nom:.2f} Ω).{anomaly_suffix}",
        )


class TestPathContinuity(Action):
    """
    Test the Thevenin resistance of the conductive path between two
    arbitrary circuit points using a multimeter (ohmmeter mode).

    Unlike ``TestContinuity`` (which probes across a single component),
    this places one probe on a terminal of *source* and the other on a
    terminal of *sink*, measuring the total resistance of the path between
    them — including all series elements and parallel shortcuts.

    The measurement simulates a de-energised circuit (all independent
    voltage sources zeroed, as with a real ohmmeter) by injecting a 1 mV
    test source between the two probe nodes and measuring the resulting
    current: R = 0.001 / |I_test|.

    Classification:
      - R < 1 Ω  → **short** (continuous, low-resistance path)
      - 1–1 MΩ   → **resistive** (path exists but high resistance)
      - R ≥ 1 MΩ → **open circuit** (no conductive path)

    Requires REACHABLE + MEASURABLE on both *source* and *sink*, and
    "multimeter" in tools_in_hand.

    Optional constructor kwargs
    ---------------------------
    source_port : str
        Port name on the source component to probe (default: first
        connected port, then first port overall).
    sink_port : str
        Port name on the sink component to probe (default: same rule).

    targets: {"source": <Component>, "sink": <Component>}
    """

    action_id = "test_path_continuity"
    description = (
        "Measure the Thevenin resistance between a terminal of one component "
        "and a terminal of another (point-to-point continuity test)."
    )
    cost = ActionCost(time=30.0, equipment=["multimeter"])

    def __init__(self, source_port: str = "", sink_port: str = ""):
        self.source_port = source_port
        self.sink_port = sink_port

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [
                AffordanceRequirement("source", Affordance.REACHABLE),
                AffordanceRequirement("source", Affordance.MEASURABLE),
                AffordanceRequirement("sink", Affordance.REACHABLE),
                AffordanceRequirement("sink", Affordance.MEASURABLE),
                ToolRequirement("multimeter"),
            ],
            targets, context,
        )
        return ok, "; ".join(failures)

    @staticmethod
    def _pick_node(comp, preferred_port: str) -> "str | None":
        """Return node_id for *preferred_port*, or the first connected port."""
        if preferred_port:
            p = next((p for p in comp.ports if p.name == preferred_port), None)
            if p is not None:
                return p.node_id
        # Fall back to first connected port, then first port overall
        for p in comp.ports:
            if p.is_connected():
                return p.node_id
        return comp.ports[0].node_id if comp.ports else None

    def execute(self, targets, graph, context, last_result):
        source = targets["source"]
        sink = targets["sink"]

        record = ObservationRecord(
            component_id=f"{source.component_id}_to_{sink.component_id}",
            action_id=self.action_id,
        )

        node_a = self._pick_node(source, self.source_port)
        node_b = self._pick_node(sink, self.sink_port)

        record.add("source", source.display_name)
        record.add("sink", sink.display_name)
        record.add("probe_node_source", str(node_a))
        record.add("probe_node_sink", str(node_b))

        if node_a is None or node_b is None:
            record.add("status", "open circuit")
            record.add("note", "One or both probe nodes are unresolvable.")
            return ActionResult(
                observation=record,
                message=(
                    f"Path continuity {source.display_name!r} → {sink.display_name!r}: "
                    f"open circuit (probe node unresolvable)."
                ),
            )

        if node_a == node_b:
            record.add("resistance", 0.0, "Ω")
            record.add("status", "short")
            return ActionResult(
                observation=record,
                message=(
                    f"Path continuity {source.display_name!r} → {sink.display_name!r}: "
                    f"short — both probes are on the same circuit node."
                ),
            )

        backend = context.extra.get("backend")
        if backend is None:
            record.add("note", "No backend available for path continuity measurement.")
            return ActionResult(
                observation=record,
                success=False,
                message="test_path_continuity requires a simulation backend.",
            )

        r = backend.solve_continuity(graph, node_a, node_b)
        record.add("resistance", round(r, 4), "Ω")

        if r < 1.0:
            status = "short"
        elif r < 1e6:
            status = "resistive"
        else:
            status = "open circuit"
        record.add("status", status)

        return ActionResult(
            observation=record,
            message=(
                f"Path continuity {source.display_name!r} → {sink.display_name!r}: "
                f"{status} (R={r:.2f} Ω)."
            ),
        )


class TestDiode(Action):
    """
    Test a diode or LED using the multimeter's diode-test mode.

    Requires REACHABLE and MEASURABLE affordance and "multimeter" in tools_in_hand.
    Reads ``current_parameters()["forward_voltage"]`` and compares it against
    the nominal value.  Reports one of:
      - nominal           — Vf within 10 % of spec
      - shorted           — Vf ≈ 0 V (< 0.05 V)
      - open circuit      — Vf not present / very high (> 5 V or key absent)
      - degraded          — Vf present but outside spec

    targets: {"subject": <Diode or LED component>}
    """

    action_id = "test_diode"
    description = "Test a diode or LED forward voltage with a multimeter (diode-test mode)."
    cost = ActionCost(time=20.0, equipment=["multimeter"])

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [
                AffordanceRequirement("subject", Affordance.REACHABLE),
                AffordanceRequirement("subject", Affordance.MEASURABLE),
                ToolRequirement("multimeter"),
            ],
            targets, context,
        )
        return ok, "; ".join(failures)

    def execute(self, targets, graph, context, last_result):
        comp = targets["subject"]
        params = comp.current_parameters()
        nominal = comp.nominal_parameters()

        record = ObservationRecord(
            component_id=comp.component_id,
            action_id=self.action_id,
        )

        if "forward_voltage" not in nominal:
            record.add("note", "Component has no forward_voltage parameter.")
            return ActionResult(
                observation=record,
                message=f"Cannot run diode test on {comp.display_name!r}: not a diode/LED.",
            )

        vf_nom = nominal["forward_voltage"]
        vf_now = params.get("forward_voltage", vf_nom)
        record.add("vf_measured", round(vf_now, 4), "V")
        record.add("vf_nominal", round(vf_nom, 4), "V")

        if vf_now < 0.05:
            status = "shorted"
        elif vf_now > 5.0:
            status = "open circuit"
        elif abs(vf_now - vf_nom) / max(vf_nom, 0.01) < 0.10:
            status = "nominal"
        else:
            status = "degraded"

        record.add("status", status)
        return ActionResult(
            observation=record,
            message=f"{comp.display_name!r} diode test: {status} (Vf={vf_now:.3f} V, nominal={vf_nom:.3f} V).",
        )


class InspectConnections(Action):
    """
    Physically inspect which cables are plugged into each port of a component.

    Requires OBSERVABLE affordance (the technician must be able to see the ports).
    For each port, reports the node_id it is connected to and any Cable instances
    sharing that node (i.e. plugged into the same junction).
    A port whose node_id is None is reported as "disconnected (floating)".

    targets: {"subject": <any Component>}
    """

    action_id = "inspect_connections"
    description = "Inspect which cables are physically connected to each port."
    cost = ActionCost(time=10.0)

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [AffordanceRequirement("subject", Affordance.OBSERVABLE)],
            targets, context,
        )
        return ok, "; ".join(failures)

    def execute(self, targets, graph, context, last_result):
        from diagnosable_systems_simulation.world.components import Cable
        comp = targets["subject"]
        record = ObservationRecord(
            component_id=comp.component_id,
            action_id=self.action_id,
        )

        # Build node → [cable_display_name, ...] index from the circuit graph
        node_cables: dict[str, list[str]] = {}
        for edge in graph.get_netlist():
            if isinstance(edge.component, Cable):
                for node_id in edge.port_nodes.values():
                    node_cables.setdefault(node_id, []).append(edge.component.display_name)

        lines = []
        for port in comp.ports:
            node_id = port.node_id
            if node_id is None:
                lines.append(f"port '{port.name}': disconnected (floating)")
                record.add(f"port_{port.name}", "disconnected")
            else:
                cables = node_cables.get(node_id, [])
                cable_str = ", ".join(cables) if cables else "no cable"
                lines.append(f"port '{port.name}': {cable_str}")
                record.add(f"port_{port.name}", cable_str)

        summary = "; ".join(lines)
        anomalies = _nearby_anomalies(comp, graph)
        if anomalies:
            record.add("nearby_anomalies", "; ".join(anomalies))
        anomaly_suffix = (" NEARBY ANOMALY: " + "; ".join(anomalies)) if anomalies else ""
        return ActionResult(
            observation=record,
            message=f"Connections on {comp.display_name!r}: {summary}.{anomaly_suffix}",
        )


class VerifyRepair(Action):
    """
    Hypothesis-verification action.

    The NL interface resolves a free-text fault hypothesis to a concrete
    component ID by mapping it to this action.  Execution is intentionally
    a no-op: the service agent reads the resolved ``subject`` ID from the
    parsed action entry and calls ``DiagnosableSystem.repair_component()``.

    targets: {"subject": <any Component>}
    """

    action_id = "verify_repair"
    description = (
        "Identify the component that the technician suspects is faulty and "
        "mark it for hypothesis verification (repair-and-test)."
    )
    cost = ActionCost(time=120.0)

    def check_preconditions(self, targets, context):
        return True, ""

    def execute(self, targets, graph, context, last_result):
        comp = targets.get("subject")
        name = comp.display_name if comp is not None else "unknown"
        return ActionResult(message=f"Marked '{name}' as hypothesis-verification target.")
