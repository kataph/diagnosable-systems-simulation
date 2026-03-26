from __future__ import annotations

from diagnosable_systems_simulation.actions.base import Action, ActionCost, ActionResult
from diagnosable_systems_simulation.actions.observation import ObservationRecord, observe_component
from diagnosable_systems_simulation.actions.preconditions import (
    AffordanceRequirement, ToolRequirement, PreconditionChecker
)
from diagnosable_systems_simulation.world.affordances import Affordance


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
    Measure voltage at a component's ports using a multimeter.

    Requires OBSERVABLE and MEASURABLE affordance and "multimeter" in tools_in_hand.
    Returns an ObservationRecord with port voltages and branch current.

    targets: {"subject": <any Component>}
    """

    action_id = "measure_voltage"
    description = "Measure voltage at component ports with a multimeter."
    cost = ActionCost(time=20.0, equipment=["multimeter"])

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [
                AffordanceRequirement("subject", Affordance.OBSERVABLE),
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
        else:
            for port in comp.ports:
                if port.node_id is not None:
                    v = last_result.voltage(port.node_id)
                    if v is not None:
                        record.add(f"voltage_{port.name}", round(v, 4), "V")
            i = last_result.current(comp.component_id)
            if i is not None:
                record.add("current", round(i, 6), "A")
        return ActionResult(
            observation=record,
            message=f"Measured {comp.display_name!r}.",
        )


class ToggleSwitch(Action):
    """
    Flip a switch between open and closed.

    Requires TOGGLABLE affordance.

    targets: {"switch": <Switch component>}
    """

    action_id = "toggle_switch"
    description = "Flip a switch open or closed."
    cost = ActionCost(time=10.0)
    mutates_graph = True

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [AffordanceRequirement("subject", Affordance.TOGGLABLE),
            AffordanceRequirement("subject", Affordance.OBSERVABLE),
             ],
            targets, context,
        )
        return ok, "; ".join(failures)

    def execute(self, targets, graph, context, last_result):
        from diagnosable_systems_simulation.world.components import Switch
        sw: Switch = targets["subject"]  # type: ignore[assignment]
        sw.is_closed = not sw.is_closed
        state = "closed" if sw.is_closed else "open"
        return ActionResult(message=f"Switch {sw.display_name!r} is now {state}.")


class ReplaceComponent(Action):
    """
    Replace a faulty component with a fresh one (restores nominal parameters).

    Requires REPLACEABLE affordance. Clears fault overlay. Consumes a replacement part.

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
            [AffordanceRequirement("subject", Affordance.REPLACEABLE)],
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

    Makes components inside visible through the open bottom face.
    Requires MOVABLE affordance on the enclosure.

    targets: {"enclosure": <Component representing the enclosure>}
    """

    action_id = "invert_enclosure"
    description = "Lift and invert an enclosure to look inside."
    cost = ActionCost(time=10.0)

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [AffordanceRequirement("subject", Affordance.MOVABLE)],
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

    targets: {"enclosure": <Component representing the enclosure>}
    """

    action_id = "restore_enclosure"
    description = "Return an inverted enclosure to its normal orientation."
    cost = ActionCost(time=10.0)

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [AffordanceRequirement("subject", Affordance.MOVABLE)],
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
    Requires OPENABLE affordance on the peephole component.

    targets: {"subject": <Peephole>}
    """

    action_id = "open_peephole"
    description = "Open a peephole to observe internal components."
    cost = ActionCost(time=10.0)

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [AffordanceRequirement("subject", Affordance.OPENABLE)],
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
    Requires CLOSEABLE affordance on the peephole component.

    targets: {"subject": <Peephole>}
    """

    action_id = "close_peephole"
    description = "Close an open peephole."
    cost = ActionCost(time=10.0)

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [AffordanceRequirement("subject", Affordance.CLOSEABLE)],
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

    Requires ADJUSTABLE affordance.

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
            [AffordanceRequirement("subject", Affordance.ADJUSTABLE)],
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
    Test resistance / continuity of a component using a multimeter in ohmmeter mode.

    Requires MEASURABLE affordance and "multimeter" in tools_in_hand.
    Does NOT require the component to be visible (probes only need physical access).

    Compares ``current_parameters()["resistance"]`` against the nominal value and
    reports one of: nominal / open circuit (R > 1 MΩ) / short circuit (R < 0.01 Ω) /
    degraded.

    targets: {"subject": <any Component with a "resistance" parameter>}
    """

    action_id = "test_continuity"
    description = "Measure resistance / continuity with a multimeter (ohmmeter mode)."
    cost = ActionCost(time=20.0, equipment=["multimeter"])

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [
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

        if r_now > 1e6:
            status = "open circuit"
        elif r_now < 0.01:
            status = "short circuit"
        elif abs(r_now - r_nom) / max(r_nom, 1.0) < 0.05:
            status = "nominal"
        else:
            status = "degraded"

        record.add("status", status)
        return ActionResult(
            observation=record,
            message=f"{comp.display_name!r} continuity: {status} (R={r_now:.2f} Ω, nominal={r_nom:.2f} Ω).",
        )


class TestDiode(Action):
    """
    Test a diode or LED using the multimeter's diode-test mode.

    Requires MEASURABLE affordance and "multimeter" in tools_in_hand.
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
                lines.append(f"port '{port.name}' → node {node_id!r}: {cable_str}")
                record.add(f"port_{port.name}", cable_str)

        summary = "; ".join(lines)
        return ActionResult(
            observation=record,
            message=f"Connections on {comp.display_name!r}: {summary}.",
        )
