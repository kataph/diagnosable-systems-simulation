from __future__ import annotations

from diagnosable_systems_simulation.actions.base import Action, ActionCost, ActionResult, CompositeAction
from diagnosable_systems_simulation.actions.preconditions import (
    AffordanceRequirement, PreconditionChecker
)
from diagnosable_systems_simulation.world.affordances import Affordance


class DisconnectCable(Action):
    """
    Physically detach a cable from the circuit.

    The cable's ports become floating (node_id = None).
    DETACHABLE affordance is replaced with RECONNECTABLE. 
    
    Neightbouring components are also affected: their status
    as disconnected is saved in their affordances. 

    targets: {"cable": <Cable component>}
    port_names: ports to disconnect; None means all ports.
    """

    action_id = "disconnect_cable"
    description = "Detach a cable's connector from the circuit."
    cost = ActionCost(time=10.0)
    mutates_graph = True

    def __init__(self, port_names: list[str] | None = None):
        self.port_names = port_names

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [AffordanceRequirement("subject", Affordance.DETACHABLE)],
            targets, context,
        )
        return ok, "; ".join(failures)

    def execute(self, targets, graph, context, last_result):
        cable = targets["subject"]
        ports = self.port_names or [p.name for p in cable.ports]
        # Collect node_ids BEFORE disconnecting so we can find peer components.
        port_nodes: dict[str, str] = {
            p: cable.port(p).node_id
            for p in ports
            if cable.port(p).node_id is not None
        }
        # Mark every non-cable component sharing a node with a disconnected port
        # as RECONNECTABLE and record the port-to-cable mapping on it.  This lets
        # the repair layer know that fixing e.g. "Switch3" means reconnecting
        # the cable port that was detached from it.
        for cable_port_name, node_id in port_nodes.items():
            for edge in graph.get_netlist():
                if edge.component_id == cable.component_id:
                    continue
                for peer_port_name, peer_node_id in edge.port_nodes.items():
                    if peer_node_id != node_id:
                        continue
                    peer = edge.component
                    if not hasattr(peer, "_detached_cable_ports"):
                        peer._detached_cable_ports = {}
                    peer._detached_cable_ports[peer_port_name] = (
                        cable.component_id, cable_port_name, node_id
                    )
                    peer.affordances.add(Affordance.RECONNECTABLE)
        # Save original connections on the cable and physically disconnect.
        # Use setdefault so that _orig_connections saved by _add_loose_connection
        # (for a port that is already floating) is not overwritten with an empty entry.
        if not hasattr(cable, '_orig_connections'):
            cable._orig_connections = {}
        for port_name, node_id in port_nodes.items():
            cable._orig_connections.setdefault(port_name, node_id)
        disconnected = [
            p for p in ports
            if port_nodes.get(p) is not None
            and graph.disconnect_port(cable.component_id, p) is not None
        ]
        cable.affordances.remove(Affordance.DETACHABLE)
        cable.affordances.add(Affordance.RECONNECTABLE)
        return ActionResult(message=f"Disconnected ports {disconnected} of {cable.display_name!r}.")


class ReconnectCable(Action):
    """
    Reconnect a previously detached cable.

    If *connections* is omitted (or empty), the cable is restored to its
    original wiring using the ``_orig_connections`` dict saved by
    ``DisconnectCable``.  This is the normal diagnostic use-case: a
    technician puts the cable back where it was without needing to know
    the underlying node IDs.
    
    Neightbouringh components status is also ripristinated. 

    targets: {"subject": <Cable>}
    connections: optional port name -> node_id override
    """

    action_id = "reconnect_cable"
    description = "Reconnect a detached cable to its original position (or to specified nodes)."
    cost = ActionCost(time=10.0)
    mutates_graph = True

    def __init__(self, connections: dict[str, str] | None = None):
        self.connections = connections or {}

    def check_preconditions(self, targets, context):
        ok, failures = PreconditionChecker.check_all(
            [AffordanceRequirement("subject", Affordance.RECONNECTABLE)],
            targets, context,
        )
        return ok, "; ".join(failures)

    def execute(self, targets, graph, context, last_result):
        cable = targets["subject"]
        connections = self.connections or getattr(cable, "_orig_connections", {})
        if not connections:
            return ActionResult(
                success=False,
                message=(
                    f"Cannot reconnect {cable.display_name!r}: no connections specified "
                    f"and no original connection data available."
                ),
            )
        for cable_port_name, node_id in connections.items():
            # The original node this port was disconnected from (may differ from
            # node_id when SwapCablePolarities reconnects to a different node).
            orig_node_id = getattr(cable, "_orig_connections", {}).get(cable_port_name, node_id)
            graph.reconnect_port(cable.component_id, cable_port_name, node_id)
            # removes disconnected state change from neighbouring components.
            # We match on orig_node_id (the node the port was on before
            # DisconnectCable ran) because that is what peer._detached_cable_ports
            # recorded — not the new node_id after a polarity swap.
            for edge in graph.get_netlist():
                if edge.component_id == cable.component_id:
                    continue
                peer = edge.component
                if not hasattr(peer, "_detached_cable_ports"):
                    continue

                to_delete = []
                for p_port, (cid, c_port, nid) in peer._detached_cable_ports.items():
                    if cid == cable.component_id and nid == orig_node_id:
                        # This was the entry created by DisconnectCable
                        to_delete.append(p_port)
                for p_port in to_delete:
                    del peer._detached_cable_ports[p_port]

                if not peer._detached_cable_ports:
                    peer.affordances.remove(Affordance.RECONNECTABLE)
                    del peer._detached_cable_ports
                  
        # Remove any LooseConnectionCoupling on this cable so the fault doesn't
        # re-disconnect the port on the next simulate() call.
        from diagnosable_systems_simulation.electrical_simulation.couplings import LooseConnectionCoupling
        system = context.extra.get("_system")
        if system is not None and hasattr(system, '_runner'):
            system._runner.couplings = [
                c for c in system._runner.couplings
                if not (isinstance(c, LooseConnectionCoupling) and c.component_id == cable.component_id)
            ]
        cable.affordances.remove(Affordance.RECONNECTABLE)
        cable.affordances.add(Affordance.DETACHABLE)
        return ActionResult(message=f"Reconnected {cable.display_name!r} to original position.")


class ShortCircuit(Action):
    """
    Create a short between two nodes (fault injection).

    targets: {"start": "component1", "end": "component2"}
    """

    action_id = "short_circuit"
    description = "Insert a short circuit between two nodes."
    cost = ActionCost(time=30.0)
    mutates_graph = True

    def __init__(self, node_a: str, node_b: str, short_id: str):
        self.node_a = node_a
        self.node_b = node_b
        self.short_id = short_id

    def check_preconditions(self, targets, context):
        return True, ""

    def execute(self, targets, graph, context, last_result):
        graph.short_nodes(self.node_a, self.node_b, self.short_id)

        comp1 = targets.get("start")
        comp2 = targets.get("end")
        if comp1 is not None and comp2 is not None:
            comp1.apply_fault({"short_circuit_with": comp2.component_id, "short_graph_id": self.short_id, "short_node_a": self.node_a, "short_node_b": self.node_b})
            comp2.apply_fault({"short_circuit_with": comp1.component_id, "short_graph_id": self.short_id, "short_node_a": self.node_a, "short_node_b": self.node_b})
            return ActionResult(
                message=f"Applied short fault overlay to components {comp1.display_name!r} and {comp2.display_name!r}.",
            )

        return ActionResult(message=f"Shorted nodes {self.node_a!r} and {self.node_b!r}.")


class DegradeComponent(Action):
    """
    Apply a parameter degradation to a component (fault injection).

    Stores a fault overlay on the component; no topology change.

    targets: {"subject": <any Component>}
    degradation: dict of parameter overrides, e.g. {"resistance": 1e9}
    """

    action_id = "degrade_component"
    description = "Degrade one or more electrical parameters of a component."
    cost = ActionCost(time=120.0)
    mutates_graph = True

    def __init__(self, degradation: dict):
        self.degradation = degradation

    def check_preconditions(self, targets, context):
        if "subject" not in targets:
            return False, "No 'subject' target provided."
        return True, ""

    def execute(self, targets, graph, context, last_result):
        comp = targets["subject"]
        comp.apply_fault(self.degradation)
        return ActionResult(
            message=f"Applied fault overlay {self.degradation} to {comp.display_name!r}.",
        )


class BlowFuse(Action):
    """
    Blow a fuse (fault injection).

    targets: {"fuse": <Fuse component>}
    """

    action_id = "blow_fuse"
    description = "Blow a fuse, making it an open circuit."
    cost = ActionCost(time=120.0)
    mutates_graph = True

    def check_preconditions(self, targets, context):
        if "subject" not in targets:
            return False, "No 'fuse' target provided."
        return True, ""

    def execute(self, targets, graph, context, last_result):
        from diagnosable_systems_simulation.world.components import Fuse
        fuse: Fuse = targets["subject"]  # type: ignore[assignment]
        fuse.is_blown = True
        return ActionResult(message=f"Fuse {fuse.display_name!r} is now blown.")


class ForceSwitch(Action):
    """
    Force a switch open or closed (fault injection — bypasses normal toggle).

    targets: {"switch": <Switch component>}
    """

    action_id = "force_switch"
    description = "Force a switch to a specific position as a fault."
    cost = ActionCost(time=120.0)
    mutates_graph = True

    def __init__(self, is_closed: bool):
        self.is_closed = is_closed

    def check_preconditions(self, targets, context):
        if "subject" not in targets:
            return False, "No 'switch' target provided."
        return True, ""

    def execute(self, targets, graph, context, last_result):
        sw = targets["subject"]
        sw.apply_fault({"is_closed": self.is_closed})
        state = "closed" if self.is_closed else "open"
        return ActionResult(message=f"Switch {sw.display_name!r} forced {state}.")


class ReverseBattery(Action):
    """
    Physically reverse the polarity of a battery (install it backwards).

    Toggles the sign of the battery voltage: +V → -V → +V.
    When the result equals the nominal voltage the fault overlay is cleared
    (battery is back to correct orientation).

    This is the canonical action for both fault injection (correct → reversed)
    and repair (reversed → correct) of a reversed-polarity battery fault.
    Requires REACHABLE affordance on the VoltageSource.

    targets: {"subject": <VoltageSource>}
    """

    action_id = "reverse_battery"
    description = "Reverse the polarity of a battery (install it backwards or correct it)."
    cost = ActionCost(time=30.0)
    mutates_graph = True

    def check_preconditions(self, targets, context):
        if "subject" not in targets:
            return False, "No 'subject' target provided."
        return True, ""

    def execute(self, targets, graph, context, last_result):
        comp = targets["subject"]
        nominal_v = comp.nominal_parameters()["voltage"]
        current_v = comp.current_parameters()["voltage"]
        new_v = -current_v
        if new_v == nominal_v:
            comp.clear_fault()
        else:
            comp.apply_fault({"voltage": new_v})
        return ActionResult(
            message=f"Battery {comp.display_name!r} polarity reversed (voltage now {new_v:.1f} V)."
        )


class SwapCablePolarities(CompositeAction):
    """
    Swap the connections of a specified port between two cables.

    Models crossing (or uncrossing) two cables: e.g. the positive input of
    cable A ends up connected to the node previously held by cable B, and
    vice versa.  Calling it twice restores the original wiring.

    Implemented as a CompositeAction of 2 × DisconnectCable + 2 × ReconnectCable,
    so its cost is automatically 4 × 40s = 160s.

    targets: {"cable_a": <Cable>, "cable_b": <Cable>}
    """

    action_id = "swap_cable_polarities"
    description = "Swap the connections of a specified port between two cables (cross/uncross)."

    def __init__(self, cable_a_id: str = "", cable_b_id: str = "", port_name: str = "p"):
        self.cable_a_id = cable_a_id
        self.cable_b_id = cable_b_id
        self.port_name = port_name
        self._cable_a = None
        self._cable_b = None

    def check_preconditions(self, targets, context):
        self._cable_a = targets.get("cable_a")
        self._cable_b = targets.get("cable_b")
        if self._cable_a is None or self._cable_b is None:
            return False, "Both 'cable_a' and 'cable_b' targets are required."
        from diagnosable_systems_simulation.actions.preconditions import AffordanceRequirement, PreconditionChecker
        ok, failures = PreconditionChecker.check_all(
            [
                AffordanceRequirement("cable_a", Affordance.DETACHABLE),
                AffordanceRequirement("cable_b", Affordance.DETACHABLE),
            ],
            targets, context,
        )
        return ok, "; ".join(failures)

    def execute(self, targets, graph, context, last_result):
        self._cable_a = targets.get("cable_a")
        self._cable_b = targets.get("cable_b")
        return super().execute(targets, graph, context, last_result)

    @property
    def sub_actions(self):
        a, b, pn = self._cable_a, self._cable_b, self.port_name
        node_a = a.port(pn).node_id
        node_b = b.port(pn).node_id
        return [
            (DisconnectCable(port_names=[pn]), {"subject": a}),
            (DisconnectCable(port_names=[pn]), {"subject": b}),
            (ReconnectCable({pn: node_b}),     {"subject": a}),
            (ReconnectCable({pn: node_a}),     {"subject": b}),
        ]
