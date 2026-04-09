from __future__ import annotations

from diagnosable_systems_simulation.actions.base import Action, ActionCost, ActionResult
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
    cost = ActionCost(time=40.0)
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
        cable._orig_connections = dict(port_nodes)
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
    cost = ActionCost(time=40.0)
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
            graph.reconnect_port(cable.component_id, cable_port_name, node_id)
            # removes disconnected state change from neightbouring components
            for edge in graph.get_netlist():
                if edge.component_id == cable.component_id:
                    continue
                for peer_port_name, peer_node_id in edge.port_nodes.items():
                    if peer_node_id != node_id:
                        continue
                    peer = edge.component
                    if not hasattr(peer, "_detached_cable_ports"):
                        continue

                    to_delete = []
                    for p_port, (cid, c_port, nid) in peer._detached_cable_ports.items():
                        if cid == cable.component_id and nid == node_id:
                            # This was the entry created by DisconnectCable
                            to_delete.append(p_port)
                    for p_port in to_delete:
                        del peer._detached_cable_ports[p_port]

                    if not peer._detached_cable_ports:
                        peer.affordances.discard(Affordance.RECONNECTABLE)
                        # Optional: remove the empty attribute to keep objects clean
                        del peer._detached_cable_ports
                  
        cable.affordances.remove(Affordance.RECONNECTABLE)
        cable.affordances.add(Affordance.DETACHABLE)
        return ActionResult(message=f"Reconnected {cable.display_name!r} to original position.")


class ShortCircuit(Action):
    """
    Create a short between two nodes (fault injection).

    targets: {} (no component targets — acts on nodes directly)
    """

    action_id = "short_circuit"
    description = "Insert a short circuit between two nodes."
    cost = ActionCost(time=40.0)
    mutates_graph = True

    def __init__(self, node_a: str, node_b: str, short_id: str):
        self.node_a = node_a
        self.node_b = node_b
        self.short_id = short_id

    def check_preconditions(self, targets, context):
        return True, ""

    def execute(self, targets, graph, context, last_result):
        graph.short_nodes(self.node_a, self.node_b, self.short_id)
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
    cost = ActionCost(time=40.0)
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
    cost = ActionCost(time=40.0)
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
    cost = ActionCost(time=40.0)
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
