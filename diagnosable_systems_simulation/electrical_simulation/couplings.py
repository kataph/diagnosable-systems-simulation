from __future__ import annotations

import random
from typing import Optional

from diagnosable_systems_simulation.electrical_simulation.circuit import CircuitGraph
from diagnosable_systems_simulation.electrical_simulation.results import SimulationResult
from diagnosable_systems_simulation.electrical_simulation.solver import PhysicalCoupling
from diagnosable_systems_simulation.world.affordances import Affordance
from diagnosable_systems_simulation.world.context import WorldContext


class LooseConnectionCoupling(PhysicalCoupling):
    """
    Models an intermittent open circuit on a single cable port.

    Once per simulate() call, the coin is flipped: with probability *p*
    (default 0.5) the port is disconnected for the entire solve; otherwise
    it stays connected.  The state is held for all coupling-loop iterations
    within that solve and restored at the start of the next simulate() call.

    The coupling also marks the system context so that callers know not
    to trust a single passing simulation when this fault is active.
    """

    def __init__(self, component_id: str, port_name: str, p: float = 0.5) -> None:
        self.component_id = component_id
        self.port_name = port_name
        self.p = p
        self._currently_disconnected = False
        self._saved_node: Optional[int] = None
        self._flipped_this_run = False

    def reset(self) -> None:
        """Return coupling to clean initial state (not mid-disconnect)."""
        self._currently_disconnected = False
        self._saved_node = None
        self._flipped_this_run = False

    def apply(self, result: SimulationResult, graph: CircuitGraph, context: WorldContext) -> bool:
        context.extra["has_loose_connection"] = True

        if not graph.has_component(self.component_id):
            return False

        comp = graph.get_component(self.component_id)
        port = next((p for p in comp.ports if p.name == self.port_name), None)
        if port is None:
            return False

        if not self._flipped_this_run:
            # First iteration of this simulate() call: flip the coin once.
            self._flipped_this_run = True
            if self._currently_disconnected:
                # Previous simulate() left the port open — restore it now.
                if self._saved_node is not None:
                    graph.reconnect_port(self.component_id, self.port_name, self._saved_node)
                self._currently_disconnected = False
                self._saved_node = None
            if random.random() < self.p and port.is_connected():
                self._saved_node = port.node_id
                graph.disconnect_port(self.component_id, self.port_name)
                self._currently_disconnected = True
                return True
            return False
        else:
            # Subsequent iterations within the same simulate() call: hold state.
            return False


def _add_loose_connection(
    sys: "object",
    component_id: str,
    port_name: str,
    p: float = 0.5,
) -> None:
    """Attach a LooseConnectionCoupling to a DiagnosableSystem and flag the context.

    Also stores the original port connection in _orig_connections so that test_repair()
    can restore the port when the loose connection fault is repaired.
    """
    if component_id not in sys.all_components():
        return

    comp = sys.component(component_id)
    port = next((p for p in comp.ports if p.name == port_name), None)
    if port is None:
        return

    # Store original connection for repair
    if not hasattr(comp, '_orig_connections'):
        comp._orig_connections = {}
    comp._orig_connections[port_name] = port.node_id
    comp.affordances.add(Affordance.RECONNECTABLE)

    # Add the coupling
    coupling = LooseConnectionCoupling(component_id, port_name, p=p)
    sys._runner.couplings.append(coupling)
    sys.context.extra["has_loose_connection"] = True
    loose_ids = sys.context.extra.setdefault("loose_component_ids", set())
    loose_ids.add(component_id)
