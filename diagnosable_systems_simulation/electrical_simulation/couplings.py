from __future__ import annotations

import random
from typing import Optional

from diagnosable_systems_simulation.electrical_simulation.circuit import CircuitGraph
from diagnosable_systems_simulation.electrical_simulation.results import SimulationResult
from diagnosable_systems_simulation.electrical_simulation.solver import PhysicalCoupling
from diagnosable_systems_simulation.world.context import WorldContext


class LooseConnectionCoupling(PhysicalCoupling):
    """
    Models an intermittent open circuit on a single cable port.

    On each simulation step, the port is randomly disconnected with
    probability *p* (default 0.5). When disconnected the port is
    immediately reconnected before the next step so the simulation
    can converge with either an open or closed connection.

    The coupling also marks the system context so that callers know not
    to trust a single passing simulation when this fault is active.
    """

    def __init__(self, component_id: str, port_name: str, p: float = 0.5) -> None:
        self.component_id = component_id
        self.port_name = port_name
        self.p = p
        self._currently_disconnected = False
        self._saved_node: Optional[int] = None

    def reset(self) -> None:
        """Return coupling to clean initial state (not mid-disconnect)."""
        self._currently_disconnected = False
        self._saved_node = None

    def apply(self, result: SimulationResult, graph: CircuitGraph, context: WorldContext) -> bool:
        context.extra["has_loose_connection"] = True

        if not graph.has_component(self.component_id):
            return False

        comp = graph.get_component(self.component_id)
        port = next((p for p in comp.ports if p.name == self.port_name), None)
        if port is None:
            return False

        if self._currently_disconnected:
            if self._saved_node is not None:
                graph.reconnect_port(self.component_id, self.port_name, self._saved_node)
            self._currently_disconnected = False
            self._saved_node = None
            return True
        else:
            if random.random() < self.p and port.is_connected():
                self._saved_node = port.node_id
                graph.disconnect_port(self.component_id, self.port_name)
                self._currently_disconnected = True
                return True
            return False


def _add_loose_connection(
    sys: "object",
    component_id: str,
    port_name: str,
    p: float = 0.5,
) -> None:
    """Attach a LooseConnectionCoupling to a DiagnosableSystem and flag the context."""
    coupling = LooseConnectionCoupling(component_id, port_name, p=p)
    sys._runner.couplings.append(coupling)
    sys.context.extra["has_loose_connection"] = True
