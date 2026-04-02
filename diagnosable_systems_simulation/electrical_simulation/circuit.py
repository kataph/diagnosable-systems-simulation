from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from diagnosable_systems_simulation.world.components import Component


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class CircuitNode:
    node_id: str
    is_ground: bool = False

    def __repr__(self) -> str:
        tag = " [GND]" if self.is_ground else ""
        return f"Node({self.node_id!r}{tag})"


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

@dataclass
class CircuitEdge:
    """One component wired between two nodes."""
    component: Component
    # Maps port name -> node_id for every port of the component.
    # Kept as a dict so multi-port devices (potentiometers, etc.) are handled.
    port_nodes: dict[str, str]

    @property
    def component_id(self) -> str:
        return self.component.component_id

    def node_for_port(self, port_name: str) -> str:
        return self.port_nodes[port_name]

    def __repr__(self) -> str:
        return f"Edge({self.component_id!r}, ports={self.port_nodes})"


# ---------------------------------------------------------------------------
# CircuitGraph
# ---------------------------------------------------------------------------

class CircuitGraph:
    """
    Mutable graph of nodes and component edges.

    This is the single source of truth for circuit topology.
    Fault injection and diagnostic actions that change topology
    (disconnect, short, reconnect) operate exclusively on this object.

    Components whose parameters change but whose topology does not
    (degraded resistance, blown fuse) are handled via
    ``Component.apply_fault()`` and do not require graph surgery.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, CircuitNode] = {}
        self._edges: dict[str, CircuitEdge] = {}  # keyed by component_id

    # ------------------------------------------------------------------
    # Building the graph
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, is_ground: bool = False) -> CircuitNode:
        if node_id in self._nodes:
            raise ValueError(f"Node {node_id!r} already exists.")
        node = CircuitNode(node_id=node_id, is_ground=is_ground)
        self._nodes[node_id] = node
        return node

    def get_or_add_node(self, node_id: str, is_ground: bool = False) -> CircuitNode:
        if node_id not in self._nodes:
            return self.add_node(node_id, is_ground=is_ground)
        return self._nodes[node_id]

    def add_component(
        self,
        component: Component,
        connections: dict[str, str],
    ) -> CircuitEdge:
        """
        Wire ``component`` into the graph.

        ``connections`` maps each port name to an existing node_id.
        All referenced nodes must already exist (use ``get_or_add_node``).

        The component's ``ElectricalPort.node_id`` fields are updated
        in-place so the component itself knows where it lives.
        """
        if component.component_id in self._edges:
            raise ValueError(
                f"Component {component.component_id!r} is already in the graph."
            )
        for port_name, node_id in connections.items():
            if node_id not in self._nodes:
                raise KeyError(f"Node {node_id!r} does not exist in the graph.")
            component.port(port_name).node_id = node_id

        edge = CircuitEdge(component=component, port_nodes=dict(connections))
        self._edges[component.component_id] = edge
        return edge

    # ------------------------------------------------------------------
    # Topology mutations (used by actions)
    # ------------------------------------------------------------------

    def disconnect_port(self, component_id: str, port_name: str) -> str:
        """
        Disconnect one port of a component, making it float.

        Returns the node_id that the port was connected to.
        The port's ``node_id`` is set to None.
        A new isolated node is NOT created; the component simply loses
        its connection to the network on that port.
        """
        edge = self._get_edge(component_id)
        node_id = edge.port_nodes.pop(port_name)
        edge.component.port(port_name).node_id = None
        return node_id

    def reconnect_port(
        self,
        component_id: str,
        port_name: str,
        node_id: str,
    ) -> None:
        """Connect a previously disconnected port to a node."""
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id!r} does not exist.")
        edge = self._get_edge(component_id)
        edge.port_nodes[port_name] = node_id
        edge.component.port(port_name).node_id = node_id

    def remove_component(self, component_id: str) -> CircuitEdge:
        """
        Remove a component from the graph entirely (all ports disconnected).
        Returns the removed edge.
        """
        edge = self._get_edge(component_id)
        for port in edge.component.ports:
            port.node_id = None
        del self._edges[component_id]
        return edge

    def short_nodes(self, node_a: str, node_b: str, short_id: str, resistance: float = 1e-6) -> None:
        """
        Insert a resistor between two nodes.

        The default resistance (1 µΩ) acts as a near-zero wire.  Pass a
        higher value to model a deliberate bridging probe (e.g. 0.01 Ω).
        """
        from diagnosable_systems_simulation.world.components import Resistor

        r = Resistor(
            component_id=short_id,
            display_name=f"Short({node_a}↔{node_b})",
            resistance=resistance,
        )
        self.add_component(r, {"p": node_a, "n": node_b})

    def merge_nodes(self, keep_id: str, remove_id: str) -> None:
        """
        Merge two nodes by redirecting all edges that reference
        ``remove_id`` to ``keep_id``, then deleting ``remove_id``.
        """
        if keep_id not in self._nodes or remove_id not in self._nodes:
            raise KeyError("Both nodes must exist before merging.")
        for edge in self._edges.values():
            for port_name, nid in list(edge.port_nodes.items()):
                if nid == remove_id:
                    edge.port_nodes[port_name] = keep_id
                    edge.component.port(port_name).node_id = keep_id
        del self._nodes[remove_id]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_netlist(self) -> list[CircuitEdge]:
        return list(self._edges.values())

    def get_nodes(self) -> list[CircuitNode]:
        return list(self._nodes.values())

    def ground_node(self) -> Optional[CircuitNode]:
        for node in self._nodes.values():
            if node.is_ground:
                return node
        return None

    def has_component(self, component_id: str) -> bool:
        return component_id in self._edges

    def get_component(self, component_id: str) -> Component:
        return self._get_edge(component_id).component

    def nodes_of(self, component_id: str) -> dict[str, str]:
        """Return the port→node_id mapping for a component."""
        return dict(self._get_edge(component_id).port_nodes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_edge(self, component_id: str) -> CircuitEdge:
        try:
            return self._edges[component_id]
        except KeyError:
            raise KeyError(
                f"Component {component_id!r} is not in the graph."
            )

    def __repr__(self) -> str:
        return (
            f"CircuitGraph("
            f"nodes={len(self._nodes)}, "
            f"components={len(self._edges)})"
        )
