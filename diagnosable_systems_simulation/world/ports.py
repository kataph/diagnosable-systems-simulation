from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class PortRole(Enum):
    POSITIVE  = auto()  # higher-potential terminal
    NEGATIVE  = auto()  # lower-potential terminal
    ANODE     = auto()  # polarised device: anode
    CATHODE   = auto()  # polarised device: cathode
    COMMON    = auto()  # unpolarised / multi-terminal passives


@dataclass
class BoundPort:
    """A port reference that carries its owning component alongside the port name."""
    component: object   # Component (typed as object to avoid circular import)
    port_name: str


@dataclass
class ElectricalPort:
    """
    One terminal of a component.
    `node_id` is assigned when the component is wired into a CircuitGraph;
    it is None until then.
    """
    name: str               # e.g. "p", "n", "anode", "cathode"
    role: PortRole
    node_id: Optional[str] = None

    def is_connected(self) -> bool:
        return self.node_id is not None

    def __repr__(self) -> str:
        node = self.node_id or "unconnected"
        return f"Port({self.name}/{self.role.name} → {node})"
