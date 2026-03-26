from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Position:
    """3-D position in metres."""
    x: float
    y: float
    z: float

    def distance_to(self, other: Position) -> float:
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def is_within(self, other: Position, radius: float) -> bool:
        return self.distance_to(other) <= radius

    def __repr__(self) -> str:
        return f"Position({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


@dataclass
class Enclosure:
    """
    Represents a physical container (e.g. a wooden cube).

    `contained_ids` lists the component_ids of components physically
    inside this enclosure. Enclosures can themselves be nested.

    `is_open` and `is_inverted` are world-state flags that actions
    set/clear; affordance conditions can key off them.
    """
    enclosure_id: str
    contained_ids: list[str] = field(default_factory=list)
    is_open: bool = False
    is_inverted: bool = False
    position: Optional[Position] = None

    def add(self, component_id: str) -> None:
        if component_id not in self.contained_ids:
            self.contained_ids.append(component_id)

    def remove(self, component_id: str) -> None:
        self.contained_ids = [c for c in self.contained_ids if c != component_id]

    def contains(self, component_id: str) -> bool:
        return component_id in self.contained_ids

    def __repr__(self) -> str:
        return (
            f"Enclosure({self.enclosure_id!r}, "
            f"open={self.is_open}, inverted={self.is_inverted}, "
            f"contains={self.contained_ids})"
        )
