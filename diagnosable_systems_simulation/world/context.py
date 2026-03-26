from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorldContext:
    """
    Global mutable state of the physical world, independent of the
    electrical circuit.

    Affordance conditions and action precondition/effect checkers read
    and write this object. It is intentionally a plain dataclass with no
    hidden behaviour: all intelligence lives in the code that uses it.

    Fields
    ------
    tools_in_hand
        Set of tool identifiers the agent currently has available.
        Examples: ``{"multimeter", "screwdriver", "replacement_fuse_5A"}``.
        Actions may require specific tools as preconditions.

    extra
        Escape hatch for system-specific world state that does not fit
        the standard fields (e.g. ambient light level, room temperature).
        Keys should be namespaced to avoid collisions across systems.
    """

    tools_in_hand: set[str] = field(default_factory=set)
    extra: dict[str, Any] = field(default_factory=dict)

    def has_tool(self, tool: str) -> bool:
        return tool in self.tools_in_hand
