from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from diagnosable_systems_simulation.world.context import WorldContext
    from diagnosable_systems_simulation.world.components import Component


class Affordance(Enum):
    OBSERVABLE    = auto()  # can be seen and described
    REACHABLE     = auto()  # can be physically touched/probed (implies OBSERVABLE)
    MEASURABLE    = auto()  # electrical meter can be placed on its ports (requires REACHABLE)
    DETACHABLE    = auto()  # can be physically disconnected from circuit (requires REACHABLE)
    RECONNECTABLE = auto()  # can be reconnected after detachment (requires REACHABLE)
    REPLACEABLE   = auto()  # can be swapped for a new unit (requires REACHABLE)
    ADJUSTABLE    = auto()  # a parameter can be changed (e.g. potentiometer wiper) (requires REACHABLE)
    MOVABLE       = auto()  # physical position can be changed (requires REACHABLE)
    OPENABLE      = auto()  # enclosure or access port can be opened (requires REACHABLE)
    CLOSEABLE     = auto()  # enclosure or access port can be closed (requires REACHABLE)
    TOGGLABLE     = auto()  # switch state can be flipped (requires REACHABLE)


@dataclass(frozen=True)
class ConditionalAffordance:
    """
    An affordance that is active only when a predicate holds.
    The predicate receives both the component and the world context,
    so it can key off either or both.
    """
    affordance: Affordance
    condition: Callable[[Component, WorldContext], bool]
    description: str = ""  # human-readable summary of the condition


class AffordanceSet:
    """
    Tracks which affordances are active for a single component.

    Three tiers:
    - static:      always present, set at construction time.
    - dynamic:     toggled at runtime by action effects.
    - conditional: evaluated on demand; depend on component + world context.
    """

    def __init__(
        self,
        static: set[Affordance] | None = None,
        conditional: list[ConditionalAffordance] | None = None,
    ):
        self._static: frozenset[Affordance] = frozenset(static or [])
        self._dynamic: set[Affordance] = set()
        self._conditional: list[ConditionalAffordance] = list(conditional or [])

    # ------------------------------------------------------------------
    # Mutation (dynamic tier only — static and conditional are immutable)
    # ------------------------------------------------------------------

    def add(self, affordance: Affordance) -> None:
        self._dynamic.add(affordance)

    def remove(self, affordance: Affordance) -> None:
        self._dynamic.discard(affordance)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def has(
        self,
        affordance: Affordance,
        component: Component,
        context: WorldContext,
    ) -> bool:
        return affordance in self.all_active(component, context)

    def all_active(
        self,
        component: Component,
        context: WorldContext,
    ) -> set[Affordance]:
        active = set(self._static) | set(self._dynamic)
        for ca in self._conditional:
            if ca.condition(component, context):
                active.add(ca.affordance)
        # REACHABLE implies OBSERVABLE: if you can touch it, you can see it.
        if Affordance.REACHABLE in active:
            active.add(Affordance.OBSERVABLE)
        return active

    def __repr__(self) -> str:
        return (
            f"AffordanceSet(static={set(self._static)}, "
            f"dynamic={self._dynamic}, "
            f"conditional_count={len(self._conditional)})"
        )
