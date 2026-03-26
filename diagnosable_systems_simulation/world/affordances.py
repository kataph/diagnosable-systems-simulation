from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from diagnosable_systems_simulation.world.context import WorldContext
    from diagnosable_systems_simulation.world.components import Component


class Affordance(Enum):
    OBSERVABLE    = auto()  # can be seen and described
    MEASURABLE    = auto()  # electrical meter can be placed on its ports
    DETACHABLE    = auto()  # can be physically disconnected from circuit
    RECONNECTABLE = auto()  # can be reconnected after detachment
    REPLACEABLE   = auto()  # can be swapped for a new unit
    ADJUSTABLE    = auto()  # a parameter can be changed (e.g. potentiometer wiper)
    MOVABLE       = auto()  # physical position can be changed
    OPENABLE      = auto()  # enclosure or access port can be opened
    CLOSEABLE     = auto()  # enclosure or access port can be closed
    TOGGLABLE     = auto()  # switch state can be flipped


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
        if affordance in self._static:
            return True
        if affordance in self._dynamic:
            return True
        return any(
            ca.affordance == affordance and ca.condition(component, context)
            for ca in self._conditional
        )

    def all_active(
        self,
        component: Component,
        context: WorldContext,
    ) -> set[Affordance]:
        active = set(self._static) | set(self._dynamic)
        for ca in self._conditional:
            if ca.condition(component, context):
                active.add(ca.affordance)
        return active

    def __repr__(self) -> str:
        return (
            f"AffordanceSet(static={set(self._static)}, "
            f"dynamic={self._dynamic}, "
            f"conditional_count={len(self._conditional)})"
        )
