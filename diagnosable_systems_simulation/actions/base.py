from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from diagnosable_systems_simulation.actions.observation import ObservationRecord
    from diagnosable_systems_simulation.electrical_simulation.circuit import CircuitGraph
    from diagnosable_systems_simulation.electrical_simulation.results import SimulationResult
    from diagnosable_systems_simulation.world.components import Component
    from diagnosable_systems_simulation.world.context import WorldContext


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

@dataclass
class ActionCost:
    """
    Multidimensional cost of performing an action.

    time
        Estimated duration in seconds.

    equipment
        Tools or instruments required (must be present in
        ``WorldContext.tools_in_hand`` as a precondition if the action
        enforces it, or simply listed here for planning purposes).

    resources_consumed
        Consumable resources used up by the action.
        Keys are resource identifiers, values are quantities.
        Examples::

            {"replacement_fuse_5A": 1}
            {"solder_grams": 2.0, "heat_shrink_cm": 5.0}
    """
    time: float = 0.0
    equipment: list[str] = field(default_factory=list)
    resources_consumed: dict[str, float] = field(default_factory=dict)

    def __add__(self, other: ActionCost) -> ActionCost:
        combined_resources = dict(self.resources_consumed)
        for k, v in other.resources_consumed.items():
            combined_resources[k] = combined_resources.get(k, 0.0) + v
        return ActionCost(
            time=self.time + other.time,
            equipment=list(set(self.equipment) | set(other.equipment)),
            resources_consumed=combined_resources,
        )


# ---------------------------------------------------------------------------
# Action result
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    """
    Returned by ``Action.execute()`` and ``DiagnosableSystem.apply_action()``.

    success
        True when the action executed successfully; False only when
        preconditions were not met (set by the system, not by execute()).

    observation
        Structured observation record for diagnostic actions; None for
        pure structural mutations.

    message
        Human-readable description of what happened or why it failed.
    """
    success: bool = True
    observation: Optional[ObservationRecord] = None
    message: str = ""


# ---------------------------------------------------------------------------
# Action base class
# ---------------------------------------------------------------------------

class Action(ABC):
    """
    Abstract base for all fault-injection and diagnostic actions.

    Subclasses declare:
    - ``action_id``: unique string identifier.
    - ``description``: one-line human-readable description.
    - ``cost``: nominal ``ActionCost``.
    - ``mutates_graph``: True if execute() changes circuit topology or
      component parameters, requiring re-simulation.  Defaults to False.

    And implement:
    - ``check_preconditions()``: returns ``(ok, reason)`` without side effects.
    - ``execute()``: performs the action, returns ``ActionResult``.
      execute() is only called after preconditions pass; it must not
      return success=False.
    """

    action_id: str
    description: str
    cost: ActionCost
    mutates_graph: bool = False

    @abstractmethod
    def check_preconditions(
        self,
        targets: dict[str, Component],
        context: WorldContext,
    ) -> tuple[bool, str]:
        """
        Verify that all preconditions are met.

        ``targets`` maps role names (e.g. ``"subject"``, ``"tool"``) to
        component instances.

        Returns ``(True, "")`` if all preconditions are satisfied, or
        ``(False, <reason>)`` otherwise.  Must have no side effects.
        """
        ...

    @abstractmethod
    def execute(
        self,
        targets: dict[str, Component],
        graph: CircuitGraph,
        context: WorldContext,
        last_result: Optional[SimulationResult],
    ) -> ActionResult:
        """
        Perform the action.  May mutate ``graph``, ``context``, and/or
        component state.

        ``last_result`` is the most recent simulation result; it may be
        ``None`` if no simulation has been run yet.
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.action_id!r})"
