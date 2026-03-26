from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

from diagnosable_systems_simulation.world.affordances import Affordance

if TYPE_CHECKING:
    from diagnosable_systems_simulation.world.components import Component
    from diagnosable_systems_simulation.world.context import WorldContext


# ---------------------------------------------------------------------------
# Requirement types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AffordanceRequirement:
    """Target component (by role name) must have this affordance active."""
    component_role: str
    affordance: Affordance

    def check(
        self,
        targets: dict[str, Component],
        context: WorldContext,
    ) -> tuple[bool, str]:
        comp = targets.get(self.component_role)
        if comp is None:
            return False, f"No target with role {self.component_role!r} provided."
        if not comp.affordances.has(self.affordance, comp, context):
            return (
                False,
                f"{comp.display_name!r} does not have affordance "
                f"{self.affordance.name}.",
            )
        return True, ""


@dataclass(frozen=True)
class ToolRequirement:
    """The agent must have this tool available in the world context."""
    tool_name: str

    def check(
        self,
        targets: dict[str, Component],
        context: WorldContext,
    ) -> tuple[bool, str]:
        if not context.has_tool(self.tool_name):
            return False, f"Tool {self.tool_name!r} is not in hand."
        return True, ""


@dataclass(frozen=True)
class ContextRequirement:
    """Arbitrary predicate on WorldContext."""
    description: str
    predicate: Callable[[WorldContext], bool]

    def check(
        self,
        targets: dict[str, Component],
        context: WorldContext,
    ) -> tuple[bool, str]:
        if not self.predicate(context):
            return False, f"Context requirement not met: {self.description}"
        return True, ""


# Type alias for the union
Requirement = AffordanceRequirement | ToolRequirement | ContextRequirement


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

class PreconditionChecker:
    """
    Evaluates a list of requirements and collects all failures.
    """

    @staticmethod
    def check_all(
        requirements: list[Requirement],
        targets: dict[str, Component],
        context: WorldContext,
    ) -> tuple[bool, list[str]]:
        """
        Returns ``(all_satisfied, list_of_failure_messages)``.
        All requirements are checked even if an earlier one fails,
        so the caller receives a complete picture.
        """
        failures: list[str] = []
        for req in requirements:
            ok, msg = req.check(targets, context)
            if not ok:
                failures.append(msg)
        return len(failures) == 0, failures
