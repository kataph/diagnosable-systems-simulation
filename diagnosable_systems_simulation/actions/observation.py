from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from diagnosable_systems_simulation.world.affordances import Affordance

if TYPE_CHECKING:
    from diagnosable_systems_simulation.electrical_simulation.results import SimulationResult
    from diagnosable_systems_simulation.world.components import Component
    from diagnosable_systems_simulation.world.context import WorldContext


@dataclass
class ObservableProperty:
    """A single named property that was observed."""
    name: str
    value: Any
    unit: Optional[str] = None

    def __repr__(self) -> str:
        u = f" {self.unit}" if self.unit else ""
        return f"{self.name}={self.value!r}{u}"


@dataclass
class ObservationRecord:
    """
    Structured output of a diagnostic observation or measurement action.

    Only properties gated by currently active affordances are included —
    this enforces the rule that the agent cannot know things it hasn't
    earned the right to know.

    ``simulation_snapshot`` holds the full ``SimulationResult`` at the
    moment of observation; downstream NLG code can use it to describe
    node voltages, currents, and LED states in natural language.
    """
    component_id: str
    action_id: str
    timestamp: float = field(default_factory=_time.time)
    properties: list[ObservableProperty] = field(default_factory=list)
    simulation_snapshot: Optional[SimulationResult] = None

    def add(self, name: str, value: Any, unit: Optional[str] = None) -> None:
        self.properties.append(ObservableProperty(name, value, unit))

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "action_id": self.action_id,
            "timestamp": self.timestamp,
            "properties": [
                {"name": p.name, "value": p.value, "unit": p.unit}
                for p in self.properties
            ],
        }

    def __repr__(self) -> str:
        props = ", ".join(repr(p) for p in self.properties)
        return f"ObservationRecord({self.component_id!r}, [{props}])"


# ---------------------------------------------------------------------------
# Helper: build observation records gated by affordances
# ---------------------------------------------------------------------------

def observe_component(
    component: Component,
    context: WorldContext,
    action_id: str,
    result: Optional[SimulationResult] = None,
) -> ObservationRecord:
    """
    Create an ``ObservationRecord`` for ``component`` containing only
    the properties accessible under current affordances.

    What is included:
    - If OBSERVABLE: visual/state properties (switch position, LED color,
      component type, fault presence).
    - If MEASURABLE and a result is provided: port voltages and branch
      current from the simulation result.
    """
    record = ObservationRecord(
        component_id=component.component_id,
        action_id=action_id,
        simulation_snapshot=result,
    )

    active = component.affordances.all_active(component, context)

    if Affordance.OBSERVABLE in active:
        record.add("display_name", component.display_name)
        record.add("type", type(component).__name__)
        record.add("has_fault", component.has_fault())

        # Type-specific visual properties
        from diagnosable_systems_simulation.world.components import (
            Switch, LED, Bulb, Fuse, Potentiometer
        )
        if isinstance(component, Switch):
            record.add("is_closed", component.is_closed)
        if isinstance(component, LED):
            record.add("color", component.color)
            if result is not None:
                record.add("is_lit", result.is_lit(component.component_id))
        if isinstance(component, Bulb):
            if result is not None:
                record.add("is_lit", result.is_lit(component.component_id))
        if isinstance(component, Fuse):
            record.add("is_blown", component.is_blown)
        if isinstance(component, Potentiometer):
            record.add("wiper_position", component.wiper_position)

    if Affordance.MEASURABLE in active and result is not None:
        for port in component.ports:
            if port.node_id is not None:
                v = result.voltage(port.node_id)
                if v is not None:
                    record.add(f"voltage_{port.name}", round(v, 4), "V")
        i = result.current(component.component_id)
        if i is not None:
            record.add("current", round(i, 6), "A")
        p = result.power(component.component_id)
        if p is not None:
            record.add("power", round(p, 6), "W")

    return record
