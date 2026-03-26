from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SimulationResult:
    """
    Output of a single simulation run.

    All quantities use SI base units (V, A, W) unless noted otherwise.

    node_voltages
        Maps node_id -> voltage in Volts relative to ground.

    branch_currents
        Maps component_id -> conventional current through the device in Amps.
        For a two-terminal device this is the current entering the positive port.

    component_power
        Maps component_id -> power dissipated (or generated) in Watts.
        Positive = power consumed.

    emitting_light
        Set of component_ids (Bulb / LED) that are currently emitting light,
        determined by the solver after comparing ``component_power`` against
        each component's ``power_threshold``.

    converged
        True if the solver reached a stable solution within its tolerance.

    warnings
        Non-fatal messages from the backend or solver (e.g. iteration limit
        reached, open-circuit nodes, etc.).
    """

    node_voltages: dict[str, float] = field(default_factory=dict)
    branch_currents: dict[str, float] = field(default_factory=dict)
    component_power: dict[str, float] = field(default_factory=dict)
    emitting_light: frozenset[str] = field(default_factory=frozenset)
    converged: bool = True
    warnings: tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def voltage(self, node_id: str) -> Optional[float]:
        return self.node_voltages.get(node_id)

    def current(self, component_id: str) -> Optional[float]:
        return self.branch_currents.get(component_id)

    def power(self, component_id: str) -> Optional[float]:
        return self.component_power.get(component_id)

    def is_lit(self, component_id: str) -> bool:
        return component_id in self.emitting_light

    def voltage_across(
        self,
        node_pos: str,
        node_neg: str,
    ) -> Optional[float]:
        vp = self.node_voltages.get(node_pos)
        vn = self.node_voltages.get(node_neg)
        if vp is None or vn is None:
            return None
        return vp - vn

    def __repr__(self) -> str:
        lit = sorted(self.emitting_light)
        return (
            f"SimulationResult("
            f"nodes={len(self.node_voltages)}, "
            f"converged={self.converged}, "
            f"lit={lit})"
        )
