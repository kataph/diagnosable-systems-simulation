from __future__ import annotations

from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional

from diagnosable_systems_simulation.electrical_simulation.backend.base import SimulationBackend
from diagnosable_systems_simulation.electrical_simulation.circuit import CircuitGraph
from diagnosable_systems_simulation.electrical_simulation.results import SimulationResult
from diagnosable_systems_simulation.world.components import LightSensor
from diagnosable_systems_simulation.world.context import WorldContext
from diagnosable_systems_simulation.world.spatial import Position


# ---------------------------------------------------------------------------
# Physical coupling abstraction
# ---------------------------------------------------------------------------

class PhysicalCoupling(ABC):
    """
    Describes a physical interaction between the simulation result and
    circuit parameters that may require re-simulation.

    The coupling loop in ``SimulationRunner`` calls ``apply()`` after each
    solve. If any coupling modifies the graph it returns ``True`` and the
    runner re-simulates. The loop stops when no coupling modifies anything
    or the iteration limit is reached.

    Subclasses must be stateless with respect to the graph — the same
    coupling object can be applied to different graph states.
    """

    @abstractmethod
    def apply(
        self,
        result: SimulationResult,
        graph: CircuitGraph,
        context: WorldContext,
    ) -> bool:
        """
        Inspect ``result`` and optionally mutate ``graph`` (component
        parameters only — not topology) and/or ``context``.

        Returns True if anything was changed (triggering re-simulation).
        """
        ...


class LampToLightSensorCoupling(PhysicalCoupling):
    """
    If a lamp / LED (``emitter_id``) is lit AND a light sensor
    (``sensor_id``) is within ``coupling_radius`` metres of the emitter,
    set the sensor's resistance to its lit value.  Otherwise use the
    dark value.

    The coupling also respects physical barriers modelled via
    ``WorldContext.extra["light_barriers"]`` — a set of component_ids
    whose enclosures block light when closed.
    """

    def __init__(
        self,
        emitter_id: str,
        sensor_id: str,
        emitter_position: Position,
        sensor_position: Position,
        coupling_radius: float,
        barrier_enclosure_ids: list[str] | None = None,
    ):
        self.emitter_id = emitter_id
        self.sensor_id = sensor_id
        self.emitter_position = emitter_position
        self.sensor_position = sensor_position
        self.coupling_radius = coupling_radius
        self.barrier_enclosure_ids: list[str] = barrier_enclosure_ids or []

    def _is_blocked(self, context: WorldContext) -> bool:
        """Return True if a physical barrier prevents light transfer."""
        enclosures = context.extra.get("enclosures", {})
        for eid in self.barrier_enclosure_ids:
            enc = enclosures.get(eid)
            if enc is not None and not enc.is_open and not enc.is_inverted:
                return True
        return False

    def apply(
        self,
        result: SimulationResult,
        graph: CircuitGraph,
        context: WorldContext,
    ) -> bool:
        if not graph.has_component(self.sensor_id):
            return False

        sensor: LightSensor = graph.get_component(self.sensor_id)  # type: ignore[assignment]
        if not isinstance(sensor, LightSensor):
            return False

        in_range = self.emitter_position.is_within(
            self.sensor_position, self.coupling_radius
        )
        emitter_lit = result.is_lit(self.emitter_id)
        blocked = self._is_blocked(context)

        should_be_lit = emitter_lit and in_range and not blocked

        old_resistance = sensor._current_resistance
        sensor.set_illuminated(should_be_lit)
        return sensor._current_resistance != old_resistance


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

class SimulationRunner:
    """
    Orchestrates the solve → couple → re-solve loop.

    Steps:
    1. Run the backend solver on the current graph.
    2. Apply each registered ``PhysicalCoupling`` in order.
    3. If any coupling modified circuit parameters, re-run from step 1.
    4. Stop when stable (no coupling made changes) or ``MAX_ITERATIONS``
       is reached.

    The runner does NOT modify circuit topology — only component
    parameters are updated by couplings.
    """

    MAX_ITERATIONS: int = 10

    def __init__(
        self,
        backend: SimulationBackend,
        couplings: list[PhysicalCoupling] | None = None,
        logger: Logger | None = None,
    ):
        self.backend = backend
        self.couplings: list[PhysicalCoupling] = list(couplings or [])
        self.logger = logger

    def run(
        self,
        graph: CircuitGraph,
        context: WorldContext,
    ) -> SimulationResult:
        result: Optional[SimulationResult] = None

        for iteration in range(self.MAX_ITERATIONS):
            result = self.backend.solve(graph, self.logger)

            if not result.converged:
                # Propagate failure immediately; no point iterating.
                return result

            # Apply all couplings; check if any changed parameters.
            any_changed = False
            for coupling in self.couplings:
                changed = coupling.apply(result, graph, context)
                any_changed = any_changed or changed

            if not any_changed:
                return result

            # At least one coupling changed something → re-simulate.
            # (result will be replaced in the next iteration)

        # Reached iteration limit without convergence of coupling loop.
        # Run one final SPICE solve so that the returned node voltages are
        # consistent with the current component state (which was last updated
        # by the coupling *after* the previous solve).  Without this extra
        # solve the voltages belong to a different oscillation phase than the
        # component flags, producing paradoxical measurements (e.g. relay
        # reports is_closed=True while node voltage shows only leakage current).
        assert result is not None
        final_result = self.backend.solve(graph, self.logger)
        base = final_result if final_result.converged else result
        warnings = base.warnings + (
            f"Physical coupling loop did not stabilise after "
            f"{self.MAX_ITERATIONS} iterations.",
        )
        return SimulationResult(
            node_voltages=base.node_voltages,
            branch_currents=base.branch_currents,
            component_power=base.component_power,
            emitting_light=base.emitting_light,
            converged=False,
            warnings=warnings,
        )
