from __future__ import annotations

from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional

from diagnosable_systems_simulation.electrical_simulation.circuit import CircuitGraph
from diagnosable_systems_simulation.electrical_simulation.results import SimulationResult


class SimulationBackend(ABC):
    """
    Abstract interface between the simulation runner and a concrete solver.

    Implementations translate a ``CircuitGraph`` into backend-specific
    representation, run the solver, and return a ``SimulationResult``.

    The backend must NOT know about affordances, world context, actions,
    or physical coupling — those live in ``SimulationRunner``.
    """

    @abstractmethod
    def solve(self, graph: CircuitGraph, logger: Optional[Logger] = None) -> SimulationResult:
        """
        Perform a DC operating-point analysis of the circuit described
        by ``graph`` and return the result.
        """
        ...

    @abstractmethod
    def supports_nonlinear(self) -> bool:
        """
        Return True if the backend can handle nonlinear devices
        (diodes, LEDs, transistors).
        """
        ...

    def solve_continuity(
        self,
        graph: CircuitGraph,
        node_a: str,
        node_b: str,
        logger: Optional[Logger] = None,
    ) -> float:
        """
        Return the Thevenin resistance (Ω) seen between *node_a* and *node_b*
        with all independent sources zeroed (de-energised circuit, as a real
        ohmmeter measures).

        Implemented by injecting a 1 mV test voltage source between the two
        nodes and returning R = V_test / I_test from the DC operating point.

        Returns 1e9 (1 GΩ) when the path is open or the solve fails.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement solve_continuity."
        )

    def name(self) -> str:
        return type(self).__name__
