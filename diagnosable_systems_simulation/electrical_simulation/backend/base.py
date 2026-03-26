from __future__ import annotations

from abc import ABC, abstractmethod

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
    def solve(self, graph: CircuitGraph) -> SimulationResult:
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

    def name(self) -> str:
        return type(self).__name__
