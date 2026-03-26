from __future__ import annotations

import math
from typing import Any

from diagnosable_systems_simulation.electrical_simulation.backend.base import SimulationBackend
from diagnosable_systems_simulation.electrical_simulation.circuit import CircuitGraph, CircuitEdge
from diagnosable_systems_simulation.electrical_simulation.results import SimulationResult
from diagnosable_systems_simulation.world.components import (
    Bulb, Cable, Diode, Fuse, LED, LightSensor,
    Potentiometer, Resistor, Switch, VoltageSource,
)


class PySpiceBackend(SimulationBackend):
    """
    Simulation backend using PySpice + ngspice.

    Each component type is translated to its SPICE equivalent:

    ┌─────────────┬──────────────────────────────────────────────┐
    │ Component   │ SPICE representation                         │
    ├─────────────┼──────────────────────────────────────────────┤
    │ Resistor    │ R element                                    │
    │ Cable       │ R element (very small if resistance==0)      │
    │ Bulb        │ R element                                    │
    │ LightSensor │ R element (value from current_parameters)    │
    │ Fuse        │ R element (0 Ω if intact, 1e9 Ω if blown)   │
    │ Switch      │ R element (0 Ω if closed, 1e9 Ω if open)    │
    │ LED         │ D element (simple Shockley model)            │
    │ Diode       │ D element                                    │
    │ Potentio.   │ Two R elements sharing the wiper node        │
    │ VoltageSource│ V element                                   │
    └─────────────┴──────────────────────────────────────────────┘

    Requires: ``pip install PySpice`` and ngspice installed.
    """

    # Resistance used to represent an open switch / blown fuse
    _OPEN_RESISTANCE = 1e9   # 1 GΩ
    _WIRE_RESISTANCE = 1e-6  # 1 µΩ  (ideal wire)

    def supports_nonlinear(self) -> bool:
        return True

    def solve(self, graph: CircuitGraph) -> SimulationResult:
        try:
            from PySpice.Spice.Netlist import Circuit
            from PySpice.Spice.NgSpice.Shared import NgSpiceShared
            import PySpice.Unit as U
        except ImportError as exc:
            raise ImportError(
                "PySpice is not installed. "
                "Run: pip install PySpice  (and install ngspice separately)."
            ) from exc

        circuit = Circuit("diagnosable_system")
        ground_node = graph.ground_node()
        if ground_node is None:
            raise RuntimeError("Circuit has no ground node.")

        gnd_id = ground_node.node_id
        warnings_list: list[str] = []

        v_supply = max(
            (abs(e.component.current_parameters().get("voltage", 0.0))
             for e in graph.get_netlist()
             if isinstance(e.component, VoltageSource) and e.port_nodes),
            default=12.0,
        )

        # Translate every edge ------------------------------------------
        for edge in graph.get_netlist():
            self._add_element(circuit, edge, gnd_id, warnings_list, v_supply)

        # Run DC operating-point analysis --------------------------------
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        try:
            analysis = simulator.operating_point()
        except Exception as exc:
            return SimulationResult(converged=False, warnings=(str(exc),))

        # Parse results --------------------------------------------------
        node_voltages: dict[str, float] = {}
        for node in graph.get_nodes():
            if node.is_ground:
                node_voltages[node.node_id] = 0.0
            else:
                try:
                    node_voltages[node.node_id] = float(
                        analysis.nodes[node.node_id.lower()].item()
                    )
                except KeyError:
                    warnings_list.append(
                        f"Node {node.node_id!r} not found in analysis output."
                    )

        branch_currents: dict[str, float] = {}
        component_power: dict[str, float] = {}

        for edge in graph.get_netlist():
            cid = edge.component_id
            comp = edge.component
            params = comp.current_parameters()
            pnodes = edge.port_nodes

            def _v(port: str) -> float:
                node = pnodes.get(port)
                return node_voltages.get(node, 0.0) if node else 0.0

            if isinstance(comp, (LED, Diode)):
                v_across = _v("anode") - _v("cathode")
            elif isinstance(comp, VoltageSource):
                v_across = _v("pos") - _v("neg")
            else:
                v_across = _v("p") - _v("n")

            if isinstance(comp, (Resistor, Cable, Bulb, LightSensor)):
                r = params.get("resistance", 1.0) or 1.0
                i = v_across / r
            elif isinstance(comp, Fuse):
                r = self._OPEN_RESISTANCE if params.get("is_blown") else self._WIRE_RESISTANCE
                i = v_across / r
            elif isinstance(comp, Switch):
                r = self._WIRE_RESISTANCE if params.get("is_closed") else self._OPEN_RESISTANCE
                i = v_across / r
            elif isinstance(comp, Potentiometer):
                r = params.get("total_resistance", 1.0) or 1.0
                i = v_across / r
            elif isinstance(comp, LED):
                # Derive current from the synthetic mid-node inserted by _add_element
                r_series = 1.0
                try:
                    v_mid = float(analysis.nodes[f"_led_mid_{cid}".lower()].item())
                except (KeyError, Exception):
                    v_mid = _v("cathode")
                i = (v_mid - _v("cathode")) / r_series
            elif isinstance(comp, Diode):
                # Reconstruct current from Shockley equation using solved node voltages
                VT = 0.02585  # thermal voltage at 25 °C
                i = 1e-14 * (math.exp(min(v_across / VT, 500)) - 1) if v_across > 0 else 0.0
            elif isinstance(comp, VoltageSource):
                try:
                    i = -float(analysis.branches[f"v{cid.lower()}"].item())
                except (KeyError, Exception):
                    i = 0.0
            else:
                i = 0.0

            branch_currents[cid] = i
            component_power[cid] = abs(v_across * i)

        # Determine light emitters ----------------------------------------
        emitting: set[str] = set()
        for edge in graph.get_netlist():
            comp = edge.component
            pwr = component_power.get(comp.component_id, 0.0)
            if isinstance(comp, (Bulb, LED)):
                threshold = comp.current_parameters().get("power_threshold", 0.01)
                if pwr >= threshold:
                    emitting.add(comp.component_id)

        return SimulationResult(
            node_voltages=node_voltages,
            branch_currents=branch_currents,
            component_power=component_power,
            emitting_light=frozenset(emitting),
            converged=True,
            warnings=tuple(warnings_list),
        )

    # ------------------------------------------------------------------
    # Internal translation helpers
    # ------------------------------------------------------------------

    def _node_name(self, node_id: str, gnd_id: str) -> str:
        """PySpice uses '0' for ground."""
        return "0" if node_id == gnd_id else node_id

    def _add_element(
        self,
        circuit: Any,
        edge: CircuitEdge,
        gnd_id: str,
        warnings_list: list[str],
        v_supply: float = 12.0,
    ) -> None:
        comp = edge.component
        params = comp.current_parameters()
        pnodes = edge.port_nodes
        nn = lambda nid: self._node_name(nid, gnd_id)  # noqa: E731
        cid = comp.component_id

        if not pnodes:
            return  # fully disconnected — nothing to stamp

        try:
            self._add_element_inner(circuit, comp, params, pnodes, nn, cid,
                                    warnings_list, v_supply)
        except KeyError as exc:
            warnings_list.append(
                f"Component {cid!r} has disconnected port {exc}; skipped."
            )

    def _add_element_inner(
        self,
        circuit: Any,
        comp: Any,
        params: dict,
        pnodes: dict,
        nn: Any,
        cid: str,
        warnings_list: list[str],
        v_supply: float,
    ) -> None:
        if isinstance(comp, (Resistor, Cable, LightSensor)):
            r = params.get("resistance", self._WIRE_RESISTANCE)
            if r == 0.0:
                r = self._WIRE_RESISTANCE
            circuit.R(cid, nn(pnodes["p"]), nn(pnodes["n"]), r)

        elif isinstance(comp, Bulb):
            r = params.get("resistance", 120.0) or 120.0
            circuit.R(cid, nn(pnodes["p"]), nn(pnodes["n"]), r)

        elif isinstance(comp, Fuse):
            r = self._OPEN_RESISTANCE if params.get("is_blown") else self._WIRE_RESISTANCE
            circuit.R(cid, nn(pnodes["p"]), nn(pnodes["n"]), r)

        elif isinstance(comp, Switch):
            r = self._WIRE_RESISTANCE if params.get("is_closed") else self._OPEN_RESISTANCE
            circuit.R(cid, nn(pnodes["p"]), nn(pnodes["n"]), r)

        elif isinstance(comp, LED):
            # Model as diode + tiny series resistor (pure-diode behaviour).
            # Current limiting is expected to come from an external resistor in
            # the circuit; the internal 1 Ω stub merely satisfies SPICE syntax.
            vf = params.get("forward_voltage", 2.0)
            r_series = 1.0
            model_name = f"LED_{cid}"
            circuit.model(model_name, "D", IS=1e-14, N=1.8, VJ=vf)
            mid_node = f"_led_mid_{cid}"
            circuit.D(cid, nn(pnodes["anode"]), mid_node, model=model_name)
            circuit.R(f"_rled_{cid}", mid_node, nn(pnodes["cathode"]), r_series)

        elif isinstance(comp, Diode):
            vf = params.get("forward_voltage", 0.7)
            model_name = f"D_{cid}"
            circuit.model(model_name, "D", IS=1e-14, N=1.0, VJ=vf)
            circuit.D(cid, nn(pnodes["anode"]), nn(pnodes["cathode"]), model=model_name)

        elif isinstance(comp, Potentiometer):
            total_r = params.get("total_resistance", 1000.0) or 1000.0
            wiper = params.get("wiper_position", 0.5)
            r_upper = total_r * wiper
            r_lower = total_r * (1.0 - wiper)
            wiper_node = pnodes.get("wiper", f"_wiper_{cid}")
            circuit.R(f"{cid}_upper", nn(pnodes["p"]), nn(wiper_node), max(r_upper, self._WIRE_RESISTANCE))
            circuit.R(f"{cid}_lower", nn(wiper_node), nn(pnodes["n"]), max(r_lower, self._WIRE_RESISTANCE))

        elif isinstance(comp, VoltageSource):
            v = params.get("voltage", 0.0)
            circuit.V(cid, nn(pnodes["pos"]), nn(pnodes["neg"]), v)

        else:
            warnings_list.append(
                f"Component {cid!r} of type {type(comp).__name__!r} "
                f"has no SPICE translation; skipped."
            )
