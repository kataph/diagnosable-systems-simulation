"""
Stub simulation backend for testing without ngspice.

Performs a DC analysis using Modified Nodal Analysis (MNA) with a
piecewise-linear diode model:
  - Pass 1: all diodes/LEDs treated as open circuits.
  - Pass 2: diodes/LEDs forward-biased in Pass 1 are stamped as low-R;
            reverse-biased remain open.

Handles: VoltageSource, Resistor, Cable, Bulb, LightSensor, Fuse, Switch,
         LED, Diode, Potentiometer.

Self-contained — only numpy required (no ngspice).
"""
from __future__ import annotations

import numpy as np

from diagnosable_systems_simulation.electrical_simulation.backend.base import SimulationBackend
from diagnosable_systems_simulation.electrical_simulation.circuit import CircuitGraph
from diagnosable_systems_simulation.electrical_simulation.results import SimulationResult
from diagnosable_systems_simulation.world.components import (
    Bulb, Cable, Diode, Fuse, LED, LightSensor,
    Potentiometer, Resistor, Switch, VoltageSource,
)

_OPEN = 1e9   # GΩ — open circuit / reverse-biased diode
_WIRE = 1e-2  # 10 mΩ — ideal wire


class StubBackend(SimulationBackend):
    """MNA DC stub with piecewise-linear diode model."""

    def supports_nonlinear(self) -> bool:
        return False  # linearised model

    def solve_continuity(self, graph: CircuitGraph, node_a: str, node_b: str, logger=None) -> float:
        """
        MNA-based Thevenin resistance between *node_a* and *node_b*.

        All independent voltage sources are zeroed (V=0, i.e. stamped as
        ideal shorts). A 1 mV test voltage source is stamped between the
        two target nodes. Returns R = 0.001 / |I_test|, or 1e9 if open.
        """
        ground = graph.ground_node()
        if ground is None or node_a is None or node_b is None:
            return 1e9

        gnd_id = ground.node_id
        non_gnd = [n for n in graph.get_nodes() if not n.is_ground]
        n_nodes = len(non_gnd)
        node_idx = {n.node_id: i for i, n in enumerate(non_gnd)}

        real_v_sources = [
            e for e in graph.get_netlist()
            if isinstance(e.component, VoltageSource) and e.port_nodes
        ]
        # One extra row for the test voltage source
        vsrc_idx = {e.component_id: n_nodes + i for i, e in enumerate(real_v_sources)}
        test_row = n_nodes + len(real_v_sources)
        size = n_nodes + len(real_v_sources) + 1

        G = np.zeros((size, size))
        b = np.zeros(size)

        def nidx(nid: str):
            return None if nid == gnd_id else node_idx.get(nid)

        def stamp_r(na: str, nb: str, conductance: float) -> None:
            ia, ib = nidx(na), nidx(nb)
            if ia is not None:
                G[ia, ia] += conductance
            if ib is not None:
                G[ib, ib] += conductance
            if ia is not None and ib is not None:
                G[ia, ib] -= conductance
                G[ib, ia] -= conductance

        def stamp_v(row: int, np_: str, nn_: str, v: float) -> None:
            ip, in_ = nidx(np_), nidx(nn_)
            if ip is not None:
                G[row, ip] = 1.0
                G[ip, row] = 1.0
            if in_ is not None:
                G[row, in_] = -1.0
                G[in_, row] = -1.0
            b[row] = v

        for edge in graph.get_netlist():
            comp = edge.component
            params = comp.current_parameters()
            pn = edge.port_nodes
            if not pn:
                continue
            try:
                if isinstance(comp, VoltageSource):
                    # Zero the source
                    row = vsrc_idx[comp.component_id]
                    stamp_v(row, pn["pos"], pn["neg"], 0.0)
                elif isinstance(comp, (Resistor, Cable, Bulb, LightSensor)):
                    r = params.get("resistance", _WIRE) or _WIRE
                    stamp_r(pn["p"], pn["n"], 1.0 / r)
                elif isinstance(comp, Fuse):
                    r = _OPEN if params.get("is_blown") else _WIRE
                    stamp_r(pn["p"], pn["n"], 1.0 / r)
                elif isinstance(comp, Switch):
                    r = params.get("resistance", _WIRE if params.get("is_closed") else _OPEN)
                    stamp_r(pn["p"], pn["n"], 1.0 / r)
                elif isinstance(comp, (LED, Diode)):
                    # Treat as open — no forward bias without supply
                    stamp_r(pn.get("anode", ""), pn.get("cathode", ""), 1.0 / _OPEN)
                elif isinstance(comp, Potentiometer):
                    total_r = params.get("total_resistance", 1000.0) or 1000.0
                    wiper = params.get("wiper_position", 0.5)
                    wiper_node = pn.get("wiper", f"_w_{comp.component_id}")
                    stamp_r(pn["p"], wiper_node, 1.0 / max(total_r * wiper, _WIRE))
                    stamp_r(wiper_node, pn["n"], 1.0 / max(total_r * (1 - wiper), _WIRE))
            except KeyError:
                pass  # disconnected port — skip

        # Stamp 1 mV test source between node_a and node_b
        test_v = 0.001
        stamp_v(test_row, node_a, node_b, test_v)

        try:
            x = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            return 1e9

        i_test = abs(float(x[test_row]))
        return (test_v / i_test) if i_test > 1e-15 else 1e9

    def solve(self, graph: CircuitGraph, logger=None) -> SimulationResult:
        ground = graph.ground_node()
        if ground is None:
            return SimulationResult(converged=False, warnings=("No ground node.",))

        non_gnd = [n for n in graph.get_nodes() if not n.is_ground]
        n_nodes = len(non_gnd)
        node_idx = {n.node_id: i for i, n in enumerate(non_gnd)}

        v_sources = [
            e for e in graph.get_netlist()
            if isinstance(e.component, VoltageSource) and e.port_nodes
        ]
        n_vsrc = len(v_sources)
        vsrc_idx = {e.component_id: n_nodes + i for i, e in enumerate(v_sources)}
        size = n_nodes + n_vsrc

        v_supply = max(
            (abs(e.component.current_parameters().get("voltage", 0.0))
             for e in v_sources),
            default=12.0,
        )

        warnings: list[str] = []

        def solve_pass(diode_states: dict[str, bool]) -> np.ndarray | None:
            """
            diode_states maps component_id -> True (forward) / False (reverse).
            Returns solution vector x, or None if singular.
            """
            G = np.zeros((size, size))
            b = np.zeros(size)

            def nidx(node_id: str) -> int | None:
                return None if node_id == ground.node_id else node_idx.get(node_id)

            def stamp_r(na: str, nb: str, conductance: float) -> None:
                ia, ib = nidx(na), nidx(nb)
                if ia is not None:
                    G[ia, ia] += conductance
                if ib is not None:
                    G[ib, ib] += conductance
                if ia is not None and ib is not None:
                    G[ia, ib] -= conductance
                    G[ib, ia] -= conductance

            def stamp_v(row: int, np_: str, nn_: str, v: float) -> None:
                ip, in_ = nidx(np_), nidx(nn_)
                if ip is not None:
                    G[row, ip] = 1.0
                    G[ip, row] = 1.0
                if in_ is not None:
                    G[row, in_] = -1.0
                    G[in_, row] = -1.0
                b[row] = v

            for edge in graph.get_netlist():
                comp = edge.component
                params = comp.current_parameters()
                pn = edge.port_nodes
                if not pn:
                    continue  # fully disconnected

                try:
                    if isinstance(comp, VoltageSource):
                        row = vsrc_idx[comp.component_id]
                        stamp_v(row, pn["pos"], pn["neg"], params["voltage"])

                    elif isinstance(comp, (Resistor, Cable, Bulb, LightSensor)):
                        r = params.get("resistance", _WIRE) or _WIRE
                        stamp_r(pn["p"], pn["n"], 1.0 / r)

                    elif isinstance(comp, Fuse):
                        r = _OPEN if params.get("is_blown") else _WIRE
                        stamp_r(pn["p"], pn["n"], 1.0 / r)

                    elif isinstance(comp, Switch):
                        r = params.get("resistance", _WIRE if params.get("is_closed") else _OPEN)
                        stamp_r(pn["p"], pn["n"], 1.0 / r)

                    elif isinstance(comp, LED):
                        fwd = diode_states.get(comp.component_id, False)
                        if fwd:
                            # Model LED as resistance = vf/ifwd so that when the
                            # right current flows the drop across this element ≈ vf.
                            # Current limiting is expected from an external resistor.
                            vf = params.get("forward_voltage", 2.0)
                            ifwd = params.get("forward_current", 0.02)
                            r = max(vf / ifwd, 1.0)
                        else:
                            r = _OPEN
                        stamp_r(pn["anode"], pn["cathode"], 1.0 / r)

                    elif isinstance(comp, Diode):
                        fwd = diode_states.get(comp.component_id, False)
                        if fwd:
                            vf = params.get("forward_voltage", 0.7)
                            ifwd = params.get("forward_current", 0.1)
                            r = max(vf / ifwd, 1.0)
                        else:
                            r = _OPEN
                        stamp_r(pn["anode"], pn["cathode"], 1.0 / r)

                    elif isinstance(comp, Potentiometer):
                        total_r = params.get("total_resistance", 1000.0) or 1000.0
                        wiper = params.get("wiper_position", 0.5)
                        wiper_node = pn.get("wiper", f"_w_{comp.component_id}")
                        stamp_r(pn["p"], wiper_node, 1.0 / max(total_r * wiper, _WIRE))
                        stamp_r(wiper_node, pn["n"], 1.0 / max(total_r * (1 - wiper), _WIRE))

                    else:
                        warnings.append(
                            f"StubBackend: {comp.component_id!r} "
                            f"({type(comp).__name__}) has no stamp; skipped."
                        )
                except KeyError as e:
                    warnings.append(
                        f"StubBackend: missing port {e} for {comp.component_id!r}; skipped."
                    )

            try:
                return np.linalg.solve(G, b)
            except np.linalg.LinAlgError:
                return None

        # ── Pass 1: all diodes reverse-biased (open) ─────────────────────
        x1 = solve_pass({})
        if x1 is None:
            return SimulationResult(converged=False, warnings=("MNA solve failed (pass 1).",))

        def node_v(x: np.ndarray, node_id: str) -> float:
            if node_id == ground.node_id:
                return 0.0
            i = node_idx.get(node_id)
            return float(x[i]) if i is not None else 0.0

        # ── Determine diode states from Pass 1 ───────────────────────────
        diode_states: dict[str, bool] = {}
        for edge in graph.get_netlist():
            comp = edge.component
            pn = edge.port_nodes
            if not pn:
                continue
            if isinstance(comp, (LED, Diode)):
                va = node_v(x1, pn.get("anode", ""))
                vc = node_v(x1, pn.get("cathode", ""))
                diode_states[comp.component_id] = (va > vc)

        # ── Pass 2: apply diode states ────────────────────────────────────
        x2 = solve_pass(diode_states)
        if x2 is None:
            warnings.append("MNA solve failed (pass 2); using pass-1 result.")
            x2 = x1

        # ── Extract node voltages ─────────────────────────────────────────
        node_voltages: dict[str, float] = {ground.node_id: 0.0}
        for node in non_gnd:
            node_voltages[node.node_id] = node_v(x2, node.node_id)

        # ── Branch currents and power ─────────────────────────────────────
        branch_currents: dict[str, float] = {}
        component_power: dict[str, float] = {}

        for edge in graph.get_netlist():
            comp = edge.component
            params = comp.current_parameters()
            pn = edge.port_nodes
            cid = comp.component_id

            if not pn:
                branch_currents[cid] = 0.0
                component_power[cid] = 0.0
                continue

            def _v(port: str) -> float:
                node = pn.get(port)
                return node_voltages.get(node, 0.0) if node else 0.0

            if isinstance(comp, (LED, Diode)):
                v_across = _v("anode") - _v("cathode")
            elif isinstance(comp, VoltageSource):
                v_across = _v("pos") - _v("neg")
            else:
                v_across = _v("p") - _v("n")

            if isinstance(comp, VoltageSource):
                row = vsrc_idx[cid]
                i = float(x2[row])
            elif isinstance(comp, (Resistor, Cable, Bulb, LightSensor)):
                r = params.get("resistance", _WIRE) or _WIRE
                i = v_across / r
            elif isinstance(comp, Fuse):
                r = _OPEN if params.get("is_blown") else _WIRE
                i = v_across / r
            elif isinstance(comp, Switch):
                r = params.get("resistance", _WIRE if params.get("is_closed") else _OPEN)
                i = v_across / r
            elif isinstance(comp, Potentiometer):
                total_r = params.get("total_resistance", 1000.0) or 1000.0
                i = v_across / total_r
            elif isinstance(comp, LED):
                fwd = diode_states.get(cid, False)
                if fwd:
                    vf = params.get("forward_voltage", 2.0)
                    ifwd = params.get("forward_current", 0.02)
                    r = max(vf / ifwd, 1.0)
                else:
                    r = _OPEN
                i = v_across / r
            elif isinstance(comp, Diode):
                fwd = diode_states.get(cid, False)
                vf = params.get("forward_voltage", 0.7)
                ifwd = params.get("forward_current", 0.1)
                r = max(vf / ifwd, 1.0) if fwd else _OPEN
                i = v_across / r
            else:
                i = 0.0

            branch_currents[cid] = i
            component_power[cid] = abs(v_across * i)

        # ── Light emitters ────────────────────────────────────────────────
        emitting: set[str] = set()
        for edge in graph.get_netlist():
            comp = edge.component
            if isinstance(comp, (Bulb, LED)):
                threshold = comp.current_parameters().get("power_threshold", 0.01)
                # For LEDs, only count as emitting if forward biased
                if isinstance(comp, LED) and not diode_states.get(comp.component_id, False):
                    continue
                if component_power.get(comp.component_id, 0.0) >= threshold:
                    emitting.add(comp.component_id)

        return SimulationResult(
            node_voltages=node_voltages,
            branch_currents=branch_currents,
            component_power=component_power,
            emitting_light=frozenset(emitting),
            converged=True,
            warnings=tuple(warnings),
        )
