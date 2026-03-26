"""
Simulation dump utilities.

Two levels of reporting:

  dump_electrical(result, graph)
      Raw electrical snapshot: every node voltage and every branch
      current / power.  No interpretation — just numbers.

  dump_state(system)
      Full component-level report.  For each component: type, current
      parameters (including any fault overlay), port→node mapping,
      active affordances, and — if a simulation result is available —
      the electrical quantities and lit status.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from diagnosable_systems_simulation.utils.units import format_value
from diagnosable_systems_simulation.world.components import Bulb, LED

if TYPE_CHECKING:
    from diagnosable_systems_simulation.electrical_simulation.circuit import CircuitGraph
    from diagnosable_systems_simulation.electrical_simulation.results import SimulationResult
    from diagnosable_systems_simulation.systems.base_system import DiagnosableSystem


# ---------------------------------------------------------------------------
# 2.1  Raw electrical dump
# ---------------------------------------------------------------------------

def dump_electrical(
    result: "SimulationResult",
    graph: "CircuitGraph",
    *,
    width: int = 72,
) -> str:
    """
    Return a multi-line string with all node voltages and branch
    currents / power values from *result*.

    Parameters
    ----------
    result : SimulationResult
        The object returned by ``system.simulate()`` or
        ``system.last_result``.
    graph : CircuitGraph
        The circuit graph (``system.graph``).  Used to annotate
        whether a node is ground and to sort components by type.
    width : int
        Approximate line width for separators.
    """
    lines: list[str] = []
    sep = "─" * width

    def h(title: str) -> None:
        lines.append(sep)
        lines.append(f"  {title}")
        lines.append(sep)

    # ── Header ────────────────────────────────────────────────────────────
    status = "CONVERGED" if result.converged else "NOT CONVERGED"
    lines.append(f"Electrical Dump  [{status}]")
    if result.warnings:
        for w in result.warnings:
            lines.append(f"  ⚠ {w}")

    # ── Node voltages ─────────────────────────────────────────────────────
    h("Node Voltages")
    ground_ids = {n.node_id for n in graph.get_nodes() if n.is_ground}

    # Build reverse map: node_id -> list of "component_id.port" strings
    node_members: dict[str, list[str]] = {}
    for edge in graph.get_netlist():
        for port, nid in edge.port_nodes.items():
            node_members.setdefault(nid, []).append(f"{edge.component_id}.{port}")

    rows = sorted(result.node_voltages.items(), key=lambda kv: kv[1], reverse=True)
    col_w = max((len(nid) for nid, _ in rows), default=6) + 2
    for nid, v in rows:
        gnd_tag = "  [GND]" if nid in ground_ids else ""
        members = "  ← " + ", ".join(sorted(node_members.get(nid, [])))
        lines.append(f"  {nid:<{col_w}}  {format_value(v, 'V'):>12}{gnd_tag}{members}")

    # ── Branch currents and power ─────────────────────────────────────────
    h("Branch Currents & Power")
    all_cids = sorted(
        set(result.branch_currents) | set(result.component_power)
    )
    col_w = max((len(c) for c in all_cids), default=10) + 2
    lines.append(
        f"  {'component':<{col_w}}  {'current':>12}  {'power':>12}  lit"
    )
    lines.append("  " + "·" * (col_w + 32))
    for cid in all_cids:
        i = result.branch_currents.get(cid, 0.0)
        p = result.component_power.get(cid, 0.0)
        lit = "✓" if cid in result.emitting_light else " "
        lines.append(
            f"  {cid:<{col_w}}  {format_value(i, 'A'):>12}  {format_value(p, 'W'):>12}  {lit}"
        )

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2.2  Full component state report
# ---------------------------------------------------------------------------

def dump_state(
    system: "DiagnosableSystem",
    *,
    width: int = 72,
) -> str:
    """
    Return a multi-line string with the full state of every component.

    For each component the report shows:
      - Component type and display name
      - Current parameters (nominal merged with any active fault overlay)
      - Port names and the circuit node each port is wired to
      - Active affordances under the current world context
      - Electrical quantities (voltage across, current, power) — only
        when ``system.last_result`` is available
      - Whether the component is currently emitting light

    Parameters
    ----------
    system : DiagnosableSystem
        The system to inspect.
    """
    from diagnosable_systems_simulation.electrical_simulation.circuit import CircuitGraph

    result = system.last_result
    graph: CircuitGraph = system.graph
    context = system.context

    lines: list[str] = []
    sep = "─" * width
    thin = "·" * width

    lines.append(f"Component State Report — {system.name}")
    if result is None:
        lines.append("  (no simulation result available — run system.simulate() first)")
    else:
        status = "CONVERGED" if result.converged else "NOT CONVERGED"
        lines.append(f"  simulation: {status}")
    lines.append(sep)

    all_comps = system.all_components()
    for cid, comp in sorted(all_comps.items()):
        lines.append(f"  [{cid}]  {comp.display_name}  ({type(comp).__name__})")

        # Parameters
        params = comp.current_parameters()
        nominal = comp.nominal_parameters()
        for key, val in params.items():
            nom_val = nominal.get(key)
            changed = (nom_val is not None) and (val != nom_val)
            tag = "  ← FAULTED" if changed else ""
            lines.append(f"    param  {key} = {val}{tag}")

        # Port → node mapping
        port_nodes = graph.nodes_of(cid) if _has_component(graph, cid) else {}
        for port_name, node_id in (port_nodes or {}).items():
            v = result.node_voltages.get(node_id, "n/a") if result else "n/a"
            v_str = format_value(v, "V") if isinstance(v, float) else v
            lines.append(f"    port   {port_name:6s} → {node_id:<12}  {v_str}")

        # Active affordances
        active_aff = comp.affordances.all_active(comp, context)
        if active_aff:
            lines.append(f"    afford {', '.join(a.name for a in sorted(active_aff, key=lambda a: a.name))}")

        # Electrical summary (only if result available)
        if result is not None:
            i = result.branch_currents.get(cid)
            p = result.component_power.get(cid)
            lit = cid in result.emitting_light
            if i is not None:
                lines.append(f"    elec   I = {format_value(i, 'A')},  P = {format_value(p or 0.0, 'W')}")
            if isinstance(comp, (Bulb, LED)):
                lines.append(f"    light  {'ON  ✓' if lit else 'OFF  '}")

        lines.append("  " + thin)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _has_component(graph: "CircuitGraph", cid: str) -> bool:
    """True if the graph has an entry for this component id."""
    try:
        graph.nodes_of(cid)
        return True
    except (KeyError, AttributeError):
        return False
