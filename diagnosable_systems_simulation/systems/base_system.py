from __future__ import annotations

from logging import Logger
from typing import Optional

from diagnosable_systems_simulation.actions.base import Action, ActionResult
from diagnosable_systems_simulation.electrical_simulation.circuit import CircuitGraph
from diagnosable_systems_simulation.electrical_simulation.results import SimulationResult
from diagnosable_systems_simulation.electrical_simulation.solver import SimulationRunner
from diagnosable_systems_simulation.world.affordances import Affordance
from diagnosable_systems_simulation.world.components import Component
from diagnosable_systems_simulation.world.context import WorldContext
from diagnosable_systems_simulation.world.knowledge_graph import (
    EntityType, RelationType, SystemGraph,
)

def build_circuit_from_kg(kg: SystemGraph) -> CircuitGraph:
    """
    Derive a ``CircuitGraph`` from the nominal wiring in the KG.

    Uses union-find over ELECTRICALLY_CONNECTED edges to group ports into
    nets, then assigns each net a synthetic node ID (``"gnd"`` for the net
    marked ``is_ground=True``, ``"net_<i>"`` for all others).
    """
    ec_edges = kg.edges_of_relation(RelationType.ELECTRICALLY_CONNECTED)

    # --- Union-Find ---------------------------------------------------
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ground_ports: list = []

    for e in ec_edges:
        a = (e.from_id, e.attrs["from_port"])
        b = (e.to_id,   e.attrs["to_port"])
        union(a, b)
        if e.attrs.get("is_ground"):
            ground_ports.append(a)

    # --- Assign node IDs after all unions are done --------------------
    ground_roots = {find(p) for p in ground_ports}
    node_ids: dict = {}
    counter = 0

    def node_id_for(port_key):
        nonlocal counter
        root = find(port_key)
        if root not in node_ids:
            if root in ground_roots:
                node_ids[root] = "gnd"
            else:
                node_ids[root] = f"net_{counter}"
                counter += 1
        return node_ids[root]

    # --- Build port → node_id map per component -----------------------
    port_map: dict[str, dict[str, str]] = {}
    for e in ec_edges:
        a = (e.from_id, e.attrs["from_port"])
        b = (e.to_id,   e.attrs["to_port"])
        port_map.setdefault(e.from_id, {})[e.attrs["from_port"]] = node_id_for(a)
        port_map.setdefault(e.to_id,   {})[e.attrs["to_port"]]   = node_id_for(b)

    # --- Populate CircuitGraph ----------------------------------------
    g = CircuitGraph()
    for node_id in dict.fromkeys(node_ids.values()):   # preserve insertion order
        g.add_node(node_id, is_ground=(node_id == "gnd"))

    for cid, pmap in port_map.items():
        g.add_component(kg.get_entity(cid), pmap)

    return g

class DiagnosableSystem:
    """
    Top-level assembly.  Coordinates all four layers.

    Primary data structures
    -----------------------
    kg : SystemGraph
        The system's knowledge graph.  Holds all entities (components,
        modules) and structural relations (PART_OF, CONTAINED_IN,
        ELECTRICALLY_CONNECTED).

    context : WorldContext
        Dynamic world state: tools available, inverted enclosures, open
        peepholes, etc.
    """

    def __init__(
        self,
        name: str,
        kg: SystemGraph,
        context: WorldContext,
        runner: SimulationRunner,
    ):
        self.name = name
        self._kg = kg
        self._graph = build_circuit_from_kg(kg)
        self._context = context
        self._runner = runner
        self._last_result: Optional[SimulationResult] = None
        # Make the backend accessible to actions via context.extra
        self._context.extra.setdefault("backend", self._runner.backend)
        # Tracks components that have been physically removed (id → display_name).
        # Kept separate from the KG so the NL interface can still map to them
        # and return a meaningful "not present" result.
        self._removed_components: dict[str, str] = {}
        # Nominal emitting-light set — captured on the first simulate() call so
        # test_repair() knows which Bulbs are expected to be on after a repair.
        self._nominal_emitting_light: "frozenset[str]" = frozenset()
        self._nominal_captured: bool = False
        # Snapshot taken just before the first inject_fault() call; used by
        # test_repair() to reset to the fault state non-destructively.
        self._fault_snapshot: "dict | None" = None

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def add_logger(self, logger: Logger) -> None:
        self._runner.logger = logger
    
    def simulate(self) -> SimulationResult:
        self._last_result = self._runner.run(self._graph, self._context)
        if not self._nominal_captured:
            self._nominal_emitting_light = self._last_result.emitting_light
            self._nominal_captured = True
        return self._last_result

    @property
    def last_result(self) -> Optional[SimulationResult]:
        return self._last_result

    def is_system_nominal(self) -> bool:
        """
        Return True if the current simulation result shows all nominal Bulbs lit
        AND there are no active LooseConnectionCouplings on the system.

        A loose connection is an intermittent fault: even if the last simulate()
        happened to land on the "connected" coin flip, the system is not reliably
        nominal as long as the coupling is present.
        """
        from diagnosable_systems_simulation.world.components import Bulb
        from diagnosable_systems_simulation.electrical_simulation.couplings import LooseConnectionCoupling
        if self._last_result is None or not self._last_result.converged:
            return False
        if any(isinstance(c, LooseConnectionCoupling) for c in self._runner.couplings):
            return False
        nominal_bulbs = frozenset(
            cid for cid in self._nominal_emitting_light
            if isinstance(self._kg.get_entity(cid), Bulb)
        )
        return bool(nominal_bulbs) and nominal_bulbs.issubset(self._last_result.emitting_light)

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def apply_action(self, action: Action, targets: dict[str, Component]) -> ActionResult:
        ok, reason = action.check_preconditions(targets, self._context)
        if not ok:
            return ActionResult(success=False, message=reason)
        if self._last_result is None:
            self.simulate()
        self._context.extra["_system"] = self  # actions may call system.component()
        result = action.execute(targets, self._graph, self._context, self._last_result)
        if action.mutates_graph:
            self._last_result = self._runner.run(self._graph, self._context)
            if result.observation is not None:
                result.observation.simulation_snapshot = self._last_result
        return result

    def inject_fault(self, fault_action: Action, targets: dict[str, Component]) -> ActionResult:
        if self._fault_snapshot is None:
            if self._last_result is None:
                self.simulate()
            self._fault_snapshot = self.snapshot()
        return self.apply_action(fault_action, targets)

    def remove_component(self, component_id: str) -> None:
        """
        Physically remove a component from the system entirely.

        Removes it from both the ``CircuitGraph`` (severs all port connections)
        and the ``SystemGraph`` (removes entity + all KG edges).  The component
        object is no longer reachable via ``all_components()`` or
        ``component()`` after this call.

        The component's display_name is saved in ``_removed_components`` so
        that the NL interface can still map agent requests to it and return
        a meaningful "not present" result rather than silently falling through
        to a nearby component.

        Use this to model physical removal (e.g. pulled-out LED) rather than
        degradation.
        """
        comp = self._kg.get_entity(component_id)  # fetch before deletion
        self._removed_components[component_id] = comp.display_name
        self._graph.remove_component(component_id)
        self._kg.remove_entity(component_id)

    # ------------------------------------------------------------------
    # Entity access (components & modules via the knowledge graph)
    # ------------------------------------------------------------------

    def component(self, component_id: str) -> Component:
        try:
            return self._kg.get_entity(component_id)
        except KeyError:
            raise KeyError(f"No component {component_id!r} in system {self.name!r}.")

    def all_components(self) -> dict[str, Component]:
        return self._kg.entities_of_type(EntityType.COMPONENT)

    def module_display_name(self, module_id: str) -> str:
        """Return the display name of a module (enclosure acting as module anchor)."""
        return self.component(module_id).display_name

    def all_modules(self) -> dict[str, str]:
        """Return {module_id: display_name} for all components that have PART_OF edges."""
        module_ids = {e.to_id for e in self._kg.edges_of_relation(RelationType.PART_OF)}
        return {mid: self._kg.get_entity(mid).display_name for mid in module_ids}

    def parts_of_module(self, module_id: str) -> list[Component]:
        """All components that are PART_OF the given module."""
        return [
            self._kg.get_entity(e.from_id)
            for e in self._kg.incoming(module_id, RelationType.PART_OF)
        ]

    def contained_in(self, enclosure_id: str) -> list[Component]:
        """All components physically CONTAINED_IN the given enclosure."""
        return [
            self._kg.get_entity(e.from_id)
            for e in self._kg.incoming(enclosure_id, RelationType.CONTAINED_IN)
        ]

    def get_affordances(self, component_id: str) -> set[Affordance]:
        comp = self.component(component_id)
        return comp.affordances.all_active(comp, self._context)

    @property
    def kg(self) -> SystemGraph:
        return self._kg

    @property
    def context(self) -> WorldContext:
        return self._context

    @property
    def graph(self) -> CircuitGraph:
        return self._graph

    # ------------------------------------------------------------------
    # State snapshot / restore  (used by hypothesis verification)
    # ------------------------------------------------------------------

    # Component attributes that hold mutable non-overlay state.
    _STATEFUL_ATTRS: tuple[str, ...] = ("is_closed", "is_inverted", "is_rotated", "is_open", "is_blown")

    def snapshot(self) -> dict:
        """
        Capture the full mutable state of the circuit:
        port connections, fault overlays, component state flags,
        cable _orig_connections, static affordances, and any
        circuit-graph-only components (e.g. shorts added by short_ports).
        """
        comps = self.all_components()
        # IDs present in the circuit graph but not in the KG (ghost components:
        # shorts, probes added directly by diagnostic actions).
        circuit_only_ids = set(self._graph._edges) - set(comps)
        return {
            "port_connections": {
                cid: {p.name: p.node_id for p in c.ports}
                for cid, c in comps.items()
            },
            "fault_overlays": {
                cid: dict(c._fault_overlay)
                for cid, c in comps.items()
            },
            "component_states": {
                cid: {
                    attr: getattr(c, attr)
                    for attr in self._STATEFUL_ATTRS
                    if hasattr(c, attr)
                }
                for cid, c in comps.items()
            },
            "orig_connections": {
                cid: dict(c._orig_connections)
                for cid, c in comps.items()
                if hasattr(c, "_orig_connections")
            },
            "detached_cable_ports": {
                cid: dict(c._detached_cable_ports)
                for cid, c in comps.items()
                if hasattr(c, "_detached_cable_ports")
            },
            "static_affordances": {
                cid: set(c.affordances._static)
                for cid, c in comps.items()
            },
            "dynamic_affordances": {
                cid: set(c.affordances._dynamic)
                for cid, c in comps.items()
            },
            "circuit_only_ids": circuit_only_ids,
            "_runner_couplings": list(self._runner.couplings),
        }

    def restore_snapshot(self, snap: dict, exclude_ids: "set[str] | None" = None) -> None:
        """
        Restore the circuit to a previously snapshotted state.

        Components whose IDs appear in *exclude_ids* are left untouched
        (they have been intentionally repaired and should stay fixed).
        Re-runs the simulation at the end so results are up to date.
        """
        exclude = exclude_ids or set()
        for cid, comp in self.all_components().items():
            if cid in exclude:
                continue

            # --- Port connections ----------------------------------------
            snap_ports = snap["port_connections"].get(cid, {})
            for p in comp.ports:
                snap_node = snap_ports.get(p.name)
                curr_node = p.node_id
                if snap_node == curr_node:
                    continue
                if curr_node is not None:
                    self._graph.disconnect_port(cid, p.name)
                if snap_node is not None:
                    self._graph.reconnect_port(cid, p.name, snap_node)

            # --- Fault overlays ------------------------------------------
            prev_overlay = dict(comp._fault_overlay)
            snap_overlay = snap["fault_overlays"].get(cid, {})
            comp._fault_overlay.clear()
            comp._fault_overlay.update(snap_overlay)

            # Re-insert short-circuit graph element if it was present in the
            # snapshot but was removed during a repair (apply_repairs removes it
            # from the graph but it's not tracked in the KG, so restore_snapshot
            # cannot rely on port_connections to bring it back).
            prev_short = prev_overlay.get("short_graph_id")
            snap_short = snap_overlay.get("short_graph_id")
            if snap_short and snap_short != prev_short:
                try:
                    self._graph.remove_component(snap_short)  # remove stale if any
                except (KeyError, Exception):
                    pass
                node_a = snap_overlay.get("short_node_a")
                node_b = snap_overlay.get("short_node_b")
                if node_a and node_b:
                    try:
                        self._graph.short_nodes(node_a, node_b, snap_short)
                    except Exception:
                        pass  # already present

            # --- Stateful attributes (is_closed, is_inverted, …) ---------
            for attr, val in snap["component_states"].get(cid, {}).items():
                setattr(comp, attr, val)

            # --- _orig_connections for cables -----------------------------
            orig = snap["orig_connections"].get(cid)
            if orig is not None:
                comp._orig_connections = dict(orig)
            elif hasattr(comp, "_orig_connections"):
                comp._orig_connections = {}

            # --- _detached_cable_ports for cable-neightbour -----------------------------
            detached = snap["detached_cable_ports"].get(cid)
            if detached is not None:
                comp._detached_cable_ports = dict(detached)
            elif hasattr(comp, "_detached_cable_ports"):
                comp._detached_cable_ports = {}

            # --- Static affordances --------------------------------------
            snap_static = snap["static_affordances"].get(cid, set())
            curr_static = set(comp.affordances._static)
            for a in curr_static - snap_static:
                comp.affordances.remove(a)
            for a in snap_static - curr_static:
                comp.affordances.add(a)

            # --- Dynamic affordances (e.g. RECONNECTABLE added by DisconnectCable) ---
            if "dynamic_affordances" in snap:
                snap_dynamic = snap["dynamic_affordances"].get(cid, set())
                curr_dynamic = set(comp.affordances._dynamic)
                for a in curr_dynamic - snap_dynamic:
                    comp.affordances._dynamic.discard(a)
                for a in snap_dynamic - curr_dynamic:
                    comp.affordances._dynamic.add(a)

        # --- Remove ghost circuit components added after the snapshot --------
        # These are components in _graph._edges that are not in the KG (e.g.
        # shorts inserted by short_ports diagnostic actions).  Any ghost that
        # was not present at snapshot time must be removed so it does not
        # provide a bypass path in subsequent simulations.
        snap_ghosts = snap.get("circuit_only_ids", set())
        current_ghosts = set(self._graph._edges) - set(self.all_components())
        for ghost_id in current_ghosts - snap_ghosts:
            try:
                self._graph.remove_component(ghost_id)
            except (KeyError, Exception):
                pass

        # --- Restore couplings list --------------------------------------
        if "_runner_couplings" in snap:
            from diagnosable_systems_simulation.electrical_simulation.couplings import LooseConnectionCoupling
            restored = [
                c for c in snap["_runner_couplings"]
                if not (
                    isinstance(c, LooseConnectionCoupling)
                    and exclude_ids is not None
                    and c.component_id in exclude_ids
                )
            ]
            self._runner.couplings = restored
            for c in self._runner.couplings:
                if hasattr(c, "reset"):
                    c.reset()

        self.simulate()

    # ------------------------------------------------------------------
    # Hypothesis-verification helper
    # ------------------------------------------------------------------

    def apply_repairs(self, component_ids: "set[str]") -> "ActionCost":
        """
        Physically repair components in the live circuit without simulating
        or restoring any snapshot.

        For each component ID:
          - Cables: ports that are floating OR connected to the wrong node are
            reconnected to their original nodes (from ``_orig_connections``).
            This covers both "detached cable" faults (floating port) and
            "crossed cable" faults (port connected but to the wrong net, e.g.
            after a polarity-swap fault injection).
          - Components with a fault overlay: the overlay is cleared.

        Returns the total ``ActionCost`` of the repairs performed, computed
        from the canonical action costs (ReconnectCable = 10s per cable,
        component replacement = 120s per component).  Callers that attribute
        hypothesis-verification cost should use this value rather than any
        fixed overhead.

        Use this to persist confirmed repairs between partial hypothesis
        verifications, so that ``restore_snapshot(exclude_ids=repaired)``
        leaves those components in the repaired state rather than the
        fault state they were in when ``test_repair()`` last exited.
        """
        from diagnosable_systems_simulation.actions.base import ActionCost
        from diagnosable_systems_simulation.world.components import Cable
        _RECONNECT_COST = ActionCost(time=10.0)   # mirrors ReconnectCable.cost
        _REPLACE_COST   = ActionCost(time=120.0)  # mirrors ReplaceComponent default
        total = ActionCost()
        for cid in component_ids:
            try:
                comp = self.component(cid)
            except KeyError:
                continue
            if isinstance(comp, Cable):
                orig = getattr(comp, "_orig_connections", {})
                for port_name, node_id in orig.items():
                    port = comp.port(port_name)
                    if not port.is_connected():
                        self._graph.reconnect_port(cid, port_name, node_id)
                        total = total + _RECONNECT_COST
                    elif port.node_id != node_id:
                        # Connected to wrong net (crossed-cable fault).
                        self._graph.disconnect_port(cid, port_name)
                        self._graph.reconnect_port(cid, port_name, node_id)
                        total = total + _RECONNECT_COST
            to_delete = []
            if (detached := getattr(comp, "_detached_cable_ports", {})):
                for p_port, (cid, c_port, nid) in detached.items():
                    cable = self.component(cid)
                    self._graph.reconnect_port(cable.component_id, c_port, nid)
                    cable.affordances.remove(Affordance.RECONNECTABLE)
                    to_delete.append(p_port)
                    total = total + _RECONNECT_COST
                for p in to_delete:
                    del detached[p]

                if not comp._detached_cable_ports:
                    comp.affordances.remove(Affordance.RECONNECTABLE)
                    del comp._detached_cable_ports
            if comp._fault_overlay:
                # If this component was part of a short-circuit fault, remove the
                # synthetic graph element inserted by ShortCircuit.execute().
                short_graph_id = comp._fault_overlay.get("short_graph_id")
                if short_graph_id is not None:
                    try:
                        self._graph.remove_component(short_graph_id)
                    except (KeyError, Exception):
                        pass  # already removed by the other cable in the pair
                comp._fault_overlay.clear()
                total = total + _REPLACE_COST

        # Remove LooseConnectionCouplings for the repaired components and
        # reconnect any port the coupling left dangling.
        # A loose connection IS the fault — permanently gone after repair.
        from diagnosable_systems_simulation.electrical_simulation.couplings import LooseConnectionCoupling
        loose_to_remove = [
            c for c in self._runner.couplings
            if isinstance(c, LooseConnectionCoupling) and c.component_id in component_ids
        ]
        self._runner.couplings = [c for c in self._runner.couplings if c not in loose_to_remove]
        for c in loose_to_remove:
            if c._currently_disconnected and c._saved_node is not None:
                self._graph.reconnect_port(c.component_id, c.port_name, c._saved_node)
                c._currently_disconnected = False
                total = total + _RECONNECT_COST

        return total

    def test_repair(
        self,
        component_ids: "set[str]",
        *,
        already_repaired_ids: "set[str] | None" = None,
    ) -> bool:
        """
        Temporarily repair *component_ids*, re-simulate, and return True if
        every component that was lit in the nominal (pre-fault) state is lit.

        The circuit is always restored to the fault state before returning,
        so the caller sees no persistent side-effects.

        Parameters
        ----------
        component_ids:
            IDs of the components to repair.  For each:
              - disconnected cables: floating ports are reconnected to their
                original nodes (taken from ``_orig_connections``).
              - components with a fault overlay: the overlay is cleared.
        already_repaired_ids:
            Components confirmed repaired in previous partial verifications;
            excluded from the snapshot restore so they remain fixed during
            the test.
        """
        from diagnosable_systems_simulation.world.components import Bulb, Cable, PhysicalEnclosure

        fault_snapshot = self._fault_snapshot
        # Only check main load Bulbs — indicator bulbs (is_indicator=True) and
        # deliberately removed components must not block test_repair from
        # returning True when the primary load is restored.
        def _is_main_bulb(cid: str) -> bool:
            try:
                comp = self._kg.get_entity(cid)
                return isinstance(comp, Bulb) and not getattr(comp, "is_indicator", False)
            except KeyError:
                return False  # component was physically removed

        nominal_lit: "frozenset[str]" = frozenset(
            cid for cid in self._nominal_emitting_light
            if _is_main_bulb(cid)
        )
        already = already_repaired_ids or set()

        # 1. Reset to fault state (preserving previously confirmed repairs)
        if fault_snapshot is not None:
            self.restore_snapshot(fault_snapshot, exclude_ids=already)

        # 2. Strip LooseConnectionCouplings for the repaired components so the
        # hypothetical repair simulation is not sabotaged by the intermittent
        # fault still firing.  They are restored after restore_snapshot() so
        # test_repair() remains non-destructive — the coupling IS the fault.
        from diagnosable_systems_simulation.electrical_simulation.couplings import LooseConnectionCoupling
        loose_removed = [
            c for c in self._runner.couplings
            if isinstance(c, LooseConnectionCoupling) and c.component_id in component_ids
        ]
        self._runner.couplings = [c for c in self._runner.couplings if c not in loose_removed]

        # If the coupling left the port disconnected (new per-run semantics hold
        # the open state across the simulate() inside restore_snapshot), reconnect
        # it now using the coupling's own saved node before apply_repairs runs.
        for c in loose_removed:
            if c._currently_disconnected and c._saved_node is not None:
                self._graph.reconnect_port(c.component_id, c.port_name, c._saved_node)
                c._currently_disconnected = False

        # 2b. Apply repairs (shared logic with apply_repairs, including short removal)
        self.apply_repairs(component_ids)

        # 2c. For PhysicalEnclosure components: "repairing" means repositioning
        # (rotating/moving) so any coupling that checks is_rotated sees the
        # enclosure as displaced.  restore_snapshot undoes this automatically
        # since is_rotated is listed in _STATEFUL_ATTRS.
        for cid in component_ids:
            try:
                comp = self._kg.get_entity(cid)
                if isinstance(comp, PhysicalEnclosure):
                    comp.is_rotated = True
            except KeyError:
                pass

        # 3. Re-simulate
        result = self.simulate()

        # 4. Check whether all expected outputs are lit.
        # A non-converged result means the circuit is oscillating — the lamp
        # state is ambiguous and no repair can be confirmed from it.
        lamp_on = (
            bool(nominal_lit)
            and result.converged
            and nominal_lit.issubset(result.emitting_light)
        )

        # 4a. Loose-connection guard: if any LooseConnectionCoupling was NOT
        # removed (i.e. its component was not among the repaired candidates),
        # the fault is still active and the lamp being on is a lucky random draw.
        # Suppress the positive result in that case.
        if lamp_on and any(
            isinstance(c, LooseConnectionCoupling)
            for c in self._runner.couplings
        ):
            lamp_on = False

        # 4b. Bypass guard: if the lamp is on, verify it is controlled by the
        # switch chain and not by a diagnostic bypass (e.g. a residual short).
        # Pick one closed, manually-operated Switch, open it, re-simulate, and
        # confirm the lamp goes off.  Relay-controlled switches are identified by
        # the coupling re-closing them during simulate() and are skipped.
        if lamp_on:
            from diagnosable_systems_simulation.world.components import Switch as _Switch
            _test_switch = next(
                (sw for sw in self.all_components().values()
                 if isinstance(sw, _Switch) and getattr(sw, "is_closed", False)),
                None,
            )
            if _test_switch is not None:
                _test_switch.is_closed = False
                _check = self.simulate()
                _reclosed_by_coupling = getattr(_test_switch, "is_closed", False)
                if not _reclosed_by_coupling:
                    _test_switch.is_closed = True  # restore manually
                if not _reclosed_by_coupling:
                    # At least one nominal bulb must go off (multi-chain: other
                    # chains may stay powered, but this switch must control something)
                    lamp_on = bool(nominal_lit - _check.emitting_light)

        # 5. Restore back to fault state — caller decides what to persist
        if fault_snapshot is not None:
            # restore_snapshot already restores _runner.couplings from the snapshot,
            # so do NOT extend — that would add duplicate LooseConnectionCouplings.
            self.restore_snapshot(fault_snapshot, exclude_ids=already)
        else:
            # No snapshot: restore the stripped couplings manually.
            self._runner.couplings.extend(loose_removed)

        return lamp_on

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        comps = len(self._kg.entities_of_type(EntityType.COMPONENT))
        mods  = len(self.all_modules())
        return (
            f"DiagnosableSystem({self.name!r}, "
            f"components={comps}, modules={mods}, "
            f"simulated={self._last_result is not None})"
        )
