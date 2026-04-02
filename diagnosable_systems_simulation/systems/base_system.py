from __future__ import annotations

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

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def add_logger(self, logger: Logger) -> None:
        self._runner.logger = logger
    
    def simulate(self) -> SimulationResult:
        self._last_result = self._runner.run(self._graph, self._context)
        return self._last_result

    @property
    def last_result(self) -> Optional[SimulationResult]:
        return self._last_result

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def apply_action(self, action: Action, targets: dict[str, Component]) -> ActionResult:
        ok, reason = action.check_preconditions(targets, self._context)
        if not ok:
            return ActionResult(success=False, message=reason)
        if self._last_result is None:
            self.simulate()
        result = action.execute(targets, self._graph, self._context, self._last_result)
        if action.mutates_graph:
            self._last_result = self._runner.run(self._graph, self._context)
            if result.observation is not None:
                result.observation.simulation_snapshot = self._last_result
        return result

    def inject_fault(self, fault_action: Action, targets: dict[str, Component]) -> ActionResult:
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
    _STATEFUL_ATTRS: tuple[str, ...] = ("is_closed", "is_inverted", "is_open", "is_blown")

    def snapshot(self) -> dict:
        """
        Capture the full mutable state of the circuit:
        port connections, fault overlays, component state flags,
        cable _orig_connections, and static affordances.
        """
        comps = self.all_components()
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
            "static_affordances": {
                cid: set(c.affordances._static)
                for cid, c in comps.items()
            },
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
            comp._fault_overlay.clear()
            comp._fault_overlay.update(snap["fault_overlays"].get(cid, {}))

            # --- Stateful attributes (is_closed, is_inverted, …) ---------
            for attr, val in snap["component_states"].get(cid, {}).items():
                setattr(comp, attr, val)

            # --- _orig_connections for cables -----------------------------
            orig = snap["orig_connections"].get(cid)
            if orig is not None:
                comp._orig_connections = dict(orig)
            elif hasattr(comp, "_orig_connections"):
                comp._orig_connections = {}

            # --- Static affordances --------------------------------------
            snap_static = snap["static_affordances"].get(cid, set())
            curr_static = set(comp.affordances._static)
            for a in curr_static - snap_static:
                comp.affordances.remove(a)
            for a in snap_static - curr_static:
                comp.affordances.add(a)

        self.simulate()

    # ------------------------------------------------------------------
    # Hypothesis-verification helper
    # ------------------------------------------------------------------

    def apply_repairs(self, component_ids: "set[str]") -> None:
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

        Use this to persist confirmed repairs between partial hypothesis
        verifications, so that ``restore_snapshot(exclude_ids=repaired)``
        leaves those components in the repaired state rather than the
        fault state they were in when ``test_repair()`` last exited.
        """
        from diagnosable_systems_simulation.world.components import Cable
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
                    elif port.node_id != node_id:
                        # Connected to wrong net (crossed-cable fault).
                        self._graph.disconnect_port(cid, port_name)
                        self._graph.reconnect_port(cid, port_name, node_id)
            if comp._fault_overlay:
                comp._fault_overlay.clear()

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
        from diagnosable_systems_simulation.world.components import Bulb, Cable

        fault_snapshot = getattr(self, "_fault_snapshot", None)
        # Only check main load Bulbs — indicator LEDs are accessories and may
        # have been deliberately removed, so including them would permanently
        # prevent test_repair from ever returning True.
        def _is_bulb(cid: str) -> bool:
            try:
                return isinstance(self._kg.get_entity(cid), Bulb)
            except KeyError:
                return False  # component was physically removed

        nominal_lit: "frozenset[str]" = frozenset(
            cid for cid in getattr(self, "_nominal_emitting_light", frozenset())
            if _is_bulb(cid)
        )
        already = already_repaired_ids or set()

        # 1. Reset to fault state (preserving previously confirmed repairs)
        if fault_snapshot is not None:
            self.restore_snapshot(fault_snapshot, exclude_ids=already)

        # 2. Apply repairs
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
                    elif port.node_id != node_id:
                        # Connected to wrong net (crossed-cable fault).
                        self._graph.disconnect_port(cid, port_name)
                        self._graph.reconnect_port(cid, port_name, node_id)
            if comp._fault_overlay:
                comp._fault_overlay.clear()

        # 3. Re-simulate
        result = self.simulate()

        # 4. Check whether all expected outputs are lit
        lamp_on = bool(nominal_lit) and nominal_lit.issubset(result.emitting_light)

        # 5. Restore back to fault state — caller decides what to persist
        if fault_snapshot is not None:
            self.restore_snapshot(fault_snapshot, exclude_ids=already)

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
