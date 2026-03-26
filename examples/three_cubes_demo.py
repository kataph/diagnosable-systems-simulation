"""
Three cubes lamp — end-to-end usage demo.

Run from the repo root:
python -m examples.three_cubes_demo
"""
from diagnosable_systems_simulation.electrical_simulation.backend.stub import StubBackend
from diagnosable_systems_simulation.electrical_simulation.backend.spice import PySpiceBackend
# BACKEND = StubBackend()
BACKEND = PySpiceBackend()

from diagnosable_systems_simulation.systems.three_cubes.factory import build_three_cubes_system
from diagnosable_systems_simulation.actions.diagnostic_actions import (
    ObserveComponent, MeasureVoltage, ToggleSwitch,
    InvertEnclosure, OpenPeephole,
)
from diagnosable_systems_simulation.actions.fault_actions import DisconnectCable, ReconnectCable, DegradeComponent, ForceSwitch


def separator(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def show(result, action=None) -> None:
    print(f"  success : {result.success}")
    print(f"  message : {result.message}")
    if result.observation:
        for prop in result.observation.properties:
            unit = f" {prop.unit}" if prop.unit else ""
            print(f"  obs     : {prop.name} = {prop.value}{unit}")
    if action is not None and action.cost.time:
        cost = action.cost
        print(f"  cost    : {cost.time}s", end="")
        if cost.equipment:
            print(f", needs {cost.equipment}", end="")
        print()


# ── Build the system ────────────────────────────────────────────────────────

system = build_three_cubes_system(
    backend=BACKEND,
    extra_tools={"multimeter"},        # agent starts with a multimeter
)

# ── Module overview ──────────────────────────────────────────────────────────

separator("Modules (via knowledge graph)")
for mid, display_name in system.all_modules().items():
    parts = system.parts_of_module(mid)
    print(f"  {mid:15s}  '{display_name}'")
    print(f"    parts ({len(parts)}): {[c.component_id for c in parts]}")

# ── Nominal simulation ───────────────────────────────────────────────────────

separator("Nominal simulation")
result = system.simulate()
print(f"  converged : {result.converged}")
print(f"  lit       : {result.emitting_light}")
psu_pos_node = system.graph.nodes_of("psu_source")["pos"]
print(f"  psu_pos   : {result.voltage(psu_pos_node):.2f} V")

# ── Observe a component that is always visible ───────────────────────────────

separator("Observe PSU green LED (always visible on top face)")
r = system.apply_action(
    ObserveComponent(),
    {"subject": system.component("psu_green_led")},
)
show(r)

# ── Try to observe an internal component (should fail) ───────────────────────

separator("Observe internal_bulb — should fail (cube not inverted yet)")
r = system.apply_action(
    ObserveComponent(),
    {"subject": system.component("internal_bulb")},
)
show(r)

# ── Invert the load cube to expose the internals ─────────────────────────────

separator("Invert load cube")
r = system.apply_action(
    InvertEnclosure(),
    {"subject": system.component("cube_load")},
)
show(r)

# ── Now observe the internal bulb ────────────────────────────────────────────

separator("Observe internal_bulb — now visible")
r = system.apply_action(
    ObserveComponent(),
    {"subject": system.component("internal_bulb")},
)
show(r)

# ── Measure voltage across main bulb ─────────────────────────────────────────

separator("Measure voltage at main_bulb (requires multimeter)")
r = system.apply_action(
    MeasureVoltage(),
    {"subject": system.component("main_bulb")},
)
show(r)

# ── Toggle the switch (opens it) ─────────────────────────────────────────────

separator("Toggle control switch → open")
r = system.apply_action(
    ToggleSwitch(),
    {"subject": system.component("ctrl_switch")},
)
show(r)
print(f"  lit after : {system.last_result.emitting_light}")

# ── Measure voltage across main bulb ─────────────────────────────────────────

separator("Measure voltage at main_bulb 2x (requires multimeter)")
r = system.apply_action(
    MeasureVoltage(),
    {"subject": system.component("main_bulb")},
)
show(r)

# ── Inject a fault: disconnect the PSU output cable ──────────────────────────

separator("Fault injection: disconnect PSU output cable (+)")
r = system.inject_fault(
    DisconnectCable(port_names=["n"]),          # disconnect the load-side end
    {"subject": system.component("psu_cable_pos")},
)
show(r)
print(f"  lit after : {system.last_result.emitting_light}")

# ── Access module-level helpers ───────────────────────────────────────────────

separator("Access components via knowledge graph")
ctrl_name = system.module_display_name("cube_ctrl")
ctrl_parts = system.parts_of_module("cube_ctrl")
ctrl_switch = system.component("ctrl_switch")
print(f"  Module 'cube_ctrl' display name: '{ctrl_name}'")
print(f"  Parts of control module: {[c.component_id for c in ctrl_parts]}")
print(f"  ctrl_switch.is_closed: {ctrl_switch.is_closed}")
print(f"  contained_in cube_load: {[c.component_id for c in system.contained_in('cube_load')]}")
print(f"  KG summary: {system.kg}")

# ── Fault injections ───────────────────────────────────────────────

def fresh():
    return build_three_cubes_system(backend=BACKEND)

def show(label, s, expected, notes=""):
    r = s.last_result
    green = r.is_lit("psu_green_led"); red = r.is_lit("ctrl_red_led"); lamp = r.is_lit("main_bulb")
    ok = (green == expected[0] and red == expected[1] and lamp == expected[2])
    print(f"  {'✓' if ok else '✗'} {label}: green={'ON' if green else 'OFF'} red={'ON' if red else 'OFF'} lamp={'ON' if lamp else 'OFF'}  {notes}")

separator("Scenario verification")

s = fresh()
r = s.inject_fault(DisconnectCable(port_names=["n"]), {"subject": s.component("ctrl_cable_in_pos")})
show("S0 cable detached from switch", s, (True, False, False))

s = fresh()
s.inject_fault(DegradeComponent({"resistance": 1e9}), {"subject": s.component("main_bulb")})
show("S1 burned lamp filaments", s, (True, False, False), "(internal_bulb still lit but hidden inside cube)")

s = fresh()
s.inject_fault(DegradeComponent({"voltage": 0.0}), {"subject": s.component("psu_source")})
show("S2 battery depleted", s, (False, False, False))

s = fresh()
s.inject_fault(DegradeComponent({"voltage": -12.0}), {"subject": s.component("psu_source")})
show("S3 battery reversed", s, (False, True, False))

s = fresh()
pos_n = s.graph.nodes_of("ctrl_cable_in_pos")["n"]
neg_n = s.graph.nodes_of("ctrl_cable_in_neg")["n"]
s.inject_fault(DisconnectCable(port_names=["n"]), {"subject": s.component("ctrl_cable_in_pos")})
s.inject_fault(DisconnectCable(port_names=["n"]), {"subject": s.component("ctrl_cable_in_neg")})
s.inject_fault(ReconnectCable(connections={"n": neg_n}), {"subject": s.component("ctrl_cable_in_pos")})
s.inject_fault(ReconnectCable(connections={"n": pos_n}), {"subject": s.component("ctrl_cable_in_neg")})
show("S4 crossed wires", s, (True, True, False))

s = fresh()
s.inject_fault(ForceSwitch(is_closed=False), {"subject": s.component("ctrl_switch")})
show("S5 switch always open", s, (True, False, False))


# ── S0 Diagnostic walkthrough ─────────────────────────────────────────────────
#
# Scenario: ctrl_cable_in_pos has its output (n) end detached from the switch.
# Symptom:  green LED on, lamp off, red LED off.
# Task:     locate the fault through a sequence of diagnostic actions, then repair.

from diagnosable_systems_simulation.actions.diagnostic_actions import (
    ObserveComponent, MeasureVoltage, InvertEnclosure, InspectConnections,
    TestContinuity,
)

separator("S0 Diagnostic walkthrough — cable detached from control switch")


def show_action(result) -> None:
    print(f"  success : {result.success}")
    print(f"  message : {result.message}")
    if result.observation:
        for prop in result.observation.properties:
            unit = f" {prop.unit}" if prop.unit else ""
            print(f"  obs     : {prop.name} = {prop.value}{unit}")


s = build_three_cubes_system(backend=BACKEND, extra_tools={"multimeter"})

# Save the node we'll need to reconnect to later, before the fault is injected
node_ctrl_in_pos = s.graph.nodes_of("ctrl_cable_in_pos")["n"]

# ── Fault injection ───────────────────────────────────────────────────────────

separator("  [fault] Detach ctrl_cable_in_pos output end")
r = s.inject_fault(
    DisconnectCable(port_names=["n"]),
    {"subject": s.component("ctrl_cable_in_pos")},
)
show_r = s.last_result
print(f"  green={'ON' if show_r.is_lit('psu_green_led') else 'OFF'}  "
      f"red={'ON' if show_r.is_lit('ctrl_red_led') else 'OFF'}  "
      f"lamp={'ON' if show_r.is_lit('main_bulb') else 'OFF'}")

# ── Preliminary observations ──────────────────────────────────────────────────

separator("  [obs] Observe PSU green LED")
r = s.apply_action(ObserveComponent(), {"subject": s.component("psu_green_led")})
show_action(r)

separator("  [obs] Observe control red LED")
r = s.apply_action(ObserveComponent(), {"subject": s.component("ctrl_red_led")})
show_action(r)

separator("  [obs] Observe main bulb (lamp)")
r = s.apply_action(ObserveComponent(), {"subject": s.component("main_bulb")})
show_action(r)

# ── Continuity test on the lamp — should be nominal ───────────────────────────

separator("  [test] Continuity test on main_bulb — expect nominal")
r = s.apply_action(TestContinuity(), {"subject": s.component("main_bulb")})
show_action(r)

# ── Voltage measurements at control module outputs ────────────────────────────

separator("  [meas] Voltage at ctrl_cable_out_pos — expect ~0 V (no current through open switch)")
r = s.apply_action(MeasureVoltage(), {"subject": s.component("ctrl_cable_out_pos")})
show_action(r)

separator("  [meas] Voltage at ctrl_cable_out_neg — expect 0 V (ground)")
r = s.apply_action(MeasureVoltage(), {"subject": s.component("ctrl_cable_out_neg")})
show_action(r)

# ── Voltage measurements at control module inputs — should be correct ─────────

separator("  [meas] Voltage at ctrl_cable_in_pos — expect 12 V on p-port")
r = s.apply_action(MeasureVoltage(), {"subject": s.component("ctrl_cable_in_pos")})
show_action(r)

separator("  [meas] Voltage at ctrl_cable_in_neg — expect 0 V (ground)")
r = s.apply_action(MeasureVoltage(), {"subject": s.component("ctrl_cable_in_neg")})
show_action(r)

# Conclusion so far: 12 V arrives at the control cube inputs but 0 V at outputs.
# The fault is inside the control cube. Invert the cube to inspect internals.

# ── Invert the control cube ───────────────────────────────────────────────────

separator("  [action] Invert control cube to expose internals")
r = s.apply_action(InvertEnclosure(), {"subject": s.component("cube_ctrl")})
show_action(r)

# ── Inspect connections on the control switch ─────────────────────────────────

separator("  [inspect] InspectConnections on ctrl_switch — expect missing cable on p-port")
r = s.apply_action(InspectConnections(), {"subject": s.component("ctrl_switch")})
show_action(r)

# The inspection reveals that no cable is plugged into the switch's input port —
# ctrl_cable_in_pos is dangling. Root cause identified.

# ── Repair: reconnect the detached cable ──────────────────────────────────────

separator("  [repair] Reconnect ctrl_cable_in_pos to switch input")
r = s.apply_action(
    ReconnectCable(connections={"n": node_ctrl_in_pos}),
    {"subject": s.component("ctrl_cable_in_pos")},
)
show_action(r)

separator("  [verify] System state after repair")
show_r = s.last_result
lit = show_r.is_lit("main_bulb")
print(f"  lamp is {'ON ✓' if lit else 'OFF ✗'} — fault {'resolved' if lit else 'NOT resolved'}")


# ── Simulation dump ───────────────────────────────────────────────────────────
from diagnosable_systems_simulation.utils.dump import dump_electrical, dump_state

nominal_system = build_three_cubes_system(backend=BACKEND, extra_tools={"multimeter"})
nominal_system.simulate()

separator("2.1  Raw electrical dump — nominal system")
print(dump_electrical(nominal_system.last_result, nominal_system.graph))

separator("2.2  Full component state report — nominal system")
print(dump_state(nominal_system))
