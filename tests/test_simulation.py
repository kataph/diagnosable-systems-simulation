"""
Simulation tests — PySpiceBackend only.

Covers:
  - Nominal operation (light states, voltages, currents)
  - Post-fix assertions: LED lit, current sign, LED current non-zero
  - All S0–S5 fault scenarios
  - Switch toggle, cable disconnect/reconnect

Run:
    python -m pytest tests/test_simulation.py -v
"""
import logging
import math
import pytest

from diagnosable_systems_simulation.electrical_simulation.backend.spice import PySpiceBackend
from diagnosable_systems_simulation.systems.base_system import DiagnosableSystem
from diagnosable_systems_simulation.systems.three_cubes.factory import build_three_cubes_system
from diagnosable_systems_simulation.systems.ten_cubes.factory import build_ten_cubes_system
from diagnosable_systems_simulation.actions.diagnostic_actions import CloseSwitch, InvertEnclosure, MeasureVoltage, ObserveComponent, OpenSwitch, ReplaceComponent, TestContinuity, TestControlSubchain, TestPathContinuity
from diagnosable_systems_simulation.actions.fault_actions import (
    DegradeComponent, DisconnectCable, ForceSwitch, ReconnectCable, ShortCircuit,
    SwapCablePolarities,
)


@pytest.fixture(scope="module")
def backend():
    return PySpiceBackend()


@pytest.fixture(scope="module")
def nominal(backend):
    s = build_three_cubes_system(backend=backend, extra_tools={"multimeter"})
    s.simulate()
    return s


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fresh(backend):
    s = build_three_cubes_system(backend=backend, extra_tools={"multimeter"})
    s.simulate()
    return s


# ---------------------------------------------------------------------------
# 1. Convergence
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_nominal_converges(self, nominal):
        assert nominal.last_result.converged

    def test_no_warnings_nominal(self, nominal):
        assert nominal.last_result.warnings == ()


# ---------------------------------------------------------------------------
# 2. Nominal light states
# ---------------------------------------------------------------------------

class TestNominalLights:
    def test_psu_green_led_lit(self, nominal):
        assert nominal.last_result.is_lit("psu_green_led"), \
            "PSU green LED must be ON in nominal state"

    def test_main_bulb_lit(self, nominal):
        assert nominal.last_result.is_lit("main_bulb"), \
            "Main bulb must be ON in nominal state"

    def test_internal_bulb_lit(self, nominal):
        assert nominal.last_result.is_lit("internal_bulb"), \
            "Internal bulb must be ON in nominal state"

    def test_ctrl_red_led_off(self, nominal):
        assert not nominal.last_result.is_lit("ctrl_red_led"), \
            "Control red LED must be OFF in nominal state (no polarity inversion)"


# ---------------------------------------------------------------------------
# 3. Nominal voltages
# ---------------------------------------------------------------------------

class TestNominalVoltages:
    def test_psu_source_voltage(self, nominal):
        psu_pos_node = nominal.graph.nodes_of("battery")["pos"]
        v = nominal.last_result.voltage(psu_pos_node)
        assert v is not None
        assert 11.0 < v <= 12.01, f"PSU positive rail should be near 12 V, got {v:.3f}"

    def test_ground_is_zero(self, nominal):
        gnd = nominal.last_result.voltage(nominal.graph.ground_node().node_id)
        assert gnd == 0.0

    def test_main_bulb_positive_terminal_above_ground(self, nominal):
        action = MeasureVoltage()
        result = nominal.apply_action(action, {"subject": nominal.component("main_bulb")})
        assert result.success
        props = {p.name: p.value for p in result.observation.properties}
        assert props["voltage_p"] > 1.0, "Bulb p-terminal should be above ground"
        assert math.isclose(props["voltage_n"], 0.0, abs_tol=0.1), "Bulb n-terminal should be near ground"

    def test_psu_green_led_anode_above_cathode(self, nominal):
        action = MeasureVoltage()
        result = nominal.apply_action(action, {"subject": nominal.component("psu_green_led")})
        assert result.success
        props = {p.name: p.value for p in result.observation.properties}
        assert props["voltage_anode"] > props["voltage_cathode"], \
            "LED anode must be above cathode when forward biased"


# ---------------------------------------------------------------------------
# 4. Nominal currents (post port-name fix: all must be positive)
# ---------------------------------------------------------------------------

class TestNominalCurrents:
    def test_main_bulb_current_positive(self, nominal):
        i = nominal.last_result.current("main_bulb")
        assert i is not None
        assert i > 0, f"Bulb current must be positive (V_p > V_n), got {i:.6f} A"

    def test_psu_green_led_current_nonzero(self, nominal):
        """Regression: spice.py used to return i=0 for LEDs."""
        i = nominal.last_result.current("psu_green_led")
        assert i is not None
        assert abs(i) > 1e-4, \
            f"Green LED current must be non-zero when forward biased, got {i:.6e} A"

    def test_load_diode_current_nonzero(self, nominal):
        """Regression: spice.py used to return i=0 for Diodes."""
        i = nominal.last_result.current("load_diode")
        assert i is not None
        assert abs(i) > 1e-4, \
            f"Load diode current must be non-zero when forward biased, got {i:.6e} A"

    def test_power_positive_for_lit_components(self, nominal):
        for cid in ("main_bulb", "psu_green_led"):
            pwr = nominal.last_result.component_power.get(cid, 0.0)
            assert pwr > 0.0, f"{cid} power should be positive, got {pwr}"


# ---------------------------------------------------------------------------
# 5. Switch open/close
# ---------------------------------------------------------------------------

class TestSwitchToggle:
    def test_open_switch_turns_lamp_off(self, backend):
        s = _fresh(backend)
        assert s.last_result.is_lit("main_bulb"), "Lamp must be ON before opening switch"
        s.apply_action(OpenSwitch(), {"subject": s.component("ctrl_switch")})
        assert not s.last_result.is_lit("main_bulb"), "Lamp must be OFF after opening switch"

    def test_open_switch_green_stays_on(self, backend):
        s = _fresh(backend)
        s.apply_action(OpenSwitch(), {"subject": s.component("ctrl_switch")})
        assert s.last_result.is_lit("psu_green_led"), \
            "PSU green LED must stay ON after opening switch"

    def test_open_then_close_restores_lamp(self, backend):
        s = _fresh(backend)
        s.apply_action(OpenSwitch(), {"subject": s.component("ctrl_switch")})
        s.apply_action(CloseSwitch(), {"subject": s.component("ctrl_switch")})
        assert s.last_result.is_lit("main_bulb"), "Lamp must be ON again after open then close"

    def test_open_switch_idempotent(self, backend):
        s = _fresh(backend)
        s.apply_action(OpenSwitch(), {"subject": s.component("ctrl_switch")})
        result = s.apply_action(OpenSwitch(), {"subject": s.component("ctrl_switch")})
        assert result.success, "Opening an already-open switch must succeed"
        assert not s.last_result.is_lit("main_bulb")

    def test_close_switch_idempotent(self, backend):
        s = _fresh(backend)
        result = s.apply_action(CloseSwitch(), {"subject": s.component("ctrl_switch")})
        assert result.success, "Closing an already-closed switch must succeed"
        assert s.last_result.is_lit("main_bulb")


# ---------------------------------------------------------------------------
# 6. Fault scenarios S0–S5
# ---------------------------------------------------------------------------

class TestFaultScenarios:
    """
    Expected (green, red, lamp):
      S0  cable detached from switch      (True,  False, False)
      S1  burned bulb filament            (True,  False, False)
      S2  battery depleted                (False, False, False)
      S3  battery reversed                (False, True,  False)
      S4  crossed wires                   (True,  True,  False)
      S5  switch stuck open               (True,  False, False)
    """

    def _lights(self, s):
        r = s.last_result
        return r.is_lit("psu_green_led"), r.is_lit("ctrl_red_led"), r.is_lit("main_bulb")

    def test_S0_cable_detached(self, backend):
        s = _fresh(backend)
        s.inject_fault(DisconnectCable(port_names=["n"]),
                       {"subject": s.component("ctrl_cable_in_pos")})
        assert self._lights(s) == (True, False, False), \
            f"S0: got {self._lights(s)}"

    def test_S1_burned_bulb(self, backend):
        s = _fresh(backend)
        s.inject_fault(DegradeComponent({"resistance": 1e9}),
                       {"subject": s.component("main_bulb")})
        assert self._lights(s) == (True, False, False), \
            f"S1: got {self._lights(s)}"

    def test_S2_battery_depleted(self, backend):
        s = _fresh(backend)
        s.inject_fault(DegradeComponent({"voltage": 0.0}),
                       {"subject": s.component("battery")})
        assert self._lights(s) == (False, False, False), \
            f"S2: got {self._lights(s)}"

    def test_S3_battery_reversed(self, backend):
        s = _fresh(backend)
        s.inject_fault(DegradeComponent({"voltage": -12.0}),
                       {"subject": s.component("battery")})
        assert self._lights(s) == (False, True, False), \
            f"S3: got {self._lights(s)}"

    def test_S4_crossed_wires(self, backend):
        s = _fresh(backend)
        pos_n = s.graph.nodes_of("ctrl_cable_in_pos")["n"]
        neg_n = s.graph.nodes_of("ctrl_cable_in_neg")["n"]
        s.inject_fault(DisconnectCable(port_names=["n"]),
                       {"subject": s.component("ctrl_cable_in_pos")})
        s.inject_fault(DisconnectCable(port_names=["n"]),
                       {"subject": s.component("ctrl_cable_in_neg")})
        s.inject_fault(ReconnectCable(connections={"n": neg_n}),
                       {"subject": s.component("ctrl_cable_in_pos")})
        s.inject_fault(ReconnectCable(connections={"n": pos_n}),
                       {"subject": s.component("ctrl_cable_in_neg")})
        assert self._lights(s) == (True, True, False), \
            f"S4: got {self._lights(s)}"

    def test_S5_switch_stuck_open(self, backend):
        s = _fresh(backend)
        s.inject_fault(ForceSwitch(is_closed=False),
                       {"subject": s.component("ctrl_switch")})
        assert self._lights(s) == (True, False, False), \
            f"S5: got {self._lights(s)}"


# ---------------------------------------------------------------------------
# 7. Cable disconnect / reconnect round-trip
# ---------------------------------------------------------------------------

class TestCableRoundTrip:
    def test_disconnect_kills_lamp_reconnect_restores(self, backend):
        s = _fresh(backend)
        node = s.graph.nodes_of("psu_cable_pos")["n"]
        assert s.last_result.is_lit("main_bulb"), "Lamp ON before disconnect"

        s.inject_fault(DisconnectCable(port_names=["n"]),
                       {"subject": s.component("psu_cable_pos")})
        assert not s.last_result.is_lit("main_bulb"), "Lamp OFF after disconnect"

        s.inject_fault(ReconnectCable(connections={"n": node}),
                       {"subject": s.component("psu_cable_pos")})
        assert s.last_result.is_lit("main_bulb"), "Lamp ON after reconnect"


# ---------------------------------------------------------------------------
# 8. TestContinuity on a disconnected cable
# ---------------------------------------------------------------------------

class TestContinuityDisconnectedCable:
    """
    A continuity test on a cable with a floating port must:
      - Report the cable's own resistance (nominal, ~0 Ω) — NOT open circuit.
        A technician probing both physical ends of an intact cable measures
        the cable itself, regardless of whether it is plugged into the circuit.
      - Surface a NEARBY ANOMALY warning about the floating port, because
        a technician at that location would physically notice the dangling end.
    """

    def test_floating_cable_reads_nominal_resistance(self, backend):
        s = build_three_cubes_system(backend=backend, extra_tools={"multimeter"})
        s.inject_fault(DisconnectCable(port_names=["p"]),
                       {"subject": s.component("ctrl_cable_out_pos")})

        cable = s.component("ctrl_cable_out_pos")
        result = s.apply_action(TestContinuity(), {"subject": cable})

        assert result.success, f"TestContinuity failed: {result.message}"
        props = {p.name: p.value for p in result.observation.properties}
        assert props.get("status") != "open circuit", (
            "Disconnected cable should NOT report open circuit — "
            "the cable itself is intact; only its circuit connection is broken."
        )

    def test_floating_cable_triggers_anomaly_warning(self, backend):
        s = build_three_cubes_system(backend=backend, extra_tools={"multimeter"})
        s.inject_fault(DisconnectCable(port_names=["p"]),
                       {"subject": s.component("ctrl_cable_out_pos")})

        cable = s.component("ctrl_cable_out_pos")
        result = s.apply_action(TestContinuity(), {"subject": cable})

        assert result.success, f"TestContinuity failed: {result.message}"
        assert "NEARBY ANOMALY" in result.message, (
            "Continuity test on a cable with a floating port must include a "
            "NEARBY ANOMALY warning — technician would physically see the dangling end."
        )
        
        
# ---------------------------------------------------------------------------
# 8. Test scenario 11
# ---------------------------------------------------------------------------

class TestScenarioDisconnectionIn10Cubes:
    """
    Nomen omen
    """
    
    def _observe_bulb(selfm, sim) -> None:
        obs_result = sim.apply_action(
        ObserveComponent(),
        {"subject": sim.component("main_bulb")},
        )
        print(f"[observe bulb]    {obs_result}")
        assert obs_result.success, f"ObserveComponent failed: {obs_result.message}"
        
        obs_result = sim.apply_action(
        MeasureVoltage(),
        {"subject": sim.component("main_bulb")},
        )
        print(f"[measure bulb]    {obs_result}")
        assert obs_result.success, f"Measure failed: {obs_result.message}"

        print("\n=== observation record ===")
        if obs_result.observation is not None:
            for prop in obs_result.observation.properties:
                unit = f" {prop.unit}" if prop.unit else ""
                print(f"  {prop.name}: {prop.value}{unit}")
        else:
            print("  (no observation record)")
            
    def _make_stdout_logger(self, name: str = "SpiceRunner") -> logging.Logger:
        import sys
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
            logger.addHandler(handler)
        return logger

    def test(self, backend):
        s = build_ten_cubes_system(backend=backend, extra_tools={'multimeter'})
        s.inject_fault(DisconnectCable(port_names=["n"]), {"subject": s.component("ctrl3_cable_in_pos")})
        
        s.add_logger(self._make_stdout_logger())
        result = s.apply_action(TestControlSubchain(), {"source":s.component('cube_ctrl1'), "sink":s.component('cube_ctrl2')})
        self._observe_bulb(s)
        assert "lamp is ON" in result.message
        
        s = build_ten_cubes_system(backend=backend, extra_tools={'multimeter'})
        s.inject_fault(DisconnectCable(port_names=["n"]), {"subject": s.component("ctrl3_cable_in_pos")})
        
        for i in range(1, 9):
            s.remove_component(f"ctrl{i}_green_led")
            s.remove_component(f"ctrl{i}_green_resistor")
        
        s.add_logger(self._make_stdout_logger())
        result = s.apply_action(TestControlSubchain(), {"source":s.component('cube_ctrl1'), "sink":s.component('cube_ctrl2')})
        self._observe_bulb(s)
        assert "lamp is ON" in result.message
        
        # s = build_ten_cubes_system(backend=backend, extra_tools={'multimeter'})
        # s.inject_fault(DisconnectCable(port_names=["n"]), {"subject": s.component("ctrl6_cable_in_pos")})
        
        # for i in range(1, 9):
        #     s.remove_component(f"ctrl{i}_green_led")
        #     s.remove_component(f"ctrl{i}_green_resistor")

        # result = s.apply_action(TestControlSubchain(), {"source":s.component('cube_ctrl5'), "sink":s.component('cube_ctrl8')})
        # assert "lamp is OFF" in result.message
        # result = s.apply_action(TestControlSubchain(), {"source":s.component('cube_ctrl7'), "sink":s.component('cube_ctrl8')})
        # assert "lamp is ON" in result.message
        # result = s.apply_action(TestControlSubchain(), {"source":s.component('cube_ctrl6'), "sink":s.component('cube_ctrl6')})
        # assert "lamp is OFF" in result.message
        # result = s.apply_action(TestControlSubchain(), {"source":s.component('cube_ctrl1'), "sink":s.component('cube_ctrl5')})
        # assert "lamp is ON" in result.message
        # result = s.apply_action(TestControlSubchain(), {"source":s.component('cube_ctrl1'), "sink":s.component('cube_ctrl1')})
        # assert "lamp is ON" in result.message
    
    
# ---------------------------------------------------------------------------
# 8. Test scenario 14
# ---------------------------------------------------------------------------

def _observe_bulb(sim) -> None:
        obs_result = sim.apply_action(
        ObserveComponent(),
        {"subject": sim.component("main_bulb")},
        )
        print(f"[observe bulb]    {obs_result}")
        assert obs_result.success, f"ObserveComponent failed: {obs_result.message}"
        
        obs_result = sim.apply_action(
        MeasureVoltage(),
        {"subject": sim.component("main_bulb")},
        )
        print(f"[measure bulb]    {obs_result}")
        assert obs_result.success, f"Measure failed: {obs_result.message}"

        print("\n=== observation record ===")
        if obs_result.observation is not None:
            for prop in obs_result.observation.properties:
                unit = f" {prop.unit}" if prop.unit else ""
                print(f"  {prop.name}: {prop.value}{unit}")
        else:
            print("  (no observation record)")
            
def _apply(sys: DiagnosableSystem, action, targets: dict) -> None:
    """Apply a fault action and raise if the system rejects it."""
    result = sys.apply_action(action, targets)
    if not result.success:
        raise RuntimeError(
            f"Fault injection failed [{action.action_id}]: {result.message}"
        )


class TestShort():
    """
    Nomen omen
    """
    def _make_stdout_logger(self, name: str = "SpiceRunner") -> logging.Logger:
        import sys
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
            logger.addHandler(handler)
        return logger
    
    def _short_psu_output_and_discharge(self, sys: DiagnosableSystem) -> None:
        """
        Short the PSU output cables together, then mark the battery as discharged.

        The short collapses the psu_pos net to ground; combined with the discharged
        battery, replacing the battery alone will not restore function.
        """
        cable_pos = sys.component("psu_cable_pos")
        cable_neg = sys.component("psu_cable_neg")
        psu_pos_node = cable_pos.port("p").node_id   # psu_pos net
        gnd_node = cable_neg.port("p").node_id       # ground net
        _apply(sys, ShortCircuit(psu_pos_node, gnd_node, "psu_output_short"), {})
        _apply(sys, DegradeComponent({"voltage": 0.0}), {"subject": sys.component("battery")})
    
    def _short_psu_output(self, sys: DiagnosableSystem) -> None:
        """
        Short the PSU output cables together, then mark the battery as discharged.

        The short collapses the psu_pos net to ground; combined with the discharged
        battery, replacing the battery alone will not restore function.
        """
        cable_pos = sys.component("psu_cable_pos")
        cable_neg = sys.component("psu_cable_neg")
        psu_pos_node = cable_pos.port("p").node_id   # psu_pos net
        gnd_node = cable_neg.port("p").node_id       # ground net
        _apply(sys, ShortCircuit(psu_pos_node, gnd_node, "psu_output_short"), {})



    def test(self, backend):
        s = build_ten_cubes_system(backend=backend, extra_tools={'multimeter'})
        
        s.add_logger(self._make_stdout_logger())
        self._short_psu_output(s)
        
        
        _observe_bulb(s)
        
        obs_result = s.apply_action(
        InvertEnclosure(),
        {"subject":s.component('cube_psu')})
        
        print(f"[invert psu cube]    {obs_result}")
        
        obs_result = s.apply_action(
        TestPathContinuity(),
        {"source":s.component('battery'), "sink":s.component('psu_cable_pos')})
        
        print(f"[test continuity]    {obs_result}")
        assert 'short' in obs_result.message
        
        obs_result = s.apply_action(
        TestPathContinuity(),
        {"source":s.component('battery'), "sink":s.component('psu_cable_neg')})
        
        print(f"[test continuity]    {obs_result}")
        assert 'short' in obs_result.message
        
        
        

class TestRelayBehavior:
    """
    Verify that the ambient-light-sensor relay correctly tracks sensor state:

      - Not excited (sensor dark, no feedback): relay closed.
      - Excited (lamp on → sensor lit, one coupling pass): relay open.

    "Excited" here means the sensor has received enough light to activate the
    relay coil (normally-closed contacts → open).  The full feedback loop
    oscillates and never converges; we therefore test the *first coupling pass*
    in isolation: solve the circuit with the relay closed (lamp on), apply the
    coupling once, and confirm the relay flipped open.
    """

    def test_relay_closed_when_not_excited(self, backend):
        """
        Default state: als_feedback disabled, sensor dark → relay closed.
        """
        from diagnosable_systems_simulation.systems.ambient_light_sensor.factory import build_ambient_light_system
        s = build_ambient_light_system(backend=backend)
        # als_feedback=False: sensor is never illuminated → relay stays closed.
        s.simulate()
        relay = s.component("ctrl_relay")
        assert relay.current_parameters()["is_closed"] is True, (
            "Relay should be CLOSED when sensor is dark (not excited)"
        )

    def test_relay_opens_when_excited(self, backend):
        """
        First coupling pass: lamp on → sensor lit → relay must open.

        The full feedback loop oscillates (lamp→sensor→relay→lamp→…) and never
        reaches a fixed point.  We therefore drive one coupling pass explicitly:
        solve the circuit (relay closed → lamp on), apply the coupling once, and
        assert the relay is now open.  This is exactly what happens in iteration 0
        of the runner loop and is the behaviour the diagnostic agent relies on.
        """
        from diagnosable_systems_simulation.systems.ambient_light_sensor.factory import build_ambient_light_system
        s = build_ambient_light_system(backend=backend)
        s.context.extra["als_feedback"] = True

        # One solve with relay initially closed → lamp is on.
        result = s._runner.backend.solve(s._graph)
        assert result.is_lit("main_bulb"), "Lamp should be on with relay closed"

        # Apply the coupling once: lamp lit → sensor lit → relay should open.
        coupling = s._runner.couplings[0]
        coupling.apply(result, s._graph, s._context)

        relay = s.component("ctrl_relay")
        assert relay.current_parameters()["is_closed"] is False, (
            "Relay should be OPEN after sensor is illuminated (excited)"
        )


# ---------------------------------------------------------------------------
# Current sensor system tests
# ---------------------------------------------------------------------------

from diagnosable_systems_simulation.systems.current_sensor.factory import build_current_sensor_system


def _fresh_cs(backend):
    s = build_current_sensor_system(backend=backend, extra_tools={"multimeter"})
    s.simulate()
    return s


class TestCurrentSensorNominal:
    """Nominal state: relay closed, all bulbs and indicator LED lit."""

    def test_converges(self, backend):
        s = _fresh_cs(backend)
        assert s.last_result.converged

    def test_no_warnings(self, backend):
        s = _fresh_cs(backend)
        assert s.last_result.warnings == ()

    def test_main_bulb_lit(self, backend):
        s = _fresh_cs(backend)
        assert s.last_result.is_lit("main_bulb")

    def test_internal_bulb_lit(self, backend):
        s = _fresh_cs(backend)
        assert s.last_result.is_lit("internal_bulb")

    def test_psu_green_led_lit(self, backend):
        s = _fresh_cs(backend)
        assert s.last_result.is_lit("psu_green_led")

    def test_ctrl_green_led_lit(self, backend):
        # Indicator LED is on when relay is closed and current flows
        s = _fresh_cs(backend)
        assert s.last_result.is_lit("ctrl_green_led")

    def test_relay_closed_nominally(self, backend):
        s = _fresh_cs(backend)
        relay = s.component("ctrl_relay")
        assert relay.current_parameters()["is_closed"] is True


class TestCurrentSensorFaults:

    def test_relay_stuck_open_kills_lamp(self, backend):
        """Stuck-open relay: 0V return path broken → lamp off, PSU LED still on."""
        s = _fresh_cs(backend)
        s.inject_fault(ForceSwitch(is_closed=False), {"subject": s.component("ctrl_relay")})
        r = s.last_result
        assert not r.is_lit("main_bulb")
        assert not r.is_lit("internal_bulb")
        assert r.is_lit("psu_green_led")

    def test_relay_stuck_open_repair(self, backend):
        """Repair stuck-open relay by removing the fault overlay."""
        s = _fresh_cs(backend)
        s.inject_fault(ForceSwitch(is_closed=False), {"subject": s.component("ctrl_relay")})
        assert not s.last_result.is_lit("main_bulb")
        # Repair: clear fault overlay and re-simulate
        s.component("ctrl_relay")._fault_overlay.clear()
        s.simulate()
        assert s.last_result.is_lit("main_bulb")

    def test_burned_bulb_kills_main_lamp(self, backend):
        """Burned main bulb (open circuit): lamp off, PSU and ctrl LEDs still on."""
        s = _fresh_cs(backend)
        s.inject_fault(DegradeComponent({"resistance": 1e9}), {"subject": s.component("main_bulb")})
        r = s.last_result
        assert not r.is_lit("main_bulb")
        assert r.is_lit("psu_green_led")

    def test_burned_bulb_repair(self, backend):
        """Repair burned bulb by restoring nominal resistance."""
        s = _fresh_cs(backend)
        s.inject_fault(DegradeComponent({"resistance": 1e9}), {"subject": s.component("main_bulb")})
        assert not s.last_result.is_lit("main_bulb")
        s.component("main_bulb")._fault_overlay.clear()
        s.simulate()
        assert s.last_result.is_lit("main_bulb")

    def test_depleted_battery_kills_everything(self, backend):
        """Depleted battery: all lights off."""
        s = _fresh_cs(backend)
        s.inject_fault(DegradeComponent({"voltage": 0.0}), {"subject": s.component("battery")})
        r = s.last_result
        assert not r.is_lit("main_bulb")
        assert not r.is_lit("psu_green_led")
        assert not r.is_lit("ctrl_green_led")

    def test_depleted_battery_repair(self, backend):
        """Repair depleted battery."""
        s = _fresh_cs(backend)
        s.inject_fault(DegradeComponent({"voltage": 0.0}), {"subject": s.component("battery")})
        assert not s.last_result.is_lit("main_bulb")
        s.component("battery")._fault_overlay.clear()
        s.simulate()
        assert s.last_result.is_lit("main_bulb")

    def test_lamp_short_opens_relay(self, backend):
        """
        Short-circuit across the load (very low resistance) → overcurrent →
        relay opens → lamp goes off and circuit is non-converging (oscillating
        protection loop) or relay stably opens.

        Either converged=False (oscillation) or lamp off with relay open
        both represent correct overcurrent-protection behaviour.
        """
        s = _fresh_cs(backend)
        s.inject_fault(DegradeComponent({"resistance": 0.5}), {"subject": s.component("main_bulb")})
        r = s.last_result
        relay = s.component("ctrl_relay")
        relay_open = not relay.current_parameters()["is_closed"]
        lamp_off = not r.is_lit("main_bulb")
        # Either the relay tripped (opened) or the loop didn't converge — both correct
        assert relay_open or not r.converged, (
            f"Overcurrent should open relay or cause non-convergence; "
            f"relay_open={relay_open}, converged={r.converged}"
        )
        assert lamp_off or not r.converged

    def test_psu_cable_detached_kills_lamp(self, backend):
        """Detaching the PSU output positive cable cuts power to everything downstream."""
        s = _fresh_cs(backend)
        s.inject_fault(DisconnectCable(port_names=["n"]), {"subject": s.component("psu_cable_pos")})
        assert not s.last_result.is_lit("main_bulb")
        assert not s.last_result.is_lit("ctrl_green_led")

    def test_psu_cable_detached_repair(self, backend):
        """Reconnecting the PSU positive cable restores the lamp."""
        s = _fresh_cs(backend)
        node = s.graph.nodes_of("psu_cable_pos")["n"]
        s.inject_fault(DisconnectCable(port_names=["n"]), {"subject": s.component("psu_cable_pos")})
        assert not s.last_result.is_lit("main_bulb")
        s.inject_fault(ReconnectCable(connections={"n": node}), {"subject": s.component("psu_cable_pos")})
        assert s.last_result.is_lit("main_bulb")


# ---------------------------------------------------------------------------
# Asymmetric chains system tests
# ---------------------------------------------------------------------------

from diagnosable_systems_simulation.systems.asymmetric_chains.factory import build_asymmetric_chains_system


def _fresh_ac(backend):
    s = build_asymmetric_chains_system(backend=backend, extra_tools={"multimeter"})
    s.simulate()
    return s


class TestAsymmetricChainsNominal:
    """Nominal: both loads lit, all five indicator LEDs lit."""

    def test_converges(self, backend):
        s = _fresh_ac(backend)
        assert s.last_result.converged

    def test_no_warnings(self, backend):
        s = _fresh_ac(backend)
        assert s.last_result.warnings == ()

    def test_load1_main_bulb_lit(self, backend):
        s = _fresh_ac(backend)
        assert s.last_result.is_lit("load1_main_bulb")

    def test_load2_main_bulb_lit(self, backend):
        s = _fresh_ac(backend)
        assert s.last_result.is_lit("load2_main_bulb")

    def test_psu1_green_led_lit(self, backend):
        s = _fresh_ac(backend)
        assert s.last_result.is_lit("psu1_psu_green_led")

    def test_psu2_green_led_lit(self, backend):
        s = _fresh_ac(backend)
        assert s.last_result.is_lit("psu2_psu_green_led")

    def test_ctrl1_green_led_lit(self, backend):
        s = _fresh_ac(backend)
        assert s.last_result.is_lit("ctrl1_green_led")

    def test_ctrl2_green_led_lit(self, backend):
        s = _fresh_ac(backend)
        assert s.last_result.is_lit("ctrl2_green_led")

    def test_ctrl3_green_led_lit(self, backend):
        s = _fresh_ac(backend)
        assert s.last_result.is_lit("ctrl3_green_led")


class TestAsymmetricChainsFaults:

    def test_ctrl1_switch_open_kills_load1_only(self, backend):
        """
        Opening CTRL1 switch breaks the only path to LOAD1.
        LOAD2 still receives power via PSU2→CTRL2→CTRL3→LOAD2,
        so it stays lit.
        """
        s = _fresh_ac(backend)
        s.apply_action(OpenSwitch(), {"subject": s.component("ctrl1_switch")})
        r = s.last_result
        assert not r.is_lit("load1_main_bulb"), "LOAD1 must be OFF when CTRL1 switch opens"
        assert r.is_lit("load2_main_bulb"), "LOAD2 must stay ON (independent chain via CTRL2/3)"

    def test_ctrl1_switch_open_repair(self, backend):
        s = _fresh_ac(backend)
        s.apply_action(OpenSwitch(), {"subject": s.component("ctrl1_switch")})
        assert not s.last_result.is_lit("load1_main_bulb")
        s.apply_action(CloseSwitch(), {"subject": s.component("ctrl1_switch")})
        assert s.last_result.is_lit("load1_main_bulb")

    def test_ctrl3_switch_open_kills_load2_only(self, backend):
        """
        Opening CTRL3 switch breaks the path to LOAD2.
        LOAD1 is not affected (receives power via PSU1→CTRL1→LOAD1).
        """
        s = _fresh_ac(backend)
        s.apply_action(OpenSwitch(), {"subject": s.component("ctrl3_switch")})
        r = s.last_result
        assert not r.is_lit("load2_main_bulb"), "LOAD2 must be OFF when CTRL3 switch opens"
        assert r.is_lit("load1_main_bulb"), "LOAD1 must stay ON"

    def test_ctrl3_switch_open_repair(self, backend):
        s = _fresh_ac(backend)
        s.apply_action(OpenSwitch(), {"subject": s.component("ctrl3_switch")})
        assert not s.last_result.is_lit("load2_main_bulb")
        s.apply_action(CloseSwitch(), {"subject": s.component("ctrl3_switch")})
        assert s.last_result.is_lit("load2_main_bulb")

    def test_psu1_battery_depleted_load1_still_on(self, backend):
        """
        PSU1 battery dead: LOAD1 can still receive power via PSU2→diode→CTRL1→LOAD1
        cross-link, so LOAD1 stays lit (cross-chain redundancy).
        PSU1 green LED goes off; PSU2, CTRL LEDs stay on.
        """
        s = _fresh_ac(backend)
        s.inject_fault(DegradeComponent({"voltage": 0.0}), {"subject": s.component("psu1_battery")})
        r = s.last_result
        assert not r.is_lit("psu1_psu_green_led"), "PSU1 LED must go off with depleted battery"
        assert r.is_lit("load1_main_bulb"), "LOAD1 must stay ON via PSU2 cross-link"
        assert r.is_lit("load2_main_bulb"), "LOAD2 must stay ON"

    def test_psu1_battery_depleted_repair(self, backend):
        s = _fresh_ac(backend)
        s.inject_fault(DegradeComponent({"voltage": 0.0}), {"subject": s.component("psu1_battery")})
        assert not s.last_result.is_lit("psu1_psu_green_led")
        s.component("psu1_battery")._fault_overlay.clear()
        s.simulate()
        assert s.last_result.is_lit("psu1_psu_green_led")
        assert s.last_result.is_lit("load1_main_bulb")

    def test_both_batteries_depleted_kills_everything(self, backend):
        """With both PSUs dead, no path can power any load."""
        s = _fresh_ac(backend)
        s.inject_fault(DegradeComponent({"voltage": 0.0}), {"subject": s.component("psu1_battery")})
        s.inject_fault(DegradeComponent({"voltage": 0.0}), {"subject": s.component("psu2_battery")})
        r = s.last_result
        assert not r.is_lit("load1_main_bulb")
        assert not r.is_lit("load2_main_bulb")
        assert not r.is_lit("psu1_psu_green_led")
        assert not r.is_lit("psu2_psu_green_led")

    def test_load1_cable_detached_kills_load1_only(self, backend):
        """Detaching LOAD1 positive cable kills LOAD1; LOAD2 unaffected."""
        s = _fresh_ac(backend)
        s.inject_fault(DisconnectCable(port_names=["n"]), {"subject": s.component("load1_load_cable_pos")})
        r = s.last_result
        assert not r.is_lit("load1_main_bulb")
        assert r.is_lit("load2_main_bulb")

    def test_load1_cable_detached_repair(self, backend):
        s = _fresh_ac(backend)
        node = s.graph.nodes_of("load1_load_cable_pos")["n"]
        s.inject_fault(DisconnectCable(port_names=["n"]), {"subject": s.component("load1_load_cable_pos")})
        assert not s.last_result.is_lit("load1_main_bulb")
        s.inject_fault(ReconnectCable(connections={"n": node}), {"subject": s.component("load1_load_cable_pos")})
        assert s.last_result.is_lit("load1_main_bulb")

    def test_burned_load2_bulb(self, backend):
        """Burned LOAD2 main bulb: LOAD2 off, LOAD1 unaffected."""
        s = _fresh_ac(backend)
        s.inject_fault(DegradeComponent({"resistance": 1e9}), {"subject": s.component("load2_main_bulb")})
        r = s.last_result
        assert not r.is_lit("load2_main_bulb")
        assert r.is_lit("load1_main_bulb")

    def test_burned_load2_bulb_repair(self, backend):
        s = _fresh_ac(backend)
        s.inject_fault(DegradeComponent({"resistance": 1e9}), {"subject": s.component("load2_main_bulb")})
        assert not s.last_result.is_lit("load2_main_bulb")
        s.component("load2_main_bulb")._fault_overlay.clear()
        s.simulate()
        assert s.last_result.is_lit("load2_main_bulb")


# ---------------------------------------------------------------------------
# Cable repair — three paths for detached cable and loose connection
# ---------------------------------------------------------------------------
#
# "Detached cable": the cable's port is floating (DisconnectCable on the cable).
# "Loose connection": the neighbouring component has RECONNECTABLE affordance
#   and _detached_cable_ports set — the cable is the culprit but the agent
#   identifies the fault via the neighbour component.
#
# For each fault type we verify three independent repair paths:
#   (A) ReconnectCable   — direct reconnection action on the cable
#   (B) test_repair      — hypothesis verification with the cable ID
#   (C) ReplaceComponent — replace the cable with a fresh one (also reconnects)
#
# For the loose-connection case the same three paths are used; the only
# difference is that the fault is identified via the neighbour component but
# the cable is still the subject of the repair action.
# ---------------------------------------------------------------------------

class TestCableDetachedRepairPaths:
    """Three repair paths for a cable with a floating port."""

    CABLE = "ctrl_cable_in_pos"
    PORT  = "n"

    def _broken(self, backend):
        s = _fresh(backend)
        s.inject_fault(DisconnectCable(port_names=[self.PORT]),
                       {"subject": s.component(self.CABLE)})
        assert not s.last_result.is_lit("main_bulb"), "lamp must be off before repair"
        return s

    def test_repair_via_reconnect_cable(self, backend):
        """(A) ReconnectCable restores the system."""
        s = self._broken(backend)
        s.apply_action(ReconnectCable(), {"subject": s.component(self.CABLE)})
        assert s.is_system_nominal(), "lamp must be on after ReconnectCable"

    def test_repair_via_test_repair(self, backend):
        """(B) test_repair returns True when the cable is nominated."""
        s = self._broken(backend)
        assert s.test_repair({self.CABLE}), "test_repair must confirm cable repairs the system"

    def test_repair_via_replace_component(self, backend):
        """(C) ReplaceComponent on a cable reconnects floating ports."""
        s = self._broken(backend)
        s.apply_action(ReplaceComponent("spare_cable"),
                       {"subject": s.component(self.CABLE)})
        assert s.is_system_nominal(), "lamp must be on after ReplaceComponent"


class TestLooseConnectionRepairPaths:
    """
    Three repair paths when the fault is identified via the neighbour component
    (loose connection / _detached_cable_ports set on the switch).

    DisconnectCable sets RECONNECTABLE on every non-cable component that shared
    a node with the disconnected port.  Here ctrl_switch shares the node that
    ctrl_cable_in_pos.n was connected to, so it gets the loose-connection marker.
    The cable (ctrl_cable_in_pos) is still the actual repair target.
    """

    CABLE     = "ctrl_cable_in_pos"
    NEIGHBOUR = "ctrl_switch"
    PORT      = "n"

    def _broken(self, backend):
        s = _fresh(backend)
        s.inject_fault(DisconnectCable(port_names=[self.PORT]),
                       {"subject": s.component(self.CABLE)})
        assert not s.last_result.is_lit("main_bulb"), "lamp must be off before repair"
        neighbour = s.component(self.NEIGHBOUR)
        from diagnosable_systems_simulation.world.affordances import Affordance
        assert Affordance.RECONNECTABLE in neighbour.affordances.all_active(neighbour, s.context), \
            "neighbour must have RECONNECTABLE affordance after disconnect"
        return s

    def test_repair_via_reconnect_cable(self, backend):
        """(A) ReconnectCable on the cable clears the neighbour's loose-connection marker."""
        s = self._broken(backend)
        s.apply_action(ReconnectCable(), {"subject": s.component(self.CABLE)})
        assert s.is_system_nominal(), "lamp must be on after ReconnectCable"
        neighbour = s.component(self.NEIGHBOUR)
        from diagnosable_systems_simulation.world.affordances import Affordance
        assert Affordance.RECONNECTABLE not in neighbour.affordances.all_active(neighbour, s.context), \
            "RECONNECTABLE must be cleared from neighbour after reconnect"

    def test_repair_via_test_repair_cable(self, backend):
        """(B) test_repair with the cable ID confirms the repair."""
        s = self._broken(backend)
        assert s.test_repair({self.CABLE}), \
            "test_repair must confirm cable repairs the system"

    def test_repair_via_replace_component(self, backend):
        """(C) ReplaceComponent on the cable also clears the neighbour marker."""
        s = self._broken(backend)
        s.apply_action(ReplaceComponent("spare_cable"),
                       {"subject": s.component(self.CABLE)})
        assert s.is_system_nominal(), "lamp must be on after ReplaceComponent"
        neighbour = s.component(self.NEIGHBOUR)
        from diagnosable_systems_simulation.world.affordances import Affordance
        assert Affordance.RECONNECTABLE not in neighbour.affordances.all_active(neighbour, s.context), \
            "RECONNECTABLE must be cleared from neighbour after replace"


class TestReconnectCableAfterSwap:
    """ReconnectCable must clean up RECONNECTABLE on neighbours even when the
    cable is reconnected to a *different* node than it was disconnected from
    (as happens inside SwapCablePolarities)."""

    def test_swap_does_not_mark_psu_cables_reconnectable(self, backend):
        """After SwapCablePolarities, neighbouring PSU cables must NOT get RECONNECTABLE."""
        from diagnosable_systems_simulation.world.affordances import Affordance
        from diagnosable_systems_simulation.world.components import Cable
        s = _fresh_cs(backend)
        s.inject_fault(
            SwapCablePolarities(port_name="p"),
            {"cable_a": s.component("ctrl_cable_in_pos"),
             "cable_b": s.component("ctrl_cable_in_neg")},
        )
        for cid in ("psu_cable_pos", "psu_cable_neg"):
            c = s.component(cid)
            assert Affordance.RECONNECTABLE not in c.affordances.all_active(c, s.context), \
                f"{cid} must NOT be RECONNECTABLE after swap — it was not disconnected"

    def test_swap_marks_swapped_cables_wrong_node(self, backend):
        """After SwapCablePolarities, the two swapped cables must be detectable via wrong_node."""
        from diagnosable_systems_simulation.world.components import Cable
        s = _fresh_cs(backend)
        s.inject_fault(
            SwapCablePolarities(port_name="p"),
            {"cable_a": s.component("ctrl_cable_in_pos"),
             "cable_b": s.component("ctrl_cable_in_neg")},
        )
        def _is_wrong_node(comp):
            if not isinstance(comp, Cable):
                return False
            orig = getattr(comp, "_orig_connections", {})
            return any(
                p.is_connected() and orig.get(p.name) is not None and p.node_id != orig[p.name]
                for p in comp.ports
            )
        assert _is_wrong_node(s.component("ctrl_cable_in_pos")), \
            "ctrl_cable_in_pos must be wrong_node after swap"
        assert _is_wrong_node(s.component("ctrl_cable_in_neg")), \
            "ctrl_cable_in_neg must be wrong_node after swap"

    def test_swap_candidate_psu_cables_gives_wrong_not_partial(self, backend):
        """Verifying PSU cables as candidates must return WRONG (lamp stays off), not partial.

        Before the ReconnectCable fix, psu_cable_* appeared in still_broken_ids due to
        a spurious RECONNECTABLE affordance, causing a false PARTIAL outcome.
        """
        from diagnosable_systems_simulation.world.affordances import Affordance
        from diagnosable_systems_simulation.world.components import Cable
        s = _fresh_cs(backend)
        s.inject_fault(
            SwapCablePolarities(port_name="p"),
            {"cable_a": s.component("ctrl_cable_in_pos"),
             "cable_b": s.component("ctrl_cable_in_neg")},
        )
        s._fault_snapshot = s.snapshot()
        # PSU cables are not faulted — repairing them must not fix the lamp
        lamp_on = s.test_repair({"psu_cable_pos", "psu_cable_neg"})
        assert not lamp_on, \
            "repairing PSU cables must NOT restore lamp — they are not the fault"

    def test_swap_correct_cables_restores_lamp(self, backend):
        """Repairing the actually-swapped cables fully restores nominal state."""
        s = _fresh_cs(backend)
        s.inject_fault(
            SwapCablePolarities(port_name="p"),
            {"cable_a": s.component("ctrl_cable_in_pos"),
             "cable_b": s.component("ctrl_cable_in_neg")},
        )
        s._fault_snapshot = s.snapshot()
        lamp_on = s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg"})
        assert lamp_on, \
            "repairing the swapped cables must restore the lamp"


if __name__ == "__main__":
    TestShort().test(backend=PySpiceBackend())