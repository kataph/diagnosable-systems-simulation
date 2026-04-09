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
from diagnosable_systems_simulation.actions.diagnostic_actions import CloseSwitch, MeasureVoltage, ObserveComponent, OpenSwitch, TestContinuity, TestControlSubchain
from diagnosable_systems_simulation.actions.fault_actions import (
    DegradeComponent, DisconnectCable, ForceSwitch, ReconnectCable,
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

if __name__ == "__main__":
    TestScenarioDisconnectionIn10Cubes().test(backend=PySpiceBackend())