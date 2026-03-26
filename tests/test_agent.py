"""
Integration tests for the LLM agent (uses real LLM via TextClient / OpenAI backend).

Run:  python -m pytest tests/test_agent.py -v -s
"""
import pytest
from diagnosable_systems_simulation.electrical_simulation.backend.stub import StubBackend
from diagnosable_systems_simulation.electrical_simulation.backend.spice import PySpiceBackend
# BACKEND = StubBackend()
BACKEND = PySpiceBackend()
from diagnosable_systems_simulation.systems.three_cubes.factory import build_three_cubes_system
from nl_interface.interface import _instantiate, _parse, _verbalize, _REGISTRY


@pytest.fixture(scope="module")
def system():
    s = build_three_cubes_system(backend=BACKEND, extra_tools={"multimeter"})
    return s


def _show(label, value):
    print(f"\n{'─'*60}\n[{label}]\n{value}")


# ---------------------------------------------------------------------------
# text2action
# ---------------------------------------------------------------------------

def test_parse_multi_action(system):
    text = "measure voltage at the main bulb and toggle the control switch"
    result = _parse(text, system)
    _show("INPUT", text)
    _show("OUTPUT", result)
    assert isinstance(result, list) and len(result) >= 2
    for entry in result:
        assert entry["action_id"] in _REGISTRY, f"Unknown action_id: {entry['action_id']}"
        assert "subject" in entry


def test_parse_single_observe(system):
    text = "observe the green LED on the power supply"
    result = _parse(text, system)
    _show("INPUT", text)
    _show("OUTPUT", result)
    assert result[0]["action_id"] == "observe_component"
    assert result[0]["subject"] == "psu_green_led"


def test_parse_continuity_check(system):
    text = "test the continuity of the main bulb filament"
    result = _parse(text, system)
    _show("INPUT", text)
    _show("OUTPUT", result)
    assert result[0]["action_id"] == "test_continuity"
    assert result[0]["subject"] == "main_bulb"


# ---------------------------------------------------------------------------
# _instantiate (no LLM needed)
# ---------------------------------------------------------------------------

def test_instantiate_no_params():
    from diagnosable_systems_simulation.actions.diagnostic_actions import MeasureVoltage
    entry = {"action_id": "measure_voltage"}
    action = _instantiate(entry)
    _show("INPUT", entry)
    _show("OUTPUT", action)
    assert isinstance(action, MeasureVoltage)


def test_instantiate_with_params():
    from diagnosable_systems_simulation.actions.fault_actions import ForceSwitch
    entry = {"action_id": "force_switch", "params": {"is_closed": False}}
    action = _instantiate(entry)
    _show("INPUT", entry)
    _show("OUTPUT", f"ForceSwitch(is_closed={action.is_closed})")
    assert isinstance(action, ForceSwitch)
    assert action.is_closed is False


# ---------------------------------------------------------------------------
# output2verbalization
# ---------------------------------------------------------------------------

def test_verbalize_measure(system):
    from diagnosable_systems_simulation.actions.diagnostic_actions import MeasureVoltage
    action = MeasureVoltage()
    result = system.apply_action(action, {"subject": system.component("main_bulb")})
    _show("INPUT", f"action={action.action_id}, result.message={result.message!r}")
    out = _verbalize([(action, result)])
    _show("OUTPUT", out)
    assert isinstance(out, str) and len(out) > 0


def test_verbalize_failure(system):
    """Precondition not met (internal_bulb not yet observable) — failure verbalized."""
    from diagnosable_systems_simulation.actions.diagnostic_actions import MeasureVoltage
    action = MeasureVoltage()
    result = system.apply_action(action, {"subject": system.component("internal_bulb")})
    assert not result.success
    _show("INPUT", f"action={action.action_id}, result.message={result.message!r}")
    out = _verbalize([(action, result)])
    _show("OUTPUT", out)
    assert isinstance(out, str) and len(out) > 0


def test_verbalize_multi_action(system):
    from diagnosable_systems_simulation.actions.diagnostic_actions import ObserveComponent, MeasureVoltage
    pairs = []
    for cls, cid in [(ObserveComponent, "psu_green_led"), (MeasureVoltage, "main_bulb")]:
        action = cls()
        result = system.apply_action(action, {"subject": system.component(cid)})
        pairs.append((action, result))
    _show("INPUT", [(a.action_id, r.message, r) for a, r in pairs])
    out = _verbalize(pairs)
    _show("OUTPUT", out)
    assert isinstance(out, str) and len(out) > 0