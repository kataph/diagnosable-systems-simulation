"""
Execution-layer unit tests for nl_interface/interface.py.

Tests cover _instantiate, _expand_enclosure_targets, and the run() / _execute()
path without requiring a live LLM: _parse and _verbalize are patched where needed.
LLM-gated tests (decorated @requires_llm) exercise the real parse + verbalize path
and are skipped unless SKIP_LLM_TESTS=0 is set in the environment.

Run (no LLM):      python -m pytest tests/test_nl_interface_unit.py -v
Run (with LLM):    SKIP_LLM_TESTS=0 python -m pytest tests/test_nl_interface_unit.py -v
"""
import os
import unittest.mock
import pytest

from diagnosable_systems_simulation.electrical_simulation.backend.spice import PySpiceBackend
from diagnosable_systems_simulation.systems.three_cubes.factory import build_three_cubes_system
from nl_interface.interface import (
    _instantiate,
    _expand_enclosure_targets,
    _parse,
    _verbalize,
    run,
    _REGISTRY,
)

requires_llm = pytest.mark.skipif(
    os.environ.get("SKIP_LLM_TESTS", "1") == "1",
    reason="Live LLM tests skipped (set SKIP_LLM_TESTS=0 to enable)",
)


@pytest.fixture(scope="module")
def system():
    s = build_three_cubes_system(backend=PySpiceBackend(), extra_tools={"multimeter"})
    s.simulate()
    return s


# ---------------------------------------------------------------------------
# _instantiate
# ---------------------------------------------------------------------------

class TestInstantiate:
    def test_drops_hallucinated_params(self):
        from diagnosable_systems_simulation.actions.diagnostic_actions import MeasureVoltage
        entry = {"action_id": "measure_voltage", "params": {"hallucinated": "foo", "another_fake": 99}}
        action = _instantiate(entry)
        assert isinstance(action, MeasureVoltage)
        assert not hasattr(action, "hallucinated")
        assert not hasattr(action, "another_fake")

    def test_known_params_passed_through(self):
        from diagnosable_systems_simulation.actions.fault_actions import ForceSwitch
        entry = {"action_id": "force_switch", "params": {"is_closed": False}}
        action = _instantiate(entry)
        assert isinstance(action, ForceSwitch)
        assert action.is_closed is False


# ---------------------------------------------------------------------------
# _expand_enclosure_targets
# ---------------------------------------------------------------------------

class TestExpandEnclosureTargets:
    def test_source_sink_both_enclosures_unchanged(self, system):
        entries = [{"action_id": "test_path_continuity", "source": "cube_ctrl", "sink": "cube_load"}]
        assert _expand_enclosure_targets(entries, system) == entries


# ---------------------------------------------------------------------------
# run() — non-LLM (patched _parse + _verbalize)
# ---------------------------------------------------------------------------

class TestRunNoLLM:
    def test_applies_action_and_accumulates_cost(self, system):
        entries = [{"action_id": "measure_voltage", "subject": "main_bulb"}]
        with unittest.mock.patch("nl_interface.interface._parse", return_value=entries), \
             unittest.mock.patch("nl_interface.interface._verbalize", return_value="mock narrative"):
            narrative, cost, parsed, results = run("...", system, mode="collect_information")

        assert narrative == "mock narrative"
        assert cost.time > 0
        assert "multimeter" in cost.equipment
        assert parsed == entries
        assert len(results) == 1
        assert results[0][0].action_id == "measure_voltage"
        assert results[0][2].success is True

    def test_auto_access_opens_peephole_for_enclosed_component(self):
        fresh = build_three_cubes_system(backend=PySpiceBackend(), extra_tools={"multimeter"})
        fresh.simulate()
        entries = [{"action_id": "observe_component", "subject": "internal_bulb"}]
        with unittest.mock.patch("nl_interface.interface._parse", return_value=entries), \
             unittest.mock.patch("nl_interface.interface._verbalize", return_value="mock"):
            _, _, _, results = run("...", fresh, mode="collect_information")

        action_ids = [r[0].action_id for r in results]
        assert len(results) >= 2
        assert any(aid in ("open_peephole", "invert_enclosure", "open_inspection_panel") for aid in action_ids)
        # The observe_component itself must succeed after auto-access
        observe_results = [r for r in results if r[0].action_id == "observe_component"]
        assert observe_results and observe_results[-1][2].success is True

    def test_verify_mode_rejects_non_verify_repair_action(self, system):
        entries = [{"action_id": "measure_voltage", "subject": "main_bulb"}]
        with unittest.mock.patch("nl_interface.interface._parse", return_value=entries), \
             unittest.mock.patch("nl_interface.interface._verbalize", return_value="mock"):
            _, _, _, results = run("...", system, mode="verify")

        assert results[0][2].success is False
        assert "[measure_voltage] not permitted in current mode." in results[0][2].message

    def test_collect_information_mode_allows_diagnostic_action(self, system):
        entries = [{"action_id": "measure_voltage", "subject": "main_bulb"}]
        with unittest.mock.patch("nl_interface.interface._parse", return_value=entries), \
             unittest.mock.patch("nl_interface.interface._verbalize", return_value="mock"):
            _, _, _, results = run("...", system, mode="collect_information")

        assert results[0][2].success is True

    def test_invalid_mode_raises_value_error(self, system):
        with pytest.raises(ValueError):
            run("...", system, mode="diagnose")  # type: ignore[arg-type]

    def test_removed_component_returns_not_present_with_success(self):
        fresh = build_three_cubes_system(backend=PySpiceBackend(), extra_tools={"multimeter"})
        fresh.simulate()
        fresh.remove_component("psu_green_led")
        entries = [{"action_id": "observe_component", "subject": "psu_green_led"}]
        with unittest.mock.patch("nl_interface.interface._parse", return_value=entries), \
             unittest.mock.patch("nl_interface.interface._verbalize", return_value="mock"):
            _, _, _, results = run("...", fresh, mode="collect_information")

        assert results[0][2].success is True
        assert "not present" in results[0][2].message

    def test_verify_repair_on_removed_component_fails(self):
        fresh = build_three_cubes_system(backend=PySpiceBackend(), extra_tools={"multimeter"})
        fresh.simulate()
        fresh.remove_component("psu_green_led")
        entries = [{"action_id": "verify_repair", "subject": "psu_green_led"}]
        with unittest.mock.patch("nl_interface.interface._parse", return_value=entries), \
             unittest.mock.patch("nl_interface.interface._verbalize", return_value="mock"):
            _, _, _, results = run("...", fresh, mode="verify")

        assert results[0][2].success is False
        assert "not present" in results[0][2].message

    def test_unknown_action_id_returns_failure(self, system):
        entries = [{"action_id": "nonexistent_action", "subject": "main_bulb"}]
        with unittest.mock.patch("nl_interface.interface._parse", return_value=entries), \
             unittest.mock.patch("nl_interface.interface._verbalize", return_value="mock"):
            _, _, _, results = run("...", system, mode="collect_information")

        assert results[0][2].success is False
        assert "is not recognized or supported" in results[0][2].message

    def test_unknown_component_id_returns_failure(self, system):
        entries = [{"action_id": "measure_voltage", "subject": "ghost_xyz"}]
        with unittest.mock.patch("nl_interface.interface._parse", return_value=entries), \
             unittest.mock.patch("nl_interface.interface._verbalize", return_value="mock"):
            _, _, _, results = run("...", system, mode="collect_information")

        assert results[0][2].success is False
        assert "is not recognized" in results[0][2].message

    def test_empty_parse_returns_base_narrative_and_zero_cost(self, system):
        with unittest.mock.patch("nl_interface.interface._parse", return_value=[]):
            narrative, cost, entries, results = run("gibberish", system, mode="collect_information")

        assert entries == []
        assert results == []
        assert cost.time == 0
        assert "could not be mapped" in narrative


# ---------------------------------------------------------------------------
# LLM-gated tests
# ---------------------------------------------------------------------------

class TestParseLLM:
    @requires_llm
    def test_parse_two_target_action(self, system):
        result = _parse("test the continuity path from the PSU output cable to the main bulb", system)
        assert len(result) >= 1
        entry = result[0]
        assert entry["action_id"] == "test_path_continuity"
        assert "source" in entry and "sink" in entry
        assert entry["source"] in system.all_components()
        assert entry["sink"] in system.all_components()

    @requires_llm
    def test_parse_ambiguous_component_resolved(self, system):
        result = _parse("observe the red LED", system)
        assert len(result) >= 1
        assert result[0]["action_id"] == "observe_component"
        assert result[0]["subject"] in system.all_components()

    @requires_llm
    def test_parse_multi_step_with_enclosure_access(self, system):
        result = _parse("open the load module and measure the voltage at the internal bulb", system)
        assert len(result) >= 2
        action_ids = [e["action_id"] for e in result]
        assert any(aid in ("invert_enclosure", "open_peephole", "open_inspection_panel") for aid in action_ids)


class TestVerbalizeLLM:
    @requires_llm
    def test_verbalize_with_reporting_requirements(self, system):
        from diagnosable_systems_simulation.actions.diagnostic_actions import MeasureVoltage
        action = MeasureVoltage()
        result = system.apply_action(action, {"subject": system.component("main_bulb")})
        out = _verbalize([(action, result)], reporting_requirements="Return only: VOLTAGE_OK or VOLTAGE_LOW")
        assert "VOLTAGE_OK" in out or "VOLTAGE_LOW" in out

    @requires_llm
    def test_verbalize_verify_repair_outcome(self, system):
        from diagnosable_systems_simulation.actions.diagnostic_actions import VerifyRepair
        action = VerifyRepair()
        result = system.apply_action(action, {"subject": system.component("main_bulb")})
        out = _verbalize([(action, result)])
        assert isinstance(out, str) and len(out) > 0


class TestRunLLM:
    @requires_llm
    def test_run_end_to_end(self, system):
        narrative, cost, entries, results = run(
            "measure voltage at the main bulb and observe the PSU green LED",
            system,
            mode="collect_information",
        )
        assert isinstance(narrative, str) and len(narrative) > 20
        assert len(entries) >= 2
        assert cost.time > 0
        assert all(r[2].success for r in results)

    @requires_llm
    def test_run_with_reporting_requirements(self, system):
        narrative, cost, entries, results = run(
            "measure voltage at the main bulb",
            system,
            mode="collect_information",
            reporting_requirements="Return only: PASS or FAIL",
        )
        assert "PASS" in narrative or "FAIL" in narrative
