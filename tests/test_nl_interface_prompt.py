"""
Tests for the NL interface prompt construction.

Verifies what is actually exposed to the NL agent:
  1. System prompt  — _PARSE_SYSTEM contains the JSON format rules and
                      critical parsing constraints.
  2. Action menu    — _action_menu() lists action IDs, descriptions, and
                      param specs; respects the allowed_actions filter.
  3. Component menu — _component_menu() lists component IDs, display names,
                      and [inside enclosure: ...] annotations for enclosed
                      components.
  4. Instruction    — the raw NL text is appended verbatim.

Run:
    python -m pytest tests/test_nl_interface_prompt.py -v -s
    (use -s to see the printed prompt sections)
"""
import pytest

from diagnosable_systems_simulation.electrical_simulation.backend.stub import StubBackend
from diagnosable_systems_simulation.systems.three_cubes.factory import build_three_cubes_system
from nl_interface.interface import (
    _PARSE_SYSTEM,
    _REGISTRY,
    _action_menu,
    _component_menu,
    _expand_enclosure_targets,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def system():
    s = build_three_cubes_system(backend=StubBackend(), extra_tools={"multimeter"})
    s.simulate()
    return s


# ---------------------------------------------------------------------------
# Prompt printer — runs first so the output appears before the test results
# ---------------------------------------------------------------------------

def test_print_full_prompt(system):
    """Print the complete prompt that would be sent to the NL agent."""
    instruction = "measure voltage at the main bulb and inspect the control module"
    user_prompt = (
        f"Available actions:\n{_action_menu()}\n\n"
        f"System components:\n{_component_menu(system)}\n\n"
        f"Instruction: {instruction}"
    )
    sep = "─" * 72
    print(f"\n{sep}")
    print("SYSTEM PROMPT")
    print(sep)
    print(_PARSE_SYSTEM)
    print(f"\n{sep}")
    print("USER PROMPT")
    print(sep)
    print(user_prompt)
    print(sep)


# ---------------------------------------------------------------------------
# 1. System prompt
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_json_array_format_described(self):
        assert '"action_id"' in _PARSE_SYSTEM
        assert '"subject"' in _PARSE_SYSTEM

    def test_two_target_exception_documented(self):
        assert "test_path_continuity" in _PARSE_SYSTEM
        assert '"source"' in _PARSE_SYSTEM
        assert '"sink"' in _PARSE_SYSTEM

    def test_no_infer_context_actions_rule(self):
        assert "Do NOT infer" in _PARSE_SYSTEM

    def test_no_observe_unless_explicit_rule(self):
        assert "observe_component" in _PARSE_SYSTEM


# ---------------------------------------------------------------------------
# 2. Action menu
# ---------------------------------------------------------------------------

class TestActionMenu:
    def test_all_registry_actions_appear_unfiltered(self):
        menu = _action_menu()
        for action_id in _REGISTRY:
            assert action_id in menu, f"{action_id!r} missing from unfiltered action menu"

    def test_allowed_filter_restricts_menu(self):
        allowed = {"measure_voltage", "observe_component"}
        menu = _action_menu({k: v for k, v in _REGISTRY.items() if k in allowed})
        assert "measure_voltage" in menu
        assert "observe_component" in menu
        assert "replace_component" not in menu
        assert "reconnect_cable" not in menu

    def test_params_appear_for_actions_that_have_them(self):
        menu = _action_menu()
        assert "replacement_part_id" in menu   # ReplaceComponent
        assert "new_position" in menu           # AdjustPotentiometer
        assert "source_port" in menu            # TestPathContinuity

    def test_no_params_line_for_parameterless_actions(self):
        menu = _action_menu({"measure_voltage": _REGISTRY["measure_voltage"]})
        assert "params:" not in menu


# ---------------------------------------------------------------------------
# 3. Component menu
# ---------------------------------------------------------------------------

class TestComponentMenu:
    def test_all_components_listed(self, system):
        menu = _component_menu(system)
        for cid in system.all_components():
            assert cid in menu, f"Component {cid!r} missing from component menu"

    def test_display_names_present(self, system):
        menu = _component_menu(system)
        assert "Main Lightbulb" in menu
        assert "Control Switch" in menu

    def test_enclosed_components_annotated(self, system):
        menu = _component_menu(system)
        assert "[inside enclosure: cube_ctrl]" in menu

    def test_enclosures_themselves_not_annotated(self, system):
        menu = _component_menu(system)
        cube_line = next(l for l in menu.splitlines() if l.startswith("- cube_ctrl:"))
        assert "[inside enclosure:" not in cube_line


# ---------------------------------------------------------------------------
# 4. Instruction passthrough
# ---------------------------------------------------------------------------

class TestPromptAssembly:
    def test_instruction_appears_verbatim(self, system):
        instruction = "measure voltage at the main bulb"
        prompt = (
            f"Available actions:\n{_action_menu()}\n\n"
            f"System components:\n{_component_menu(system)}\n\n"
            f"Instruction: {instruction}"
        )
        assert prompt.endswith(instruction)

    def test_prompt_sections_in_order(self, system):
        prompt = (
            f"Available actions:\n{_action_menu()}\n\n"
            f"System components:\n{_component_menu(system)}\n\n"
            f"Instruction: test"
        )
        assert prompt.index("Available actions:") < prompt.index("System components:") < prompt.index("Instruction:")


# ---------------------------------------------------------------------------
# 5. Enclosure expansion
# ---------------------------------------------------------------------------

class TestExpandEnclosureTargets:
    def test_enclosure_subject_expanded_to_internal_components(self, system):
        """An action on cube_ctrl expands to one entry per enclosed component."""
        entries = [{"action_id": "inspect_connections", "subject": "cube_ctrl"}]
        expanded = _expand_enclosure_targets(entries, system)
        subjects = [e["subject"] for e in expanded]
        # cube_ctrl contains ctrl_switch, ctrl_red_led, ctrl_red_resistor
        assert "cube_ctrl" not in subjects, "Enclosure itself should not remain as a target"
        assert len(subjects) > 1, "Should expand to multiple sub-components"
        # All expanded subjects must belong to cube_ctrl
        for cid in subjects:
            comp = system.component(cid)
            assert getattr(comp, "enclosure_id", None) == "cube_ctrl"

    def test_action_id_preserved_across_expansion(self, system):
        """All expanded entries keep the original action_id."""
        entries = [{"action_id": "observe_component", "subject": "cube_ctrl"}]
        expanded = _expand_enclosure_targets(entries, system)
        assert all(e["action_id"] == "observe_component" for e in expanded)

    def test_non_enclosure_subject_unchanged(self, system):
        """Actions on regular components are not expanded."""
        entries = [{"action_id": "measure_voltage", "subject": "main_bulb"}]
        expanded = _expand_enclosure_targets(entries, system)
        assert expanded == entries

    def test_source_sink_entries_unchanged(self, system):
        """test_path_continuity entries (source/sink) are never expanded."""
        entries = [{
            "action_id": "test_path_continuity",
            "source": "cube_ctrl",   # even if source is an enclosure
            "sink": "main_bulb",
        }]
        expanded = _expand_enclosure_targets(entries, system)
        assert expanded == entries

    def test_mixed_entries_only_enclosure_expanded(self, system):
        """In a mixed list, only the enclosure entry is expanded."""
        entries = [
            {"action_id": "measure_voltage",    "subject": "main_bulb"},
            {"action_id": "inspect_connections", "subject": "cube_ctrl"},
            {"action_id": "observe_component",  "subject": "psu_cable_pos"},
        ]
        expanded = _expand_enclosure_targets(entries, system)
        subjects = [e["subject"] for e in expanded]
        assert "main_bulb" in subjects
        assert "psu_cable_pos" in subjects
        assert "cube_ctrl" not in subjects
        assert len(expanded) > 3  # cube_ctrl replaced by multiple entries

    def test_cables_not_included_in_expansion(self, system):
        """Cables have no enclosure_id and must not appear in the expansion."""
        from diagnosable_systems_simulation.world.components import Cable
        entries = [{"action_id": "inspect_connections", "subject": "cube_ctrl"}]
        expanded = _expand_enclosure_targets(entries, system)
        for e in expanded:
            comp = system.component(e["subject"])
            assert not isinstance(comp, Cable), (
                f"Cable {e['subject']!r} should not appear in enclosure expansion"
            )