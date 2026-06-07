"""
Mock natural-language interface — zero LLM calls.

Designed to work reliably with DiagnosticAction descriptions of the form
produced by DiagnosticAction.get_name():  "<Type> -> <target_component_id>"

This format is what DiagnosticAssistantRandomTrajectory always emits (its
actions have no free-text description, so execute_action() calls
``action.description or action.get_name()``).

Parsing strategy
----------------
1. Try regex ``r'(\w+)\s*->\s*(\S+)'`` on the text.
2. Map the captured type keyword to an action_id.
3. Use the captured target as the subject component ID.
4. Fall back to keyword scanning across component IDs if step 1 fails.
5. In 'verify' mode always use ``verify_repair``.
"""
from __future__ import annotations

import re
from logging import Logger
from typing import Literal, Optional

from diagnosable_systems_simulation.actions.base import ActionCost
from diagnosable_systems_simulation.actions.diagnostic_actions import (
    ObserveComponent, TestContinuity, ReplaceComponent, AdjustPotentiometer,
    MeasureVoltage, InspectConnections, VerifyRepair,
)
from diagnosable_systems_simulation.systems.base_system import DiagnosableSystem


_TYPE_TO_ACTION_ID: dict[str, str] = {
    "observe":   "observe_component",
    "test":      "test_continuity",
    "replace":   "replace_component",
    "adjust":    "adjust_potentiometer",
    "measure":   "measure_voltage",
    "inspect":   "inspect_connections",
}

_ACTION_ID_TO_CLS = {
    "observe_component":   ObserveComponent,
    "test_continuity":     TestContinuity,
    "replace_component":   ReplaceComponent,
    "adjust_potentiometer": AdjustPotentiometer,
    "measure_voltage":     MeasureVoltage,
    "inspect_connections": InspectConnections,
    "verify_repair":       VerifyRepair,
}

_ARROW_PATTERN = re.compile(r'(\w+)\s*->\s*(\S+)', re.IGNORECASE)


def _resolve_subject(target: str, system: DiagnosableSystem) -> str:
    """Return a valid component ID matching ``target``, or the first component."""
    components = system.all_components()
    if target in components:
        return target
    # Substring match
    target_lower = target.lower()
    for cid in components:
        if target_lower in cid.lower():
            return cid
    return next(iter(components))


def _parse_text(text: str, system: DiagnosableSystem) -> tuple[str, str]:
    """Return (action_id, subject_component_id)."""
    m = _ARROW_PATTERN.search(text)
    if m:
        type_kw = m.group(1).lower()
        target = m.group(2).rstrip('.,;')
        action_id = _TYPE_TO_ACTION_ID.get(type_kw, "observe_component")
        subject = _resolve_subject(target, system)
        return action_id, subject

    # Fallback: keyword scan
    text_lower = text.lower()
    action_id = "observe_component"
    for kw, aid in _TYPE_TO_ACTION_ID.items():
        if kw in text_lower:
            action_id = aid
            break

    subject = None
    for cid in system.all_components():
        if cid.lower() in text_lower:
            subject = cid
            break
    if subject is None:
        subject = next(iter(system.all_components()))

    return action_id, subject


def mock_run(
    text: str,
    system: DiagnosableSystem,
    model: str = "mock",
    mode: Literal['verify', 'collect_information'] = 'collect_information',
    _logger: Optional[Logger] = None,
    reporting_requirements: Optional[str] = None,
) -> tuple[str, ActionCost, list[dict], list[tuple]]:
    """
    Mock replacement for nl_interface.interface.run().

    Returns the same 4-tuple: (narrative, cost, entries, results).
    Zero LLM calls.
    """
    from diagnosable_systems_simulation.actions.base import ActionResult

    if mode == 'verify':
        action_id = "verify_repair"
        # Pick subject from text or first component
        _, subject = _parse_text(text, system)
    else:
        action_id, subject = _parse_text(text, system)

    cls = _ACTION_ID_TO_CLS.get(action_id, ObserveComponent)
    try:
        action_obj = cls()
    except Exception:
        action_obj = ObserveComponent()

    entry = {"action_id": action_id, "subject": subject}

    try:
        comp = system.component(subject)
        targets = {"subject": comp}
        result = system.apply_action(action_obj, targets)
    except Exception as exc:
        result = ActionResult(success=False, message=str(exc))

    narrative = f"Mock outcome: {action_id} on {subject}."
    cost = ActionCost(time=float(action_obj.cost.time))

    return narrative, cost, [entry], [(action_obj, {"subject": subject}, result)]
