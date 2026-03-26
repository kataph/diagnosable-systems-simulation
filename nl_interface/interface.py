"""
Natural language interface: free-text → action list → verbalized results.

Usage::

    from nl_interface.interface import run
    from diagnosable_systems_simulation.systems.three_cubes.factory import build_three_cubes_system
    from diagnosable_systems_simulation.electrical_simulation.backend.stub import StubBackend

    system = build_three_cubes_system(backend=StubBackend(), extra_tools={"multimeter"})
    narrative, cost = run("measure voltage at the main bulb and the control output cables", system)
    print(narrative)
    print(f"cost: {cost.time}s")
"""
from __future__ import annotations

import json
from typing import Literal

import anthropic, openai

from diagnosable_systems_simulation.actions.base import ActionCost
from diagnosable_systems_simulation.actions.diagnostic_actions import (
    AdjustPotentiometer, ClosePeephole, InspectConnections, InvertEnclosure,
    MeasureVoltage, ObserveComponent, OpenPeephole, ReplaceComponent,
    RestoreEnclosure, TestContinuity, TestDiode, ToggleSwitch,
)
from diagnosable_systems_simulation.actions.fault_actions import (
    DegradeComponent, DisconnectCable, ForceSwitch, ReconnectCable,
)
from diagnosable_systems_simulation.systems.base_system import DiagnosableSystem

MODEL = "nf-gpt-4o-2024-08-06"
Backend = Literal["openai", "anthropic"]
BACKEND: Backend = "openai"

# ---------------------------------------------------------------------------
# Action registry
# Each entry: action_id → (class, {constructor_kwarg: description})
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, tuple] = {
    "observe_component":   (ObserveComponent,   {}),
    "measure_voltage":     (MeasureVoltage,     {}),
    "toggle_switch":       (ToggleSwitch,       {}),
    "test_continuity":     (TestContinuity,     {}),
    "test_diode":          (TestDiode,          {}),
    "inspect_connections": (InspectConnections, {}),
    "invert_enclosure":    (InvertEnclosure,    {}),
    "restore_enclosure":   (RestoreEnclosure,   {}),
    "open_peephole":       (OpenPeephole,       {}),
    "close_peephole":      (ClosePeephole,      {}),
    "replace_component":   (ReplaceComponent,   {
        "replacement_part_id": "str — identifier of the replacement part",
        "replacement_cost":    "float (default 1.0)",
    }),
    "adjust_potentiometer": (AdjustPotentiometer, {
        "new_position": "float in [0.0, 1.0]",
    }),
    "disconnect_cable":    (DisconnectCable,    {
        "port_names": "list[str] — port names to disconnect, e.g. ['n']; null = all ports",
    }),
    "reconnect_cable":     (ReconnectCable,     {
        "connections": "dict[str, str] — maps port_name to node_id (e.g. {'n': 'net_4'})",
    }),
    "degrade_component":   (DegradeComponent,   {
        "overrides": "dict — parameter overrides, e.g. {'resistance': 1e9} or {'voltage': 0.0}",
    }),
    "force_switch":        (ForceSwitch,        {
        "is_closed": "bool — true = permanently closed, false = permanently open",
    }),
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TextClient:
    def __init__(self, backend: Backend):
        self.backend = backend
        match backend:
            case "openai":
                self._client = openai.OpenAI()
            case "anthropic":
                self._client = anthropic.Anthropic()
    
    def create(self, model, system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
        match self.backend:
            case "openai":
                return self._client.responses.create(
                    model=model, 
                    max_output_tokens=max_output_tokens,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},]
                    ).output_text.strip()
            case "anthropic":
                return self._client.messages.create(
                    model=model,
                    max_tokens=max_output_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    ).content[0].text.strip()


_CLIENT: "TextClient | None" = None

def _client(backend: Backend = BACKEND) -> "TextClient":
    global _CLIENT
    if _CLIENT is None or _CLIENT.backend != backend:
        _CLIENT = TextClient(backend)
    return _CLIENT


def _action_menu() -> str:
    lines = []
    for aid, (cls, params) in _REGISTRY.items():
        pstr = f"  params: {params}" if params else ""
        lines.append(f"- {aid}: {cls.description}{pstr}")
    return "\n".join(lines)


def _component_menu(system) -> str:
    from diagnosable_systems_simulation.world.knowledge_graph import EntityType
    return "\n".join(
        f"- {cid}: {c.display_name}"
        for cid, c in system.kg.entities_of_type(EntityType.COMPONENT).items()
    )


# ---------------------------------------------------------------------------
# Step 1: parse free-text → list of action descriptors
# ---------------------------------------------------------------------------

_PARSE_SYSTEM = """\
You map a technician's instruction to a JSON list of actions on a physical system.
Each action must be:
  {"action_id": "<id>", "subject": "<component_id>", "params": {<constructor kwargs>}}
"params" may be omitted if the action has no constructor parameters.
Return ONLY the JSON array — no markdown fences, no commentary.\
"""


def _parse(text: str, system) -> list[dict]:
    prompt = (
        f"Available actions:\n{_action_menu()}\n\n"
        f"System components:\n{_component_menu(system)}\n\n"
        f"Instruction: {text}"
    )
    raw = _client().create(
        model=MODEL,
        system_prompt=_PARSE_SYSTEM,
        user_prompt=prompt,
        max_output_tokens=512,
    )
    return json.loads(raw)



# ---------------------------------------------------------------------------
# Step 2: instantiate and execute
# ---------------------------------------------------------------------------

def _instantiate(entry: dict):
    cls, _ = _REGISTRY[entry["action_id"]]
    params = entry.get("params") or {}
    return cls(**params) if params else cls()


def _execute(entries: list[dict], system) -> list[tuple]:
    results = []
    for entry in entries:
        try:
            action = _instantiate(entry)
            subject_id = entry.get("subject")
            targets = {"subject": system.component(subject_id)} if subject_id else {}
            result = system.apply_action(action, targets)
        except Exception as exc:
            from diagnosable_systems_simulation.actions.base import ActionResult
            action_id = entry.get("action_id", "?")
            result = ActionResult(success=False, message=f"[{action_id}] error: {exc}")
            action = type("_stub", (), {"action_id": action_id, "cost": ActionCost()})()
        results.append((action, result))
    return results


# ---------------------------------------------------------------------------
# Step 3: verbalize
# ---------------------------------------------------------------------------

def _verbalize(results: list[tuple], original_text: str) -> str:
    lines = []
    for action, result in results:
        lines.append(f"action_id: {action.action_id}, action_description: {original_text}, success: {result.success}, message: {result.message}")
        if result.observation:
            for p in result.observation.properties:
                unit = f" {p.unit}" if p.unit else ""
                lines.append(f"  {p.name}: {p.value}{unit}")
    raw = "\n".join(lines)

    return _client().create(
        model=MODEL,
        system_prompt="You are a concise technical assistant. Summarize diagnostic results in 1–3 plain sentences for a technician.",
        user_prompt=raw,
        max_output_tokens=256,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(text: str, system: DiagnosableSystem) -> tuple[str, ActionCost]:
    """
    Parse *text*, execute the implied actions on *system*, and return a
    plain-language summary together with the total action cost.
    """
    entries = _parse(text, system)
    results = _execute(entries, system)

    total = ActionCost(
        time=sum(a.cost.time for a, _ in results),
        equipment=list({e for a, _ in results for e in a.cost.equipment}),
        resources_consumed={
            k: v
            for a, _ in results
            for k, v in a.cost.resources_consumed.items()
        },
    )

    return _verbalize(rsults=results, original_text=text), total
