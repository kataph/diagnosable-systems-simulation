"""
Natural language interface: free-text → action list → verbalized results.

IMPORTANT: there are at least two types of action that can be executed on a system: those with the goal of collecting information useful
for diagnosis, and those with the goal of setting up the system to execute action of the first type 
(let us call the latter access actions and the former information collecting actions).
The focus of the assistant evaluation is on planning the information collection actions not the access actions. 
Therefore, the interface will carry out some minimal auto-planning-correction: 
Enclosure inversion and peephole opening will be executed automatically if needed. Their cost will then be added to the assistant suggestion cost.
Not doing this will cause assistant suggestions to randomly fail, whenever the nl interface decides not to map an implict or explicit rquired access action.

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
import logging
from typing import Literal

_logger = logging.getLogger("nl_interface")

import anthropic, openai

from diagnosable_systems_simulation.actions.base import ActionCost
from diagnosable_systems_simulation.actions.diagnostic_actions import (
    AdjustPotentiometer, ClosePeephole, InspectConnections, InvertEnclosure,
    MeasureCurrent, MeasureVoltage, ObserveComponent, OpenPeephole, ReplaceComponent,
    CloseSwitch, OpenSwitch, RestoreEnclosure, TestContinuity, TestDiode,
    TestPathContinuity, VerifyRepair,
)
from diagnosable_systems_simulation.actions.fault_actions import (
    DegradeComponent, DisconnectCable, ForceSwitch, ReconnectCable,
)
from diagnosable_systems_simulation.systems.base_system import DiagnosableSystem

MODEL = "nf-gpt-4o-2024-08-06"
# MODEL = "gpt-4.1"
Backend = Literal["openai", "anthropic"]
BACKEND: Backend = "openai"

# ---------------------------------------------------------------------------
# Action registry
# Each entry: action_id → (class, {constructor_kwarg: description})
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, tuple] = {
    "observe_component":   (ObserveComponent,   {}),
    "measure_voltage":     (MeasureVoltage,     {}),
    "measure_current":     (MeasureCurrent,     {}),
    "open_switch":         (OpenSwitch,         {}),
    "close_switch":        (CloseSwitch,        {}),
    "test_continuity":     (TestContinuity,     {}),
    "test_diode":          (TestDiode,          {}),
    "test_path_continuity": (TestPathContinuity, {
        "source_port": "str — port name on source component to probe (optional)",
        "sink_port":   "str — port name on sink component to probe (optional)",
    }),
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
        "connections": "dict[str, str] — optional: maps port_name to node_id; omit to restore original wiring",
    }),
    "degrade_component":   (DegradeComponent,   {
        "degradation": "dict — parameter overrides, e.g. {'resistance': 1e9} or {'voltage': 0.0}",
    }),
    "force_switch":        (ForceSwitch,        {
        "is_closed": "bool — true = permanently closed, false = permanently open",
    }),
    "verify_repair":       (VerifyRepair,       {}),
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


def _action_menu(registry: "dict | None" = None) -> str:
    reg = registry if registry is not None else _REGISTRY
    lines = []
    for aid, (cls, params) in reg.items():
        pstr = f"  params: {params}" if params else ""
        lines.append(f"- {aid}: {cls.description}{pstr}")
    return "\n".join(lines)


def _component_menu(system) -> str:
    from diagnosable_systems_simulation.world.knowledge_graph import EntityType
    lines = []
    for cid, c in system.kg.entities_of_type(EntityType.COMPONENT).items():
        enclosure_id = getattr(c, "enclosure_id", None)
        enclosure_note = f" [inside enclosure: {enclosure_id}]" if enclosure_id else ""
        nominal_note = getattr(c, "_nominal_observation_note", None)
        nominal_suffix = f" [NOTE: {nominal_note}]" if nominal_note else ""
        lines.append(f"- {cid}: {c.display_name}{enclosure_note}{nominal_suffix}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 1: parse free-text → list of action descriptors
# ---------------------------------------------------------------------------

_PARSE_SYSTEM = """\
You map a technician's instruction to a JSON list of actions on a physical system.
Each action must be:
  {"action_id": "<id>", "subject": "<component_id>", "params": {<constructor kwargs>}}
If an action has no listed parameters, omit "params" entirely — do NOT include it at all.
Never add keys to "params" that are not listed in the action's parameter description.
Return ONLY the JSON array — no markdown fences, no commentary.

Exception — test_path_continuity uses TWO component targets instead of "subject":
  {"action_id": "test_path_continuity", "source": "<component_id>", "sink": "<component_id>", "params": {<optional port kwargs>}}

Critical rules:
- Map ONLY the core measurement or observation actions explicitly stated.
- Do NOT infer or add context actions (e.g. switch operations, enclosure inversions, peephole openings) unless the description mentions them as context ("while the switch is closed", "with the enclosure open", etc.).
- Do NOT add observe_component unless the instruction explicitly asks to visually inspect a component.\
"""


def _parse(text: str, system, model: str = MODEL, allowed_actions: "set[str] | None" = None) -> list[dict]:
    registry = (
        {k: v for k, v in _REGISTRY.items() if k in allowed_actions}
        if allowed_actions is not None
        else _REGISTRY
    )
    prompt = (
        f"Available actions:\n{_action_menu(registry)}\n\n"
        f"System components:\n{_component_menu(system)}\n\n"
        f"Instruction: {text}"
    )
    raw = _client().create(
        model=model,
        system_prompt=_PARSE_SYSTEM,
        user_prompt=prompt,
        max_output_tokens=2048,
    )
    if not raw.rstrip().endswith("]"):
        _logger.warning(
            "nl_interface._parse: response does not end with ']' — likely truncated. "
            f"Length={len(raw)} chars. Tail: {raw[-120:]!r}"
        )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        _logger.warning(
            f"nl_interface._parse: JSON decode failed ({exc}). "
            f"Raw response was: {raw!r}. Returning empty action list."
        )
        return []



# ---------------------------------------------------------------------------
# Step 2: instantiate and execute
# ---------------------------------------------------------------------------

def _instantiate(entry: dict):
    cls, declared_params = _REGISTRY[entry["action_id"]]
    raw = entry.get("params") or {}
    # Only pass params declared in the registry; drop any extra keys the LLM hallucinated.
    params = {k: v for k, v in raw.items() if k in declared_params}
    return cls(**params) if params else cls()


def _expand_enclosure_targets(entries: list[dict], system) -> list[dict]:
    """
    If an action's subject is a PhysicalEnclosure, replace that single entry
    with one entry per component contained inside the enclosure.  This turns
    "observe the control module" into individual actions on each internal
    component, which is what a technician physically does.

    Entries with source/sink targets (test_path_continuity) are left unchanged.
    """
    from diagnosable_systems_simulation.world.components import PhysicalEnclosure

    expanded: list[dict] = []
    for entry in entries:
        subject_id = entry.get("subject")
        if subject_id and not entry.get("source"):
            comp = system.component(subject_id)
            if isinstance(comp, PhysicalEnclosure):
                sub_entries = [
                    {**entry, "subject": cid}
                    for cid, c in system.all_components().items()
                    if getattr(c, "enclosure_id", None) == subject_id
                ]
                if sub_entries:
                    expanded.extend(sub_entries)
                    continue
        expanded.append(entry)
    return expanded


def _execute(entries: list[dict], system, allowed_actions: "set[str] | None" = None) -> list[tuple]:
    entries = _expand_enclosure_targets(entries, system)
    results = []
    for entry in entries:
        action_id = entry.get("action_id", "?")
        if allowed_actions is not None and action_id not in allowed_actions:
            from diagnosable_systems_simulation.actions.base import ActionResult
            result = ActionResult(
                success=False,
                message=f"[{action_id}] not permitted in current mode.",
            )
            action = type("_stub", (), {"action_id": action_id, "cost": ActionCost()})()
            results.append((action, result))
            continue
        try:
            action = _instantiate(entry)
            subject_id = entry.get("subject")
            source_id  = entry.get("source")
            sink_id    = entry.get("sink")
            if source_id and sink_id:
                targets = {
                    "source": system.component(source_id),
                    "sink":   system.component(sink_id),
                }
            else:
                targets = {"subject": system.component(subject_id)} if subject_id else {}
            result = system.apply_action(action, targets)
            # If the action failed due to a missing affordance, auto-satisfy the
            # physical prerequisite and retry once.
            if not result.success and subject_id:
                comp = targets.get("subject")
                enclosure_id = getattr(comp, "enclosure_id", None)
                if enclosure_id and "REACHABLE" in result.message:
                    enclosure = system.component(enclosure_id)
                    if enclosure is not None and not getattr(enclosure, "is_inverted", False):
                        inv_action = InvertEnclosure()
                        inv_result = system.apply_action(inv_action, {"subject": enclosure})
                        _logger.info(f"auto-inverted {enclosure_id!r} to satisfy REACHABLE for {subject_id!r}")
                        results.append((inv_action, inv_result))
                        result = system.apply_action(action, targets)
                elif enclosure_id and "OBSERVABLE" in result.message:
                    from diagnosable_systems_simulation.world.components import Peephole
                    peephole = next(
                        (c for c in system.all_components().values()
                         if isinstance(c, Peephole) and getattr(c, "enclosure_id", None) == enclosure_id
                         and not c.is_open),
                        None,
                    )
                    if peephole is not None:
                        ph_action = OpenPeephole()
                        ph_result = system.apply_action(ph_action, {"subject": peephole})
                        _logger.info(f"auto-opened peephole {peephole.component_id!r} to satisfy OBSERVABLE for {subject_id!r}")
                        results.append((ph_action, ph_result))
                        result = system.apply_action(action, targets)
                    else:
                        # No peephole: invert the enclosure (REACHABLE implies OBSERVABLE).
                        enclosure = system.component(enclosure_id)
                        if enclosure is not None and not getattr(enclosure, "is_inverted", False):
                            inv_action = InvertEnclosure()
                            inv_result = system.apply_action(inv_action, {"subject": enclosure})
                            _logger.info(f"auto-inverted {enclosure_id!r} to satisfy OBSERVABLE for {subject_id!r}")
                            results.append((inv_action, inv_result))
                            result = system.apply_action(action, targets)
        except Exception as exc:
            from diagnosable_systems_simulation.actions.base import ActionResult
            result = ActionResult(success=False, message=f"[{action_id}] error: {exc}")
            action = type("_stub", (), {"action_id": action_id, "cost": ActionCost()})()
        results.append((action, result))
    return results


# ---------------------------------------------------------------------------
# Step 3: verbalize
# ---------------------------------------------------------------------------

def _verbalize(results: list[tuple], original_text: str, model: str = MODEL) -> str:
    lines = []
    for action, result in results:
        lines.append(f"action_id: {action.action_id}, success: {result.success}, message: {result.message}")
        if result.observation:
            for p in result.observation.properties:
                unit = f" {p.unit}" if p.unit else ""
                lines.append(f"  {p.name}: {p.value}{unit}")
    raw = "\n".join(lines)

    return _client().create(
        model=model,
        system_prompt="""\
You are a summarization robot. Summarize diagnostic results in 1–3 plain sentences for a technician.
Limit yourself strictly to the information in the user prompt — do not add opinions, causes, or extra remarks.

Critical rule — polarity inversions:
If a (+)-labeled cable or port is connected to a (−)-labeled cable or port (or vice versa), you MUST
explicitly state this as a POLARITY INVERSION and name the affected cables. Do not describe such a
connection as "correct" or "nominal". Example: "POLARITY INVERSION DETECTED: PSU Output Cable (+)
is connected to Control Input Cable (−), and PSU Output Cable (−) is connected to Control Input Cable (+)."
""",
        user_prompt=raw,
        max_output_tokens=256,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(text: str, system: DiagnosableSystem, model: str = MODEL, allowed_actions: "set[str] | None" = None) -> tuple[str, ActionCost, list[dict], list[tuple]]:
    """
    Parse *text*, execute the implied actions on *system*, and return a
    plain-language summary together with the total action cost.
    Return textual outcome, cost, parsed actions, action results

    allowed_actions: if provided, restricts the action registry to this set of
        action IDs.  Use this to prevent the NL agent from attempting repair or
        fault-injection actions during ordinary diagnosis.
    """
    entries = _parse(text, system, model, allowed_actions)
    results = _execute(entries, system, allowed_actions)

    resources: dict[str, float] = {}
    for a, _ in results:
        for k, v in a.cost.resources_consumed.items():
            resources[k] = resources.get(k, 0) + v
    total = ActionCost(
        time=sum(a.cost.time for a, _ in results),
        equipment=list({e for a, _ in results for e in a.cost.equipment}),
        resources_consumed=resources,
    )
        
    return _verbalize(results=results, original_text=text, model=model), total, entries, results
