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
    from diagnosable_systems_simulation.electrical_simulation.backend.spice import PySpiceBackend

    system = build_three_cubes_system(backend=PySpiceBackend(), extra_tools={"multimeter"})
    narrative, cost = run("measure voltage at the main bulb and the control output cables", system)
    print(narrative)
    print(f"cost: {cost.time}s")
"""
from __future__ import annotations

import json
import inspect
from logging import Logger
from typing import Literal


import anthropic, openai

from diagnosable_systems_simulation.actions.base import ActionCost
from diagnosable_systems_simulation.actions.diagnostic_actions import (
    AdjustPotentiometer, CloseInspectionPanel, ClosePeephole, InspectConnections,
    InvertEnclosure, MeasureCurrent, MeasureVoltage, MoveLED, ObserveComponent,
    OpenInspectionPanel, OpenPeephole, ReplaceComponent, CloseSwitch, OpenSwitch,
    RestoreEnclosure, RotateEnclosure, ShortPorts, TestContinuity,
    DetachSequenceOfControlModulesAndAttachItToPowerAndLoad,
    TestDiode, TestPathContinuity, VerifyRepair,
)
from diagnosable_systems_simulation.actions.fault_actions import (
    DegradeComponent, DisconnectCable, ForceSwitch, ReconnectCable,
)
from diagnosable_systems_simulation.systems.base_system import DiagnosableSystem
from diagnosable_systems_simulation.world.components import Module


import diagnosable_systems_simulation.actions.diagnostic_actions
_DIAGNOSTIC_ALLOWED_ACTIONS: set[str] = {
    cls.action_id
    for name, cls in inspect.getmembers(diagnosable_systems_simulation.actions.diagnostic_actions, inspect.isclass)
    if hasattr(cls, "action_id") and cls.__module__ == diagnosable_systems_simulation.actions.diagnostic_actions.__name__
}



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
    "rotate_enclosure":    (RotateEnclosure,    {}),
    "open_peephole":           (OpenPeephole,           {}),
    "close_peephole":          (ClosePeephole,          {}),
    "open_inspection_panel":   (OpenInspectionPanel,   {}),
    "close_inspection_panel":  (CloseInspectionPanel,  {}),
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
    "move_led":            (MoveLED,            {
        "target_module_id": "str — component_id of the target PhysicalEnclosure (e.g. 'cube_psu' or 'cube_ctrl3')",
    }),
    "short_ports":         (ShortPorts,         {
        "source_port": "str — port name on source component to bridge (optional)",
        "target_port": "str — port name on sink component to bridge (optional)",
    }),
    "detach_sequence_of_control_modules_and_attach_it_to_power_and_load": (
        DetachSequenceOfControlModulesAndAttachItToPowerAndLoad, {}
    ),
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


def _component_menu(system: DiagnosableSystem) -> str:
    from diagnosable_systems_simulation.world.knowledge_graph import EntityType
    lines = []
    
    # NEW: List Aggregates/Modules first so the LLM understands the hierarchy
    lines.append("Aggregates (Modules):")
    for cid, c in system.all_components().items():
        if isinstance(c, Module):
            # Fetch children to show the LLM what is inside
            parts = [p.component_id for p in system.parts_of_module(cid)]
            lines.append(f"- {cid}: {c.display_name} (Components that are part of the module: {', '.join(parts)})")
    
    
    lines.append("\nIndividual Components:")
    for cid, c in system.kg.entities_of_type(EntityType.COMPONENT).items():
        enclosure_id = getattr(c, "enclosure_id", None)
        enclosure_note = f" [inside enclosure: {enclosure_id}]" if enclosure_id else ""
        nominal_note = getattr(c, "_nominal_observation_note", None)
        nominal_suffix = f" [NOTE: {nominal_note}]" if nominal_note else ""
        lines.append(f"- {cid}: {c.display_name}{enclosure_note}{nominal_suffix}")
    # Physically removed components are still listed without any special label so
    # the parser LLM maps to them normally. The execution layer intercepts the ID
    # and returns a "not present" result, preventing fallback to wrong components.
    for cid, display_name in getattr(system, "_removed_components", {}).items():
        lines.append(f"- {cid}: {display_name}")
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

Exception — the following actions use TWO component targets ("source" and "sink") instead of "subject":
  {"action_id": "test_path_continuity", "source": "<component_id>", "sink": "<component_id>", "params": {<optional port kwargs>}}
  {"action_id": "short_ports", "source": "<component_id>", "sink": "<component_id>", "params": {<optional port kwargs>}}
  {"action_id": "detach_sequence_of_control_modules_and_attach_it_to_power_and_load", "source": "<first_ctrl_module_id>", "sink": "<last_ctrl_module_id>"}
    — source is the first control module, sink is the last (e.g. "modules 1 to 4" → source="cube_ctrl1", sink="cube_ctrl4").
    — Use this action whenever the instruction mentions testing, detaching, or bypassing a contiguous range of control modules.
    — NEVER decompose this into individual disconnect_cable / reconnect_cable calls.

Critical rules:
- Map ONLY the core measurement or observation actions explicitly stated.
- Do NOT infer or add context actions (e.g. switch operations, enclosure inversions, peephole openings) unless the description mentions them as context ("while the switch is closed", "with the enclosure open", etc.).
- Do NOT add observe_component unless the instruction explicitly asks to visually inspect a component.
- Every action except the three source/sink exceptions above MUST include exactly one "subject".
- Under no circumstances may an action be emitted without a "subject" (except the three source/sink exceptions above).
- Distinguish between enclosures (e.g. a cube that contains some components) and aggragates/modules: the former is a thing that encloses some other components by virtue of its shape, the latter are things that are composed by multiple (sub-)components

Complex/Composite actions:
- some actions will be made from multiple sub-actions. For instace, swapping two cables will require to disconnect and then reconnect the target cables exchanging their locations. If you meet a complex action, try to map it to a corresponding sequence of valid sub-actions. 

Plural / multi-component instructions:
- If the instruction refers to multiple components (e.g., "all cables", "every wire", "all LEDs", "look at all the cables", "measure voltages everywhere"), you must output one action per component matching that noun category.
- Each of those actions must include its own "subject" (or "source"/"sink" for continuity actions).
- NEW: If an instruction targets an AGGREGATE or MODULE (e.g., "test the psu module", "measure voltages in ctrl_cube1"), you MUST emit one individual action for EVERY component listed as being aggregated by that module in the system menu.
- Never emit a “global” action that lacks the required component fields.

Component-matching for plural phrases:
- When a noun class is used ("cables", "LEDs", "switches"), map it to all system components whose identifiers contain that class name (case-insensitive substring match).
- If an instruction is ambiguous but clearly refers to a category (e.g., “check all connections”), infer the category and list all components of that type.
- If no component matches the noun, return an empty JSON list.

Return format:
- Always return a JSON array.
- No surrounding text, no explanation, no markdown code fences.
\
"""


def _parse(text: str, system: DiagnosableSystem, model: str = MODEL, allowed_actions: "set[str] | None" = None, _logger: Logger | None = None) -> list[dict]:
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
    _logger.debug(f"_parse prompt:\n{prompt}")
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
    with one entry per relevant sub-component.

    - For ``verify_repair``: expands to all PART_OF children of the module
      (switch, cables, …).  This models "replace the whole module", which
      physically restores every part inside — including cables that are
      not physically enclosed but belong to the module.
      Removed components are excluded from the expansion (they have no fault
      to repair).

    - For all other actions: expands to CONTAINED_IN components only (the
      components physically inside the enclosure box), which is what a
      technician accesses when they open the enclosure.

    Entries with source/sink targets (test_path_continuity) are left unchanged.
    """
    from diagnosable_systems_simulation.world.components import PhysicalEnclosure

    expanded: list[dict] = []
    for entry in entries:
        subject_id = entry.get("subject")
        if subject_id and not entry.get("source"):
            # Removed components are no longer in the KG; skip expansion and
            # let _execute handle them via the _removed_components dict.
            removed = getattr(system, "_removed_components", {})
            if subject_id in removed:
                expanded.append(entry)
                continue
            try:
                comp = system.component(subject_id)
            except KeyError:
                expanded.append(entry)
                continue
            if isinstance(comp, PhysicalEnclosure):
                if entry.get("action_id") == "verify_repair":
                    # Expand to all PART_OF members, skipping removed ones.
                    sub_entries = [
                        {**entry, "subject": part.component_id}
                        for part in system.parts_of_module(subject_id)
                        if part.component_id not in removed
                    ]
                elif entry.get("action_id") in ["observe_component", "measure_voltage", "measure_current", "replace_component"]:
                    # Expand to components physically inside the enclosure only for certain action types.
                    sub_entries = [
                        {**entry, "subject": cid}
                        for cid, c in system.all_components().items()
                        if getattr(c, "enclosure_id", None) == subject_id
                    ]
                else:
                    sub_entries = []
                if sub_entries:
                    expanded.extend(sub_entries)
                    continue
        expanded.append(entry)
    return expanded


def _execute(entries: list[dict], system, allowed_actions: "set[str] | None" = None, _logger: Logger | None = None) -> list[tuple]:
    from diagnosable_systems_simulation.actions.base import ActionResult

    entries = _expand_enclosure_targets(entries, system)
    results = []
    for entry in entries:
        action_id = entry.get("action_id", "?")

        # Unknown action ID (not in registry and not caught by allowed_actions check).
        if action_id not in _REGISTRY:
            result = ActionResult(
                success=False,
                message=f"Action '{action_id}' is not recognized or supported.",
            )
            action = type("_stub", (), {"action_id": action_id, "cost": ActionCost()})()
            results.append((action, dict(), result))
            continue

        if allowed_actions is not None and action_id not in allowed_actions:
            result = ActionResult(
                success=False,
                message=f"[{action_id}] not permitted in current mode.",
            )
            action = type("_stub", (), {"action_id": action_id, "cost": ActionCost()})()
            results.append((action, dict(), result))
            continue
        try:
            action = _instantiate(entry)
            subject_id = entry.get("subject")
            # Short-circuit for physically removed components: return a clean
            # "not present" result rather than letting the lookup raise KeyError
            # and falling through to a neighbouring component.
            removed = getattr(system, "_removed_components", {})
            if subject_id and subject_id in removed:
                display_name = removed[subject_id]
                # verify_repair on a removed component is a failed attempt:
                # you cannot repair what has been physically removed.
                # All other actions (observe, inspect…) succeed vacuously.
                success = action_id != "verify_repair"
                result = ActionResult(
                    success=success,
                    message=(
                        f"'{display_name}' is not present — the component has "
                        f"been physically removed from the system, and it is not "
                        f"possible to replace it."
                    ),
                )
                # Keep the instantiated action so its real cost is recorded.
                results.append((action, {"subject": subject_id}, result))
                continue
            source_id  = entry.get("source")
            sink_id    = entry.get("sink")
            # Unknown component IDs: return a clear "not recognized" result.
            for cid_key, cid_val in (("subject", subject_id), ("source", source_id), ("sink", sink_id)):
                if cid_val and cid_val not in removed:
                    try:
                        system.component(cid_val)
                    except KeyError:
                        result = ActionResult(
                            success=False,
                            message=f"Component '{cid_val}' is not recognized — it is not present in this system.",
                        )
                        action = type("_stub", (), {"action_id": action_id, "cost": ActionCost()})()
                        results.append((action, {"subject": subject_id, "source": source_id, "sink": sink_id}, result))
                        break
            else:
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
                # For source/sink actions (test_path_continuity, etc.) auto-invert any
                # enclosed component before retrying.
                if not result.success and not result.success and source_id and sink_id and "REACHABLE" in result.message:
                    inverted_any = False
                    for role_id in (source_id, sink_id):
                        comp_r = system.component(role_id)
                        enc_id_r = getattr(comp_r, "enclosure_id", None)
                        if enc_id_r:
                            enc_r = system.component(enc_id_r)
                            if enc_r is not None and not getattr(enc_r, "is_inverted", False):
                                inv_action = InvertEnclosure()
                                inv_result = system.apply_action(inv_action, targ:={"subject": enc_r})
                                _logger.info(f"auto-inverted {enc_id_r!r} to satisfy REACHABLE for {role_id!r}")
                                results.append((inv_action, targ, inv_result))
                                inverted_any = True
                    if inverted_any:
                        result = system.apply_action(action, targets)
                if not result.success and subject_id:
                    comp = targets.get("subject")
                    enclosure_id = getattr(comp, "enclosure_id", None)
                    if enclosure_id and "REACHABLE" in result.message:
                        from diagnosable_systems_simulation.world.components import InspectionPanel as _InspectionPanel
                        inspection_panel = next(
                            (c for c in system.all_components().values()
                             if isinstance(c, _InspectionPanel)
                             and getattr(c, "enclosure_id", None) == enclosure_id
                             and not c.is_open),
                            None,
                        )
                        if inspection_panel is not None:
                            ip_action = OpenInspectionPanel()
                            ip_result = system.apply_action(ip_action, targ:={"subject": inspection_panel})
                            _logger.info(f"auto-opened inspection panel {inspection_panel.component_id!r} to satisfy REACHABLE for {subject_id!r}")
                            results.append((ip_action, targ, ip_result))
                            result = system.apply_action(action, targets)
                        else:
                            enclosure = system.component(enclosure_id)
                            if enclosure is not None and not getattr(enclosure, "is_inverted", False):
                                inv_action = InvertEnclosure()
                                inv_result = system.apply_action(inv_action, targ:={"subject": enclosure})
                                _logger.info(f"auto-inverted {enclosure_id!r} to satisfy REACHABLE for {subject_id!r}")
                                results.append((inv_action, targ, inv_result))
                                result = system.apply_action(action, targets)
                    elif enclosure_id and "OBSERVABLE" in result.message:
                        from diagnosable_systems_simulation.world.components import InspectionPanel as _InspectionPanel2, Peephole
                        # Prefer InspectionPanel: gives REACHABLE (hence OBSERVABLE) without rotating.
                        inspection_panel2 = next(
                            (c for c in system.all_components().values()
                             if isinstance(c, _InspectionPanel2)
                             and getattr(c, "enclosure_id", None) == enclosure_id
                             and not c.is_open),
                            None,
                        )
                        if inspection_panel2 is not None:
                            ip_action2 = OpenInspectionPanel()
                            ip_result2 = system.apply_action(ip_action2, targ:={"subject": inspection_panel2})
                            _logger.info(f"auto-opened inspection panel {inspection_panel2.component_id!r} to satisfy OBSERVABLE for {subject_id!r}")
                            results.append((ip_action2, targ, ip_result2))
                            result = system.apply_action(action, targets)
                        else:
                            peephole = next(
                                (c for c in system.all_components().values()
                                 if isinstance(c, Peephole) and getattr(c, "enclosure_id", None) == enclosure_id
                                 and not c.is_open),
                                None,
                            )
                            if peephole is not None:
                                ph_action = OpenPeephole()
                                ph_result = system.apply_action(ph_action, targ:={"subject": peephole})
                                _logger.info(f"auto-opened peephole {peephole.component_id!r} to satisfy OBSERVABLE for {subject_id!r}")
                                results.append((ph_action, targ, ph_result))
                                result = system.apply_action(action, targets)
                            else:
                                # No panel or peephole: invert the enclosure (REACHABLE implies OBSERVABLE).
                                enclosure = system.component(enclosure_id)
                                if enclosure is not None and not getattr(enclosure, "is_inverted", False):
                                    inv_action = InvertEnclosure()
                                    inv_result = system.apply_action(inv_action, targ:={"subject": enclosure})
                                    _logger.info(f"auto-inverted {enclosure_id!r} to satisfy OBSERVABLE for {subject_id!r}")
                                    results.append((inv_action, targ, inv_result))
                                    result = system.apply_action(action, targets)
                # Propagate cost of auto-access actions taken inside
                # InspectConnections.execute() (cable port-enclosure opening).
                if action_id == "inspect_connections" and result.success and result.observation:
                    obs_lookup = {p.name: p.value for p in result.observation.properties}
                    for _panel_cid in filter(None, obs_lookup.get("auto_opened_panel_ids", "").split("; ")):
                        results.append((OpenInspectionPanel(), {"subject":_panel_cid}, ActionResult(message=f"auto-opened panel {_panel_cid} for cable inspection")))
                    for _enc_cid in filter(None, obs_lookup.get("auto_inverted_enclosure_ids", "").split("; ")):
                        results.append((InvertEnclosure(), {"subject":_enc_cid}, ActionResult(message=f"auto-inverted enclosure {_enc_cid} for cable inspection")))
                results.append((action, targets, result))
                continue
        except Exception as exc:
            import traceback
            if _logger:
                _logger.debug(f"[{action_id}] exception:\n{traceback.format_exc()}")
            result = ActionResult(success=False, message=f"[{action_id}] error: {exc}")
            targets = dict()
            action = type("_stub", (), {"action_id": action_id, "cost": ActionCost()})()
        results.append((action, targets, result))
    return results


# ---------------------------------------------------------------------------
# Step 3: verbalize
# ---------------------------------------------------------------------------

_verbalize_prompt_free ="""\
You are going to receive a description of an action executed on a system by an engineer. The action is divided into 1 or more steps, together with the resulting step outcomes.
You are to process the results and give feedback to engineer by summarizing diagnostic results in 1–3 plain sentences for the engineer.
In this case, limit yourself strictly to the information in the user prompt — do not add opinions, causes, or extra remarks.
Inthis case, also observe the following rule:
Critical rule — polarity inversions:
If a (+)-labeled cable or port is connected to a (−)-labeled cable or port (or vice versa), you MUST
explicitly state this as a POLARITY INVERSION and name the affected cables. Do not describe such a
connection as "correct" or "nominal". Example: "POLARITY INVERSION DETECTED: PSU Output Cable (+)
is connected to Control Input Cable (−), and PSU Output Cable (−) is connected to Control Input Cable (+). 
Of course, the presence of a small negative current in one cable, by itself, does not amount to polarity inversion. "
"""
_verbalize_prompt_costrained ="""\
You are going to receive a description of an action executed on a system by an engineer. The action is divided into 1 or more steps, together with the resulting step outcomes.
You are to process the results and give feedback to engineer by attaining yourself to the reporting requirements below: 
"""
def _verbalize(results: list[tuple], original_text: str, model: str = MODEL, reporting_requirements: "str | None" = None, logger: Logger | None = None) -> str:
    full_description = original_text
    if not reporting_requirements:
        system_prompt = _verbalize_prompt_free + "\n\nACTION:\n" + original_text
    else:
        system_prompt = _verbalize_prompt_costrained + "\n\nACTION:\n" + original_text + "\n\nREPORTING REQUIREMENTS:\n" + reporting_requirements
        
    lines = ["\nSTEPS:\n"]
    for action, targets, result in results:
        action_id = getattr(action, "action_id", "n.a.")
        description = getattr(action, "description", "n.a.")
        targets = targets or 'n.a.'
        lines.append(f"Step description: action_id=\"{action_id}\"; action_description=\"{description}\"; targets=\"{targets}\"\nstep outcome success: {result.success}, step outcome message: {result.message}")
        if result.observation:
            for p in result.observation.properties:
                unit = f" {p.unit}" if p.unit else ""
                lines.append(f"  {p.name}: {p.value}{unit}")
        lines.append("\n")
    if not lines:
        # No executed actions — include the description so the LLM can still
        # honour any reporting requirements (e.g. return a verdict token).
        lines.append(f"action_description: {full_description}, no actions were executed.")
    user_prompt = "\n".join(lines)

    if logger:
        logger.debug(f"dynamic system prompt in verbalize function:\n{system_prompt}")
        logger.debug(f"dynamic user prompt in verbalize function:\n{user_prompt}")
    if not logger:
        raise Exception("Why did this happen?")
    return _client().create(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_output_tokens=256,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(text: str, system: DiagnosableSystem, model: str = MODEL, mode: Literal['verify', 'collect_information'] = 'verify', _logger: Logger | None = None, reporting_requirements: "str | None" = None) -> tuple[str, ActionCost, list[dict], list[tuple]]:
    """
    Parse *text*, execute the implied actions on *system*, and return a
    plain-language summary together with the total action cost.
    Return textual outcome, cost, parsed actions, action results

    mode: 'verify', restricts the action registry to a repair action to verify an hypothesis.  
        'collect_information' prevents the NL agent from attempting repair or
        fault-injection actions during ordinary diagnosis.
    """
    
    match mode:
        case 'verify':
            allowed_actions = {'verify_repair'}
        case 'collect_information':
            allowed_actions = _DIAGNOSTIC_ALLOWED_ACTIONS
        case _:
            raise ValueError(f"Got {mode} mode. Should have been Literal['verify', 'collect_information'] type.")
    
    entries = _parse(text, system, model, allowed_actions, _logger)

    if not entries:
        base = (
            "The requested action could not be mapped to any recognized "
            "diagnostic operation. Only actions from the available action "
            "list are supported (e.g. measure_voltage, test_continuity, "
            "inspect_connections, verify_repair, …)."
        )
        # If there are reporting requirements (e.g. ANOMALOUS/NOMINAL verdict),
        # honour them so that callers depending on the verdict token do not fail.
        if reporting_requirements:
            narrative = _verbalize(
                results=[],
                original_text=base,
                model=model,
                reporting_requirements=reporting_requirements,
                logger=_logger
            )
        else:
            narrative = base
        return narrative, ActionCost(), entries, []

    if _logger:
        _logger.debug(f"_parse output: {entries}")
    results = _execute(entries, system, allowed_actions, _logger)
    if _logger:
        _logger.debug(f"_execute output: {results}")

    resources: dict[str, float] = {}
    actions = [a for a, _, _ in results]
    for a in actions:
        for k, v in a.cost.resources_consumed.items():
            resources[k] = resources.get(k, 0) + v
    total = ActionCost(
        time=sum(a.cost.time for a in actions),
        equipment=list({e for a in actions for e in a.cost.equipment}),
        resources_consumed=resources,
    )

    return _verbalize(results=results, original_text=text, model=model, reporting_requirements=reporting_requirements, logger=_logger), total, entries, results
