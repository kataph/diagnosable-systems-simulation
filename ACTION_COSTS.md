# Action Cost Rationale

Costs represent estimated real-world technician time in seconds.
Relative magnitude is prioritised over absolute precision.

## Tier system

| Tier | Time | Examples |
|---|---|---|
| Quick physical / diagnostic observation | 10s | diagnostic look at a component, toggle switch, open hatch, flip/rotate small enclosure |
| Fine/delicate action | 15s | precise motor targeting, wiggle-test inspection |
| Instrument action | 20s | probe placement + reading |
| Extended instrument action | 45s | two-point probe across separated locations |
| Repair/replacement | 120s | physical swap of a small component |

Cognitive cost (recognition, decision, verification) is included implicitly within each tier.

---

## Diagnostic actions

| Action | Time (s) | Justification |
|---|---|---|
| `observe_component` | 10 | Diagnostic look: actively checking for visible faults (burn marks, LED state, discolouration, damage). Not a glance — a deliberate inspection that produces an observation record. Quick physical tier. |
| `inspect_connections` | 15 | Observe + wiggle test on each port. Detects floating cable ends only (not crossed cables). One tier above observe due to systematic port-by-port check. |
| `open_switch` / `close_switch` | 10 | Single deliberate actuation to a hard stop. Quick physical tier. |
| `test_switch` | 60 | Composite: CloseSwitch(10) + TestContinuity(20) + OpenSwitch(10) + TestContinuity(20) = 60s. Hardcoded to match sub-action sum. |
| `invert_enclosure` | 10 | Small enclosure — pick up and flip. Quick physical tier. |
| `restore_enclosure` | 10 | Return to original position. Same effort as invert. |
| `rotate_enclosure` | 10 | Partial repositioning of a small enclosure. Same quick physical tier. |
| `open_peephole` / `close_peephole` | 5 | Slide or flip a small hatch. Observe tier — minimal effort. |
| `open_inspection_panel` / `close_inspection_panel` | 5 | Same as peephole. |
| `adjust_potentiometer` | 15 | Fine motor control: turn knob to precise target position while reading scale. One tier above quick due to targeting precision. |
| `measure_voltage` | 20 | Retrieve multimeter, place two probes, read display. Standard instrument tier. |
| `measure_current` | 20 | Same as voltage — break circuit, insert ammeter in series, read. |
| `test_continuity` | 20 | Place two probes on component terminals, read continuity tone/value. Standard instrument tier. |
| `test_path_continuity` | 45 | Two probe placements at physically separated endpoints (potentially across modules), plus repositioning between them. ~2× single continuity. |
| `test_diode` | 20 | Place probes on diode terminals, read forward voltage. Standard instrument tier. |
| `replace_component` | 120 | Physical swap: remove old, retrieve new, insert, verify seating. Components are small and accessible. Repair tier. |
| `move_led` | 30 | Unplug LED from socket, reinsert in new slot, verify alignment. Faster than full replace (no screws, same-form-factor swap). Between quick and repair tiers. |
| `short_ports` | 20 | Clip a jumper across two terminals. Standard instrument tier (requires precision placement). |
| `verify_repair` | 0 | Intentional: cost attributed to `apply_repairs()` separately. No double-counting. |

---

## Fault injection actions

> These actions are excluded from the NL diagnostic interface (`_DIAGNOSTIC_ALLOWED_ACTIONS`)
> and cannot be called by the LLM agent. Cost only applies to test setup (`inject_fault()`).

| Action | Time (s) | Justification |
|---|---|---|
| `disconnect_cable` | 10 | Pull a connector out. Single quick physical action. |
| `reconnect_cable` | 10 | Push a connector in. Single quick physical action. |
| `short_circuit` | 30 | Retrieve jumper wire, identify two terminals, clip both ends. Above quick tier due to two-step placement. |
| `degrade_component` | 120 | Modelled as physically swapping the component for a degraded one. Repair tier. |
| `blow_fuse` | 120 | Modelled as physically replacing the fuse with a blown one. Repair tier. |
| `force_switch` | 120 | Modelled as physically jamming/replacing the switch mechanism. Repair tier. |
| `reverse_battery` | 30 | Disconnect + reinsert connector in reverse polarity. Two physical steps. |
| `swap_cable_polarities` | 40 | Composite: 2× DisconnectCable(10) + 2× ReconnectCable(10) = 40s. Derived. |

---

## Potentially criticable values

| Action | Risk | Defence |
|---|---|---|
| `test_path_continuity` (45s) | May be too slow if endpoints are adjacent | Worst-case estimate; adjacent endpoints would rarely warrant a path continuity test over a simpler single-point continuity. |
| `disconnect_cable` / `reconnect_cable` (10s) | Too fast for connectors with locking mechanisms | These systems use snap-fit or friction-fit connectors. Locking connectors would warrant 20–30s. |
| `move_led` (30s) | Could be faster (5s) for a bare-socket LED | Includes visual alignment check to ensure correct polarity and seating. The 30s covers insertion + verify. |
| `degrade_component` / `blow_fuse` / `force_switch` (120s) | These are simulation injections, not real acts | Intentional: cost is set to reflect what the equivalent physical fault-introduction would cost a lab technician, for realism in scenario design. |
