# diagnosable-systems-simulation

A Python framework for building, simulating, and diagnosing **physical systems** — systems that can fail and be reasoned about.

The framework is structured in four layers:

```
world/                  — components, ports, affordances, spatial model
electrical_simulation/  — circuit graph, MNA solver, PySpice/ngspice backend
actions/                — diagnostic and fault-injection actions
systems/                — concrete system definitions
```

A companion package [`nl_interface`](nl_interface/) provides a natural-language interface on top, powered by an LLM.

---

## Installation

```bash
pip install "diagnosable-systems-simulation[spice]"      # simulation only (PySpice/ngspice)
pip install "diagnosable-systems-simulation[llm]"        # + OpenAI / Anthropic (for nl_interface)
pip install "diagnosable-systems-simulation[all]"        # everything
```

For development:

```bash
git clone https://github.com/kataph/diagnosable-systems-simulation
cd diagnosable-systems-simulation
pip install -e ".[all]" --config-settings editable_mode=compat
```

> **ngspice** must be installed separately and available on `PATH`.  
> macOS: `brew install ngspice` · Linux: `apt install ngspice`

---

## Quick start

```python
from diagnosable_systems_simulation.systems.three_cubes.factory import build_three_cubes_system
from diagnosable_systems_simulation.electrical_simulation.backend.spice import PySpiceBackend

system = build_three_cubes_system(backend=PySpiceBackend(), extra_tools={"multimeter"})
result = system.simulate()
print(result)                        # SimulationResult(nodes=14, converged=True, lit=[...])
print(result.is_lit("main_bulb"))    # True
```

### Applying diagnostic actions

```python
from diagnosable_systems_simulation.actions.diagnostic_actions import MeasureVoltage

outcome = system.apply_action(
    MeasureVoltage(),
    {"subject": system.component("main_bulb")},
)
print(outcome.observation)
```

### Injecting faults

```python
from diagnosable_systems_simulation.actions.fault_actions import DisconnectCable

system.inject_fault(
    DisconnectCable(port_names=["p", "n"]),
    {"cable": system.component("psu_cable_pos")},
)
result = system.simulate()
print(result.is_lit("main_bulb"))    # False
```

### Intermittent faults

```python
from diagnosable_systems_simulation.electrical_simulation.couplings import _add_loose_connection

_add_loose_connection(system, "psu_cable_pos", "p", p=0.5)
# port disconnects randomly with probability p on each solver step
```

### Natural language interface

```python
from nl_interface.interface import run

narrative, cost, entries, results = run(
    "measure voltage at the main bulb",
    system,
    mode="collect_information",
)
print(narrative)   # plain-English summary of findings
print(cost.time)   # estimated technician time in seconds
```

---

## The three-cubes system

The included example system models a three-module lamp assembly:

- **PSU cube** — 12 V source, status LED, output cables
- **Control cube** — on/off switch, polarity-indicator LED, interconnect cables
- **Load cube** — protection diode, main bulb, internal indicator bulb

Pre-built fault scenarios (S0–S5) cover disconnected cables, burned bulbs, depleted/reversed supply, crossed wires, and stuck switches.

---

## Simulation backend

| Backend | Requires | Notes |
|---|---|---|
| `PySpiceBackend` | PySpice + ngspice | Full SPICE `.op` analysis |

---

## Action costs

Each action carries an `ActionCost` (time in seconds, required equipment, consumed resources).
See [`ACTION_COSTS.md`](ACTION_COSTS.md) for the full rationale table.

---

## Running tests

```bash
pytest tests/                        # full suite (~110 tests)
SKIP_LLM_TESTS=0 pytest tests/       # include live LLM tests (requires API key)
```
