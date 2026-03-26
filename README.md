# diagnosable-systems-simulation

A Python framework for building, simulating, and diagnosing **physical systems** — systems that can fail and be reasoned about.

The framework is structured in four layers:

```
world/               — components, ports, affordances, spatial model
electrical_simulation/  — circuit graph, MNA solver, PySpice/ngspice backend
actions/             — diagnostic and fault-injection actions
systems/             — concrete system definitions
```

A companion package [`nl_interface`](nl_interface/) provides a natural-language interface on top, powered by an LLM.

---

## Installation

```bash
pip install diagnosable-systems-simulation          # core only (numpy)
pip install "diagnosable-systems-simulation[spice]" # + PySpice/ngspice backend
pip install "diagnosable-systems-simulation[llm]"   # + OpenAI / Anthropic (for nl_interface)
pip install "diagnosable-systems-simulation[all]"   # everything
```

For development:

```bash
git clone https://github.com/kataph/diagnosable-systems-simulation
cd diagnosable-systems-simulation
pip install -e ".[all]" --config-settings editable_mode=compat
```

---

## Quick start

```python
from diagnosable_systems_simulation.systems.three_cubes.factory import build_three_cubes_system
from diagnosable_systems_simulation.electrical_simulation.backend.stub import StubBackend

system = build_three_cubes_system(backend=StubBackend(), extra_tools={"multimeter"})
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

### Simulation dump

```python
from diagnosable_systems_simulation.utils.dump import dump_electrical, dump_state

print(dump_electrical(system.last_result, system.graph))  # node voltages + branch currents
print(dump_state(system))                                  # full per-component report
```

### Natural language interface

```python
from nl_interface import run

narrative, cost = run("measure voltage at the main bulb", system)
print(narrative)
```

---

## The three-cubes system

The included example system models a three-module lamp:

- **PSU cube** — 12 V source, status LED, output cables
- **Control cube** — on/off switch, polarity-indicator LED, interconnect cables
- **Load cube** — protection diode, main bulb, internal indicator bulb

Pre-built fault scenarios (S0–S5) cover disconnected cables, burned bulbs, depleted/reversed supply, crossed wires, and stuck switches.

---

## Simulation backends

| Backend | Requires | Notes |
|---|---|---|
| `StubBackend` | numpy only | Fast MNA solver, good for testing |
| `PySpiceBackend` | PySpice + ngspice | Full SPICE `.op` analysis |

---

## Running tests

```bash
pytest tests/test_simulation.py     # 48 tests, parametrized over both backends
```
