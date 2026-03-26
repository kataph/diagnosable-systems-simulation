"""
Component definitions for the 10-cubes lamp system.

Physical layout (top-down, all in metres):
  Power supply cube   x=0.00
  Control cube 1      x=0.15
  Control cube 2      x=0.30
  …
  Control cube 8      x=1.20
  Load cube           x=1.35

Each cube is 0.10 m on a side (5 cm gap between adjacent cubes).

The 10-cubes control modules differ from the 3-cubes control module:
  • Each has a green LED (power-flow indicator) whose anode is connected
    via a resistor to the 12 V output net (switch negative port side).
    The cathode is connected to the ground line.
  • There is no indicator LED on the input side of the switch.

The protection diode lives in the load module, same as for 3-cubes.

Call ``create_components()`` to get a fresh, independent set of component
instances.  Never share instances across ``DiagnosableSystem`` objects.
"""
from __future__ import annotations

from types import SimpleNamespace

from ..shared.module_builders import (
    create_10cubes_control_module,
    create_load_module,
    create_psu_module,
)

NUM_CTRL = 8


def create_components() -> SimpleNamespace:
    """
    Build and return a fresh, fully independent set of component instances
    for the 10-cubes system.

    Returns a ``SimpleNamespace`` with:
      psu       — PSU module namespace
      ctrl_mods — list of 8 control module namespaces (index 0 = module 1)
      load      — load module namespace
      ALL       — ``{component_id: component}`` dict (all modules combined)
    """
    psu = create_psu_module(x_left=0.00)

    ctrl_mods = [
        create_10cubes_control_module(
            prefix=f"ctrl{i}",
            x_left=0.15 * i,
            label=str(i),
        )
        for i in range(1, NUM_CTRL + 1)
    ]

    # Load cube sits one slot after the last control cube.
    load = create_load_module(x_left=0.15 * (NUM_CTRL + 1))

    all_comps: dict = {**psu.ALL}
    for ctrl in ctrl_mods:
        all_comps.update(ctrl.ALL)
    all_comps.update(load.ALL)

    ns = SimpleNamespace(psu=psu, ctrl_mods=ctrl_mods, load=load)
    ns.ALL = all_comps
    return ns
