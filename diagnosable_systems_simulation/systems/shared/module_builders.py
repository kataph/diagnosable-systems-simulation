"""
Reusable module-level component builders.

Each builder creates a fresh, independent set of component instances for one
functional module (PSU, control, or load).  All x-positions are given relative
to the module's left-edge x-coordinate so the same builders can be placed
anywhere along the x-axis.

Two distinct control-module builders are provided because the 3-cubes and
10-cubes control modules differ in LED colour, LED orientation, and the
position of the indicator LED relative to the switch:

create_3cubes_control_module
    Red LED with cathode → (resistor) → 12 V *input* net (= switch.p side).
    Anode → ground net.  The LED is reverse-biased in normal operation and
    lights only when the cables are crossed (inverted-polarity indicator).

create_10cubes_control_module
    Green LED with anode → (resistor) → 12 V *output* net (= switch.n side).
    Cathode → ground net.  The LED lights whenever the switch is closed and
    current flows, indicating that the module is passing power.

The protection diode lives in the load module for both systems.
"""
from __future__ import annotations

from types import SimpleNamespace

from diagnosable_systems_simulation.world.affordances import (
    Affordance, AffordanceSet, ConditionalAffordance,
)
from diagnosable_systems_simulation.world.components import (
    Bulb, Cable, Diode, LED, Peephole, PhysicalEnclosure, Resistor, Switch, VoltageSource,
)
from diagnosable_systems_simulation.world.spatial import Position


def _when_inverted(enclosure, peephole=None):
    """Return a condition callable: True when *enclosure* is inverted or *peephole* is open."""
    def condition(component, _ctx):
        return enclosure.is_inverted or (peephole is not None and peephole.is_open)
    return condition


def _when_inverted_only(enclosure):
    """Return a condition callable: True only when *enclosure* is inverted (not when peephole open).

    Used for REACHABLE: a peephole lets you see inside but not physically probe or manipulate.
    """
    def condition(component, _ctx):
        return enclosure.is_inverted
    return condition


# ---------------------------------------------------------------------------
# PSU module  (identical for all systems)
# ---------------------------------------------------------------------------

def create_psu_module(x_left: float = 0.0) -> SimpleNamespace:
    """
    Build and return a fresh Power Supply module.

    Returned namespace attributes
    ------------------------------
    cube, source, green_led, green_resistor, cable_pos, cable_neg
    ALL  — ``{component_id: component}`` dict
    """
    x = x_left

    cube = PhysicalEnclosure(
        component_id="cube_psu",
        display_name="Power Supply Cube",
        position=Position(x + 0.05, 0.05, 0.05),
    )
    source = VoltageSource(
        component_id="battery",
        display_name="Battery",
        voltage=12.0,
        position=Position(x + 0.05, 0.05, 0.05),
        enclosure_id="cube_psu",
    )
    source.affordances = AffordanceSet(
        static={Affordance.MEASURABLE},
        conditional=[
            ConditionalAffordance(
                Affordance.REACHABLE, _when_inverted_only(cube),
                "reachable when PSU cube is inverted",
            ),
        ],
    )
    green_led = LED(
        component_id="psu_green_led",
        display_name="PSU Status LED (green)",
        forward_voltage=2.1,
        forward_current=0.01,
        color="green",
        position=Position(x + 0.05, 0.05, 0.10),
        enclosure_id="cube_psu",
    )
    green_led.affordances = AffordanceSet(
        static={Affordance.REACHABLE, Affordance.MEASURABLE, Affordance.REPLACEABLE},
    )
    green_resistor = Resistor(
        component_id="psu_green_resistor",
        display_name="PSU Green LED Resistor",
        resistance=1000.0,
        position=Position(x + 0.05, 0.05, 0.08),
        enclosure_id="cube_psu",
    )
    green_resistor.affordances = AffordanceSet(
        static={Affordance.MEASURABLE, Affordance.REPLACEABLE},
        conditional=[ConditionalAffordance(
            Affordance.REACHABLE, _when_inverted_only(cube),
            "reachable when PSU cube is inverted",
        )],
    )
    cable_pos = Cable(
        component_id="psu_cable_pos",
        display_name="PSU Output Cable (+)",
        position=Position(x + 0.10, 0.05, 0.05),
    )
    cable_neg = Cable(
        component_id="psu_cable_neg",
        display_name="PSU Output Cable (−)",
        position=Position(x + 0.10, 0.05, 0.03),
    )
    # p-terminals connect inside the PSU cube; n-terminals are external connectors.
    cable_pos.port_enclosures = {"p": cube.component_id}
    cable_neg.port_enclosures = {"p": cube.component_id}

    # LED-slot metadata used by MoveLED action.
    # The slot anchors are the components/ports the resistor.p and LED.cathode
    # connect to in this module.  They stay in the circuit even when the LED
    # and resistor are removed, so MoveLED can find the correct node IDs.
    green_led._series_resistor_id = green_resistor.component_id
    cube._led_slot_pos = (source, "pos")   # resistor.p connects here
    cube._led_slot_neg = (source, "neg")   # LED.cathode connects here (GND)

    all_comps = [cube, source, green_led, green_resistor, cable_pos, cable_neg]
    ns = SimpleNamespace(
        cube=cube, source=source, green_led=green_led,
        green_resistor=green_resistor, cable_pos=cable_pos, cable_neg=cable_neg,
    )
    ns.ALL = {c.component_id: c for c in all_comps}
    return ns


# ---------------------------------------------------------------------------
# 3-cubes control module
# ---------------------------------------------------------------------------

def create_3cubes_control_module(
    prefix: str = "ctrl",
    x_left: float = 0.15,
) -> SimpleNamespace:
    """
    Build the control module used in the **3-cubes** system.

    Indicator LED
    -------------
    Red LED, reverse-polarity / inverted-cables indicator.
    The cathode is connected (via a current-limiting resistor) to the same net
    as the 12 V input line and the switch positive port.  The anode is
    connected to the ground (negative) line.  In normal operation the LED is
    reverse-biased and dark; it lights up only when the supply cables are
    crossed.

    NOMINAL inspect_connections observation for this LED
    ----------------------------------------------------
    The ctrl_in_n (ground) net collects three endpoints: the anode of the LED,
    the far end of the input negative cable (ctrl_cable_in_neg.n), and the near
    end of the output negative cable (ctrl_cable_out_neg.p).  Therefore,
    inspect_connections will ALWAYS report two cables at the anode and no cable
    at the cathode — even in the healthy system.  This is not a fault; it is
    the designed topology.  The cathode connects only to the internal indicator
    resistor, which is not a Cable object, so no cable end appears there.

    Component IDs use ``prefix`` as a stem, e.g. with ``prefix="ctrl"``:
        cube_ctrl, ctrl_switch, ctrl_red_led, ctrl_red_resistor,
        ctrl_cable_in_pos, ctrl_cable_in_neg,
        ctrl_cable_out_pos, ctrl_cable_out_neg

    Returned namespace attributes
    ------------------------------
    cube, switch, red_led, red_resistor,
    cable_in_pos, cable_in_neg, cable_out_pos, cable_out_neg
    ALL  — ``{component_id: component}`` dict
    """
    p = prefix + "_"
    x = x_left

    cube = PhysicalEnclosure(
        component_id=f"cube_{prefix}",
        display_name="Control Cube",
        position=Position(x + 0.05, 0.05, 0.05),
    )
    switch = Switch(
        component_id=f"{p}switch",
        display_name="Control Switch",
        is_closed=True,
        position=Position(x, 0.10, 0.05),
        enclosure_id=f"cube_{prefix}",
    )
    switch.affordances = AffordanceSet(
        static={Affordance.TOGGLABLE, Affordance.MEASURABLE},
        conditional=[ConditionalAffordance(
            Affordance.REACHABLE, _when_inverted_only(cube),
            "reachable when control cube is inverted",
        )],
    )
    red_led = LED(
        component_id=f"{p}red_led",
        display_name="Inverted Cables Indicator (red)",
        forward_voltage=2.0,
        forward_current=0.01,
        color="red",
        position=Position(x + 0.05, 0.05, 0.10),
        enclosure_id=f"cube_{prefix}",
    )
    red_led.affordances = AffordanceSet(
        static={Affordance.REACHABLE, Affordance.MEASURABLE, Affordance.REPLACEABLE},
    )
    # NOMINAL inspect_connections result for this LED:
    #   anode  → two cables (Control Input Cable (−) and Control Output Cable (−)) —
    #            both negative inter-module cables share this ground junction.
    #   cathode → no cable — connects only to the internal indicator resistor.
    # Seeing two cables at the anode and none at the cathode is EXPECTED and does NOT
    # indicate a fault.  The LED lights only when anode > cathode, i.e. when the
    # cables are inverted and ground is routed to the cathode side.
    red_led._nominal_observation_note = (
        "NOMINAL: anode always shows two cables (both negative inter-module cables share "
        "this ground junction); cathode shows no cable (internal resistor only). "
        "Two cables at anode is NOT a fault indicator."
    )
    red_resistor = Resistor(
        component_id=f"{p}red_resistor",
        display_name="Inverted Cables Indicator Resistor",
        resistance=1000.0,
        position=Position(x + 0.05, 0.05, 0.08),
        enclosure_id=f"cube_{prefix}",
    )
    red_resistor.affordances = AffordanceSet(
        static={Affordance.MEASURABLE, Affordance.REPLACEABLE},
        conditional=[ConditionalAffordance(
            Affordance.REACHABLE, _when_inverted_only(cube),
            "reachable when control cube is inverted",
        )],
    )
    cable_in_pos  = Cable(f"{p}cable_in_pos",  "Control Input Cable (+)",  position=Position(x,        0.05, 0.07))
    cable_in_neg  = Cable(f"{p}cable_in_neg",  "Control Input Cable (−)",  position=Position(x,        0.05, 0.03))
    cable_out_pos = Cable(f"{p}cable_out_pos", "Control Output Cable (+)", position=Position(x + 0.10, 0.05, 0.07))
    cable_out_neg = Cable(f"{p}cable_out_neg", "Control Output Cable (−)", position=Position(x + 0.10, 0.05, 0.03))
    # n-terminals of input cables connect inside the ctrl cube (switch.p and ground junction).
    # p-terminals of output cables connect inside the ctrl cube (switch.n and ground junction).
    cable_in_pos.port_enclosures  = {"n": cube.component_id}
    cable_in_neg.port_enclosures  = {"n": cube.component_id}
    cable_out_pos.port_enclosures = {"p": cube.component_id}
    cable_out_neg.port_enclosures = {"p": cube.component_id}

    all_comps = [cube, switch, red_led, red_resistor,
                 cable_in_pos, cable_in_neg, cable_out_pos, cable_out_neg]
    ns = SimpleNamespace(
        cube=cube, switch=switch, red_led=red_led, red_resistor=red_resistor,
        cable_in_pos=cable_in_pos, cable_in_neg=cable_in_neg,
        cable_out_pos=cable_out_pos, cable_out_neg=cable_out_neg,
    )
    ns.ALL = {c.component_id: c for c in all_comps}
    return ns


# ---------------------------------------------------------------------------
# 10-cubes control module
# ---------------------------------------------------------------------------

def create_10cubes_control_module(
    prefix: str = "ctrl1",
    x_left: float = 0.15,
    label: str | None = None,
) -> SimpleNamespace:
    """
    Build a single control module for the **10-cubes** system.

    Indicator LED
    -------------
    Green LED, power-flow indicator.
    The anode is connected (via a current-limiting resistor) to the same net
    as the 12 V *output* line and the switch negative port.  The cathode is
    connected to the ground (negative) line.  The LED lights whenever the
    switch is closed and current flows out of the module.

    Component IDs use ``prefix`` as a stem, e.g. with ``prefix="ctrl1"``:
        cube_ctrl1, ctrl1_switch, ctrl1_green_led, ctrl1_green_resistor,
        ctrl1_cable_in_pos, ctrl1_cable_in_neg,
        ctrl1_cable_out_pos, ctrl1_cable_out_neg

    Parameters
    ----------
    label
        Human-readable number appended to display names, e.g. ``"1"`` →
        "Control Switch 1".  *None* produces plain names.

    Returned namespace attributes
    ------------------------------
    cube, switch, green_led, green_resistor,
    cable_in_pos, cable_in_neg, cable_out_pos, cable_out_neg
    ALL  — ``{component_id: component}`` dict
    """
    p = prefix + "_"
    x = x_left
    lbl = f" {label}" if label is not None else ""

    cube = PhysicalEnclosure(
        component_id=f"cube_{prefix}",
        display_name=f"Control Cube{lbl}",
        position=Position(x + 0.05, 0.05, 0.05),
    )
    switch = Switch(
        component_id=f"{p}switch",
        display_name=f"Control Switch{lbl}",
        is_closed=True,
        position=Position(x, 0.10, 0.05),
        enclosure_id=f"cube_{prefix}",
    )
    switch.affordances = AffordanceSet(
        static={Affordance.TOGGLABLE, Affordance.MEASURABLE},
        conditional=[ConditionalAffordance(
            Affordance.REACHABLE, _when_inverted_only(cube),
            "reachable when control cube is inverted",
        )],
    )
    green_led = LED(
        component_id=f"{p}green_led",
        display_name=f"Power Flow Indicator (green){lbl}",
        forward_voltage=2.1,
        forward_current=0.01,
        color="green",
        position=Position(x + 0.05, 0.05, 0.10),
        enclosure_id=f"cube_{prefix}",
    )
    green_led.affordances = AffordanceSet(
        static={Affordance.REACHABLE, Affordance.MEASURABLE, Affordance.REPLACEABLE},
    )
    green_resistor = Resistor(
        component_id=f"{p}green_resistor",
        display_name=f"Power Flow Indicator Resistor{lbl}",
        resistance=1000.0,
        position=Position(x + 0.05, 0.05, 0.08),
        enclosure_id=f"cube_{prefix}",
    )
    green_resistor.affordances = AffordanceSet(
        static={Affordance.MEASURABLE, Affordance.REPLACEABLE},
        conditional=[ConditionalAffordance(
            Affordance.REACHABLE, _when_inverted_only(cube),
            "reachable when control cube is inverted",
        )],
    )
    cable_in_pos  = Cable(f"{p}cable_in_pos",  f"Control Input Cable (+){lbl}",  position=Position(x,        0.05, 0.07))
    cable_in_neg  = Cable(f"{p}cable_in_neg",  f"Control Input Cable (−){lbl}",  position=Position(x,        0.05, 0.03))
    cable_out_pos = Cable(f"{p}cable_out_pos", f"Control Output Cable (+){lbl}", position=Position(x + 0.10, 0.05, 0.07))
    cable_out_neg = Cable(f"{p}cable_out_neg", f"Control Output Cable (−){lbl}", position=Position(x + 0.10, 0.05, 0.03))
    # n-terminals of input cables connect inside the ctrl cube (switch.p and ground junction).
    # p-terminals of output cables connect inside the ctrl cube (switch.n and ground junction).
    cable_in_pos.port_enclosures  = {"n": cube.component_id}
    cable_in_neg.port_enclosures  = {"n": cube.component_id}
    cable_out_pos.port_enclosures = {"p": cube.component_id}
    cable_out_neg.port_enclosures = {"p": cube.component_id}

    # LED-slot metadata used by MoveLED action.
    # switch.n is the positive anchor (where resistor.p connects);
    # cable_in_neg.n is the negative/GND anchor (where led.cathode connects).
    green_led._series_resistor_id = green_resistor.component_id
    cube._led_slot_pos = (switch, "n")           # resistor.p connects here
    cube._led_slot_neg = (cable_in_neg, "n")     # LED.cathode connects here (GND)

    all_comps = [cube, switch, green_led, green_resistor,
                 cable_in_pos, cable_in_neg, cable_out_pos, cable_out_neg]
    ns = SimpleNamespace(
        cube=cube, switch=switch, green_led=green_led, green_resistor=green_resistor,
        cable_in_pos=cable_in_pos, cable_in_neg=cable_in_neg,
        cable_out_pos=cable_out_pos, cable_out_neg=cable_out_neg,
    )
    ns.ALL = {c.component_id: c for c in all_comps}
    return ns


# ---------------------------------------------------------------------------
# Load module  (identical for all systems — diode always here)
# ---------------------------------------------------------------------------

def create_load_module(x_left: float = 0.30) -> SimpleNamespace:
    """
    Build and return a fresh Load module.

    The protection diode is always part of the load module for all system
    variants (3-cubes and 10-cubes alike).

    Returned namespace attributes
    ------------------------------
    cube, main_bulb, internal_bulb, peephole, diode, cable_pos, cable_neg
    ALL  — ``{component_id: component}`` dict
    """
    x = x_left

    cube = PhysicalEnclosure(
        component_id="cube_load",
        display_name="Load Cube",
        position=Position(x + 0.05, 0.05, 0.05),
    )
    peephole = Peephole(
        component_id="load_peephole",
        display_name="Load Cube Peephole",
        position=Position(x, 0.10, 0.05),
        enclosure_id="cube_load",
    )

    def _visible(component, _ctx):
        """Visible when cube is inverted OR peephole is open (but not necessarily reachable)."""
        return cube.is_inverted or peephole.is_open

    def _reachable(component, _ctx):
        """Physical probe/replacement access requires inverting the cube."""
        return cube.is_inverted

    main_bulb = Bulb(
        component_id="main_bulb",
        display_name="Main Load (lamp)",
        resistance=120.0,
        power_threshold=0.05,
        position=Position(x + 0.05, 0.05, 0.10),
        enclosure_id="cube_load",
    )
    main_bulb.affordances = AffordanceSet(
        static={Affordance.REACHABLE, Affordance.MEASURABLE, Affordance.REPLACEABLE},
    )
    internal_bulb = Bulb(
        component_id="internal_bulb",
        display_name="Internal Indicator Lamp",
        resistance=500.0,
        power_threshold=0.01,
        position=Position(x + 0.05, 0.05, 0.05),
        enclosure_id="cube_load",
    )
    internal_bulb.affordances = AffordanceSet(
        conditional=[
            ConditionalAffordance(
                Affordance.OBSERVABLE, _visible,
                "observable when load cube is inverted or peephole is open",
            ),
            ConditionalAffordance(
                Affordance.REACHABLE, _reachable,
                "reachable when load cube is inverted",
            ),
            ConditionalAffordance(
                Affordance.MEASURABLE, _reachable,
                "measurable only when load cube is inverted (probe access)",
            ),
            ConditionalAffordance(
                Affordance.REPLACEABLE, _reachable,
                "replaceable only when load cube is inverted",
            ),
        ],
    )
    load_diode = Diode(
        component_id="load_diode",
        display_name="Load Protection Diode",
        forward_voltage=0.7,
        position=Position(x + 0.02, 0.05, 0.05),
        enclosure_id="cube_load",
    )
    load_diode.affordances = AffordanceSet(
        conditional=[
            ConditionalAffordance(
                Affordance.OBSERVABLE, _visible,
                "observable when load cube is inverted or peephole is open",
            ),
            ConditionalAffordance(
                Affordance.REACHABLE, _reachable,
                "reachable when load cube is inverted",
            ),
            ConditionalAffordance(
                Affordance.MEASURABLE, _reachable,
                "measurable only when load cube is inverted (probe access)",
            ),
            ConditionalAffordance(
                Affordance.REPLACEABLE, _reachable,
                "replaceable only when load cube is inverted",
            ),
        ],
    )
    cable_pos = Cable("load_cable_pos", "Load Input Cable (+)", position=Position(x,        0.05, 0.07))
    cable_neg = Cable("load_cable_neg", "Load Input Cable (−)", position=Position(x,        0.05, 0.03))
    # n-terminals connect inside the load cube (diode anode and bulb negative rails).
    cable_pos.port_enclosures = {"n": cube.component_id}
    cable_neg.port_enclosures = {"n": cube.component_id}

    all_comps = [cube, peephole, main_bulb, internal_bulb, load_diode, cable_pos, cable_neg]
    ns = SimpleNamespace(
        cube=cube, peephole=peephole, main_bulb=main_bulb, internal_bulb=internal_bulb,
        diode=load_diode, cable_pos=cable_pos, cable_neg=cable_neg,
    )
    ns.ALL = {c.component_id: c for c in all_comps}
    return ns
