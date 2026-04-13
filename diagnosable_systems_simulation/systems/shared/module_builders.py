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
    Bulb, Cable, Diode, InspectionPanel, LED, LightSensor, Module, Peephole,
    PhysicalEnclosure, Potentiometer, Resistor, Switch, VoltageSource,
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

    module = Module(
        component_id="module_psu",
        display_name="Power Supply Module",
        position=Position(x + 0.05, 0.05, 0.05),
    )
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
        static={Affordance.MEASURABLE, Affordance.REPLACEABLE},
        conditional=[
            ConditionalAffordance(
                Affordance.REACHABLE, _when_inverted_only(cube),
                "reachable when PSU cube is inverted",
            ),
        ],
    )
    battery_internal_resistor = Resistor(
        component_id="battery_internal_resistor",
        display_name="Battery Internal Resistor",
        resistance=1.0,
        position=Position(x + 0.05, 0.05, 0.05),
        enclosure_id="cube_psu",
    )
    battery_internal_resistor.affordances = AffordanceSet(
        static={Affordance.MEASURABLE, Affordance.REPLACEABLE},
        conditional=[ConditionalAffordance(
            Affordance.REACHABLE, _when_inverted_only(cube),
            "reachable when PSU cube is inverted",
        )],
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

    all_comps = [module, cube, source, green_led, green_resistor, cable_pos, cable_neg, battery_internal_resistor]
    ns = SimpleNamespace(
        module=module, cube=cube, source=source, green_led=green_led,
        green_resistor=green_resistor, cable_pos=cable_pos, cable_neg=cable_neg,
        battery_internal_resistor=battery_internal_resistor,
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

    module = Module(
        component_id=f"control_module_{prefix}",
        display_name=f"Control Module",
        position=Position(x + 0.05, 0.05, 0.05),
    )
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

    all_comps = [module, cube, switch, red_led, red_resistor,
                 cable_in_pos, cable_in_neg, cable_out_pos, cable_out_neg]
    ns = SimpleNamespace(
        module=module, cube=cube, switch=switch, red_led=red_led, red_resistor=red_resistor,
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

    module = Module(
        component_id=f"module_{prefix}",
        display_name=f"Control Module{lbl}",
        position=Position(x + 0.05, 0.05, 0.05),
    )
    
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

    all_comps = [module, cube, switch, green_led, green_resistor,
                 cable_in_pos, cable_in_neg, cable_out_pos, cable_out_neg]
    ns = SimpleNamespace(
        module=module, cube=cube, switch=switch, green_led=green_led, green_resistor=green_resistor,
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

    module = Module(
        component_id="module_load",
        display_name="Load Module",
        position=Position(x + 0.05, 0.05, 0.05),
    )
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
    peephole._nominal_observation_note = (
        "This is an access hole on the load cube, not a sensor or light source. "
        "To look inside: first call open_peephole on load_peephole, "
        "then call observe_component on internal_bulb."
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
    internal_bulb._nominal_observation_note = (
        "DIAGNOSTIC: This lamp is wired in parallel with the main load. "
        "It lights when the main load carries no current (open-circuit fault). "
        "A lit lamp indicates a main load failure — treat as ANOMALOUS, not NOMINAL."
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

    all_comps = [module, cube, peephole, main_bulb, internal_bulb, load_diode, cable_pos, cable_neg]
    ns = SimpleNamespace(
        module=module, cube=cube, peephole=peephole, main_bulb=main_bulb, internal_bulb=internal_bulb,
        diode=load_diode, cable_pos=cable_pos, cable_neg=cable_neg,
    )
    ns.ALL = {c.component_id: c for c in all_comps}
    return ns


# ---------------------------------------------------------------------------
# Ambient light sensor control module
# ---------------------------------------------------------------------------

def create_ambient_ctrl_module(prefix: str = "ctrl") -> SimpleNamespace:
    """
    Build the control module for the **ambient light sensor** system.

    Unlike the switch-based 3-cubes module, this one controls the lamp via:

    - An LDR (``ctrl_light_sensor``): high resistance in the dark, low when lit.
    - A relay (``ctrl_relay``): closed when sensor is dark (normal), opens when
      the sensor detects light above the threshold.
    - A bias resistor (``ctrl_sensor_bias``): 10 kΩ in series with the LDR,
      forming a voltage divider on the ctrl_in_p rail.
    - Two potentiometers (``ctrl_sensitivity_pot``, ``ctrl_timing_pot``):
      wired across the supply rail (p→ctrl_in_p, n→ctrl_in_n), wiper left
      unconnected.  They appear accessible and adjustable but have no effect
      on the feedback coupling that controls the relay.

    Circuit topology (all nets relative to ``ctrl_cable_in_pos.n``):

        ctrl_in_p net:
            ctrl_cable_in_pos.n → ctrl_relay.p
                               → ctrl_sensor_bias.p
                               → ctrl_sensitivity_pot.p
                               → ctrl_timing_pot.p

        sensor_mid net:
            ctrl_sensor_bias.n → ctrl_light_sensor.p

        relay_out net:
            ctrl_relay.n       → ctrl_cable_out_pos.p

        ctrl_in_n / GND net (= ctrl_cable_in_neg.n):
            ctrl_cable_in_neg.n → ctrl_cable_out_neg.p
                                → ctrl_light_sensor.n
                                → ctrl_sensitivity_pot.n
                                → ctrl_timing_pot.n

    Affordances
    -----------
    ``cube_ctrl``          : REACHABLE + MOVABLE (default for PhysicalEnclosure)
    ``ctrl_panel``         : REACHABLE + OPENABLE/CLOSEABLE (inspection panel on the side face)
    ``ctrl_light_sensor``  : OBSERVABLE always; REACHABLE when panel open OR cube rotated
    ``ctrl_relay``         : OBSERVABLE + REACHABLE when panel open OR cube rotated
    ``ctrl_sensitivity_pot``: OBSERVABLE + REACHABLE + ADJUSTABLE always (bait)
    ``ctrl_timing_pot``    : OBSERVABLE + REACHABLE + ADJUSTABLE always (bait)
    ``ctrl_sensor_bias``   : OBSERVABLE + REACHABLE when panel open OR cube rotated

    The panel enables REACHABLE access for electrical measurements *without*
    rotating/moving the enclosure.  Opening the panel does NOT break the
    optical feedback path (the coupling only checks ``cube.is_inverted``).

    Returned namespace attributes
    ------------------------------
    module, cube, panel, light_sensor, relay,
    sensitivity_pot, timing_pot, sensor_bias,
    cable_in_pos, cable_in_neg, cable_out_pos, cable_out_neg
    ALL  — ``{component_id: component}`` dict
    """
    p = prefix + "_"

    module = Module(
        component_id=f"control_module_{prefix}",
        display_name="Control Module",
        position=Position(0.05, 0.05, 0.20),
    )
    cube = PhysicalEnclosure(
        component_id=f"cube_{prefix}",
        display_name="Control Cube",
        position=Position(0.05, 0.05, 0.20),
    )
    panel = InspectionPanel(
        component_id=f"{p}panel",
        display_name="Control Module Inspection Panel",
        position=Position(0.00, 0.05, 0.20),
        enclosure_id=f"cube_{prefix}",
    )
    panel._nominal_observation_note = (
        "This is a removable side panel on the control cube used for electrical inspection. "
        "Opening it gives probe access to internal components without rotating the cube. "
        "Rotating the cube (move/rotate enclosure) is a separate physical action."
    )

    # _when_inverted(cube, peephole=panel) returns True when
    # cube.is_inverted OR panel.is_open — used for OBSERVABLE/REACHABLE.
    _panel_or_inverted = _when_inverted(cube, peephole=panel)

    light_sensor = LightSensor(
        component_id=f"{p}light_sensor",
        display_name="Ambient Light Sensor",
        resistance_dark=10_000.0,
        resistance_lit=100.0,
        position=Position(0.05, 0.05, 0.20),
        enclosure_id=f"cube_{prefix}",
    )
    light_sensor.affordances = AffordanceSet(
        static={Affordance.OBSERVABLE, Affordance.MEASURABLE},
        conditional=[
            ConditionalAffordance(
                Affordance.REACHABLE, _panel_or_inverted,
                "reachable when inspection panel is open or control cube is rotated",
            ),
        ],
    )
    relay = Switch(
        component_id=f"{p}relay",
        display_name="Control Relay",
        is_closed=True,
        position=Position(0.05, 0.05, 0.18),
        enclosure_id=f"cube_{prefix}",
    )
    relay.affordances = AffordanceSet(
        static={Affordance.MEASURABLE},
        conditional=[
            ConditionalAffordance(
                Affordance.OBSERVABLE, _panel_or_inverted,
                "observable when inspection panel is open or control cube is rotated",
            ),
            ConditionalAffordance(
                Affordance.REACHABLE, _panel_or_inverted,
                "reachable when inspection panel is open or control cube is rotated",
            ),
        ],
    )
    sensitivity_pot = Potentiometer(
        component_id=f"{p}sensitivity_pot",
        display_name="Sensitivity Potentiometer",
        total_resistance=50_000.0,
        wiper_position=0.5,
        position=Position(0.05, 0.10, 0.20),
        enclosure_id=f"cube_{prefix}",
    )
    sensitivity_pot.affordances = AffordanceSet(
        static={
            Affordance.OBSERVABLE, Affordance.REACHABLE,
            Affordance.MEASURABLE, Affordance.ADJUSTABLE,
        },
    )
    timing_pot = Potentiometer(
        component_id=f"{p}timing_pot",
        display_name="Timing Potentiometer",
        total_resistance=50_000.0,
        wiper_position=0.5,
        position=Position(0.05, 0.10, 0.22),
        enclosure_id=f"cube_{prefix}",
    )
    timing_pot.affordances = AffordanceSet(
        static={
            Affordance.OBSERVABLE, Affordance.REACHABLE,
            Affordance.MEASURABLE, Affordance.ADJUSTABLE,
        },
    )
    sensor_bias = Resistor(
        component_id=f"{p}sensor_bias",
        display_name="Sensor Bias Resistor",
        resistance=10_000.0,
        position=Position(0.05, 0.05, 0.22),
        enclosure_id=f"cube_{prefix}",
    )
    sensor_bias.affordances = AffordanceSet(
        static={Affordance.MEASURABLE},
        conditional=[
            ConditionalAffordance(
                Affordance.OBSERVABLE, _panel_or_inverted,
                "observable when inspection panel is open or control cube is rotated",
            ),
            ConditionalAffordance(
                Affordance.REACHABLE, _panel_or_inverted,
                "reachable when inspection panel is open or control cube is rotated",
            ),
        ],
    )
    cable_in_pos  = Cable(f"{p}cable_in_pos",  "Control Input Cable (+)",  position=Position(0.00, 0.05, 0.23))
    cable_in_neg  = Cable(f"{p}cable_in_neg",  "Control Input Cable (−)",  position=Position(0.00, 0.05, 0.17))
    cable_out_pos = Cable(f"{p}cable_out_pos", "Control Output Cable (+)", position=Position(0.10, 0.05, 0.23))
    cable_out_neg = Cable(f"{p}cable_out_neg", "Control Output Cable (−)", position=Position(0.10, 0.05, 0.17))
    cable_in_pos.port_enclosures  = {"n": cube.component_id}
    cable_in_neg.port_enclosures  = {"n": cube.component_id}
    cable_out_pos.port_enclosures = {"p": cube.component_id}
    cable_out_neg.port_enclosures = {"p": cube.component_id}

    all_comps = [
        module, cube, panel, light_sensor, relay,
        sensitivity_pot, timing_pot, sensor_bias,
        cable_in_pos, cable_in_neg, cable_out_pos, cable_out_neg,
    ]
    ns = SimpleNamespace(
        module=module, cube=cube, panel=panel,
        light_sensor=light_sensor, relay=relay,
        sensitivity_pot=sensitivity_pot, timing_pot=timing_pot,
        sensor_bias=sensor_bias,
        cable_in_pos=cable_in_pos, cable_in_neg=cable_in_neg,
        cable_out_pos=cable_out_pos, cable_out_neg=cable_out_neg,
    )
    ns.ALL = {c.component_id: c for c in all_comps}
    return ns
