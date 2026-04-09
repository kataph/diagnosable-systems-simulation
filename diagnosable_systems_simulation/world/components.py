from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from diagnosable_systems_simulation.world.affordances import Affordance, AffordanceSet, ConditionalAffordance
from diagnosable_systems_simulation.world.ports import BoundPort, ElectricalPort, PortRole
from diagnosable_systems_simulation.world.spatial import Position


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Component:
    """
    Abstract base for every physical component in a system.

    Subclasses carry their own electrical parameters and define their
    default port layout and static affordances.

    Fault overlay
    -------------
    ``_fault_overlay`` holds parameter overrides injected by fault actions.
    ``current_parameters()`` merges the overlay on top of the nominal values,
    so the simulation backend always sees the current (possibly degraded) state.
    """

    def __init__(
        self,
        component_id: str,
        display_name: str,
        ports: list[ElectricalPort],
        affordances: AffordanceSet,
        position: Optional[Position] = None,
        enclosure_id: Optional[str] = None,
    ):
        self.component_id = component_id
        self.display_name = display_name
        self.ports = ports
        self.affordances = affordances
        self.position = position
        self.enclosure_id = enclosure_id
        self._fault_overlay: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Port helpers
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> BoundPort:
        # Called only when normal attribute lookup fails.
        # If the name matches a port, return a BoundPort for use in wiring.
        for p in self.__dict__.get("ports", []):
            if p.name == name:
                return BoundPort(self, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def port(self, name: str) -> ElectricalPort:
        for p in self.ports:
            if p.name == name:
                return p
        raise KeyError(f"Component {self.component_id!r} has no port {name!r}")

    def port_node(self, name: str) -> str:
        p = self.port(name)
        if p.node_id is None:
            raise RuntimeError(
                f"Port {name!r} of {self.component_id!r} is not connected to a node."
            )
        return p.node_id

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def nominal_parameters(self) -> dict[str, Any]:
        """Return the unmodified electrical parameters for the backend."""
        raise NotImplementedError

    def current_parameters(self) -> dict[str, Any]:
        """Return parameters with fault overlay merged on top."""
        params = dict(self.nominal_parameters())
        params.update(self._fault_overlay)
        return params

    def apply_fault(self, overlay: dict[str, Any]) -> None:
        self._fault_overlay.update(overlay)

    def clear_fault(self) -> None:
        self._fault_overlay.clear()

    def has_fault(self) -> bool:
        return bool(self._fault_overlay)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.component_id!r})"


# ---------------------------------------------------------------------------
# Concrete component types
# ---------------------------------------------------------------------------

class Resistor(Component):
    def __init__(
        self,
        component_id: str,
        display_name: str,
        resistance: float,
        position: Optional[Position] = None,
        enclosure_id: Optional[str] = None,
    ):
        self.resistance = resistance
        super().__init__(
            component_id=component_id,
            display_name=display_name,
            ports=[
                ElectricalPort("p", PortRole.POSITIVE),
                ElectricalPort("n", PortRole.NEGATIVE),
            ],
            affordances=AffordanceSet(
                static={Affordance.MEASURABLE, Affordance.REPLACEABLE}
            ),
            position=position,
            enclosure_id=enclosure_id,
        )

    def nominal_parameters(self) -> dict[str, Any]:
        return {"resistance": self.resistance}


class LED(Component):
    def __init__(
        self,
        component_id: str,
        display_name: str,
        forward_voltage: float = 2.0,
        forward_current: float = 0.02,
        color: str = "red",
        position: Optional[Position] = None,
        enclosure_id: Optional[str] = None,
    ):
        self.forward_voltage = forward_voltage
        self.forward_current = forward_current
        self.color = color
        super().__init__(
            component_id=component_id,
            display_name=display_name,
            ports=[
                ElectricalPort("anode", PortRole.ANODE),
                ElectricalPort("cathode", PortRole.CATHODE),
            ],
            affordances=AffordanceSet(
                static={Affordance.MEASURABLE, Affordance.REPLACEABLE}
            ),
            position=position,
            enclosure_id=enclosure_id,
        )

    def nominal_parameters(self) -> dict[str, Any]:
        return {
            "forward_voltage": self.forward_voltage,
            "forward_current": self.forward_current,
        }


class Diode(Component):
    def __init__(
        self,
        component_id: str,
        display_name: str,
        forward_voltage: float = 0.7,
        position: Optional[Position] = None,
        enclosure_id: Optional[str] = None,
    ):
        self.forward_voltage = forward_voltage
        super().__init__(
            component_id=component_id,
            display_name=display_name,
            ports=[
                ElectricalPort("anode", PortRole.ANODE),
                ElectricalPort("cathode", PortRole.CATHODE),
            ],
            affordances=AffordanceSet(
                static={Affordance.MEASURABLE, Affordance.REPLACEABLE}
            ),
            position=position,
            enclosure_id=enclosure_id,
        )

    def nominal_parameters(self) -> dict[str, Any]:
        return {"forward_voltage": self.forward_voltage}


class Bulb(Component):
    """
    A light bulb. Emits light when sufficient power is dissipated.
    The simulation runner populates `SimulationResult.emitting_light`
    based on `power_threshold`.
    """

    def __init__(
        self,
        component_id: str,
        display_name: str,
        resistance: float = 120.0,
        power_threshold: float = 0.1,
        position: Optional[Position] = None,
        enclosure_id: Optional[str] = None,
    ):
        self.resistance = resistance
        self.power_threshold = power_threshold
        super().__init__(
            component_id=component_id,
            display_name=display_name,
            ports=[
                ElectricalPort("p", PortRole.POSITIVE),
                ElectricalPort("n", PortRole.NEGATIVE),
            ],
            affordances=AffordanceSet(
                static={Affordance.MEASURABLE, Affordance.OBSERVABLE, Affordance.REPLACEABLE}
            ),
            position=position,
            enclosure_id=enclosure_id,
        )

    def nominal_parameters(self) -> dict[str, Any]:
        return {
            "resistance": self.resistance,
            "power_threshold": self.power_threshold,
        }


class Switch(Component):
    def __init__(
        self,
        component_id: str,
        display_name: str,
        is_closed: bool = True,
        ron: float = 1e-2,
        roff: float = 1e9,
        position: Optional[Position] = None,
        enclosure_id: Optional[str] = None,
    ):
        self.is_closed = is_closed
        self._nominal_is_closed = is_closed  # saved for nominal_parameters()
        self.ron = ron
        self.roff = roff
        super().__init__(
            component_id=component_id,
            display_name=display_name,
            ports=[
                ElectricalPort("p", PortRole.POSITIVE),
                ElectricalPort("n", PortRole.NEGATIVE),
            ],
            affordances=AffordanceSet(
                static={Affordance.TOGGLABLE, Affordance.OBSERVABLE, Affordance.MEASURABLE}
            ),
            position=position,
            enclosure_id=enclosure_id,
        )

    def nominal_parameters(self) -> dict[str, Any]:
        nom_r = self.ron if self._nominal_is_closed else self.roff
        return {"is_closed": self._nominal_is_closed, "resistance": nom_r, "ron": self.ron, "roff": self.roff}

    def current_parameters(self) -> dict[str, Any]:
        params = super().current_parameters()
        # fault overlay can force is_closed (visible state change, e.g. stuck switch);
        # otherwise is_closed follows live user-controlled state.
        if "is_closed" not in self._fault_overlay:
            params["is_closed"] = self.is_closed
        # resistance: if the fault overlay sets it directly (e.g. internal open circuit
        # modelled as roff regardless of mechanical position), use that value.
        # Otherwise, derive from the logical is_closed state as normal.
        if "resistance" not in self._fault_overlay:
            params["resistance"] = self.ron if params["is_closed"] else self.roff
        params["ron"] = self.ron
        params["roff"] = self.roff
        return params


class Cable(Component):
    """
    A two-terminal connecting cable.
    Detachable by default; once detached it becomes reconnectable.
    Polarity can be flipped (detach, swap, reconnect).
    """

    def __init__(
        self,
        component_id: str,
        display_name: str,
        resistance: float = 0.1,
        position: Optional[Position] = None,
        enclosure_id: Optional[str] = None,
    ):
        self.resistance = resistance
        # Maps port name → enclosure component_id for ports routed inside an enclosure.
        # Only ports listed here are considered "enclosed"; unlisted ports are external.
        self.port_enclosures: dict[str, str] = {}
        super().__init__(
            component_id=component_id,
            display_name=display_name,
            ports=[
                ElectricalPort("p", PortRole.POSITIVE),
                ElectricalPort("n", PortRole.NEGATIVE),
            ],
            affordances=AffordanceSet(
                static={
                    Affordance.OBSERVABLE,
                    Affordance.REACHABLE,
                    Affordance.DETACHABLE,
                    Affordance.MEASURABLE,
                }
            ),
            position=position,
            enclosure_id=enclosure_id,
        )

    def nominal_parameters(self) -> dict[str, Any]:
        return {"resistance": self.resistance}


class VoltageSource(Component):
    def __init__(
        self,
        component_id: str,
        display_name: str,
        voltage: float = 12.0,
        position: Optional[Position] = None,
        enclosure_id: Optional[str] = None,
    ):
        self.voltage = voltage
        super().__init__(
            component_id=component_id,
            display_name=display_name,
            ports=[
                ElectricalPort("pos", PortRole.POSITIVE),
                ElectricalPort("neg", PortRole.NEGATIVE),
            ],
            affordances=AffordanceSet(
                static={Affordance.MEASURABLE, Affordance.OBSERVABLE}
            ),
            position=position,
            enclosure_id=enclosure_id,
        )

    def nominal_parameters(self) -> dict[str, Any]:
        return {"voltage": self.voltage}


class Potentiometer(Component):
    def __init__(
        self,
        component_id: str,
        display_name: str,
        total_resistance: float,
        wiper_position: float = 0.5,
        position: Optional[Position] = None,
        enclosure_id: Optional[str] = None,
    ):
        if not 0.0 <= wiper_position <= 1.0:
            raise ValueError("wiper_position must be in [0, 1]")
        self.total_resistance = total_resistance
        self.wiper_position = wiper_position
        super().__init__(
            component_id=component_id,
            display_name=display_name,
            ports=[
                ElectricalPort("p",     PortRole.POSITIVE),
                ElectricalPort("wiper", PortRole.COMMON),
                ElectricalPort("n",     PortRole.NEGATIVE),
            ],
            affordances=AffordanceSet(
                static={Affordance.ADJUSTABLE, Affordance.MEASURABLE, Affordance.OBSERVABLE}
            ),
            position=position,
            enclosure_id=enclosure_id,
        )

    def nominal_parameters(self) -> dict[str, Any]:
        return {
            "total_resistance": self.total_resistance,
            "wiper_position": self.wiper_position,
        }

    def current_parameters(self) -> dict[str, Any]:
        params = super().current_parameters()
        if "wiper_position" not in self._fault_overlay:
            params["wiper_position"] = self.wiper_position
        return params

    @property
    def resistance_upper(self) -> float:
        return self.total_resistance * self.wiper_position

    @property
    def resistance_lower(self) -> float:
        return self.total_resistance * (1.0 - self.wiper_position)


class LightSensor(Component):
    """
    A photoresistor / light-dependent resistor.
    `resistance_dark` and `resistance_lit` bracket the operating range.
    The solver sets the effective resistance based on whether an
    emitting light source is within coupling range.
    """

    def __init__(
        self,
        component_id: str,
        display_name: str,
        resistance_dark: float = 10_000.0,
        resistance_lit: float = 100.0,
        position: Optional[Position] = None,
        enclosure_id: Optional[str] = None,
    ):
        self.resistance_dark = resistance_dark
        self.resistance_lit = resistance_lit
        # Live state updated by the coupling loop
        self._current_resistance: float = resistance_dark
        super().__init__(
            component_id=component_id,
            display_name=display_name,
            ports=[
                ElectricalPort("p", PortRole.POSITIVE),
                ElectricalPort("n", PortRole.NEGATIVE),
            ],
            affordances=AffordanceSet(
                static={Affordance.MEASURABLE, Affordance.REPLACEABLE}
            ),
            position=position,
            enclosure_id=enclosure_id,
        )

    def set_illuminated(self, lit: bool) -> None:
        self._current_resistance = self.resistance_lit if lit else self.resistance_dark

    def nominal_parameters(self) -> dict[str, Any]:
        return {
            "resistance_dark": self.resistance_dark,
            "resistance_lit": self.resistance_lit,
        }

    def current_parameters(self) -> dict[str, Any]:
        params = super().current_parameters()
        params["resistance"] = self._current_resistance
        return params


class Module(Component):
    """Functional aggregate of components"""
    def __init__(
        self,
        component_id: str,
        display_name: str,
        subcomponents_ids: list[str] | None = None,
        position: Optional[Position] = None,
    ):
        self.is_inverted: bool = False
        self.contained_component_ids: list[str] = list(subcomponents_ids or [])
        super().__init__(
            component_id=component_id,
            display_name=display_name,
            ports=[],
            affordances=AffordanceSet(
                static={Affordance.REACHABLE, Affordance.MEASURABLE}
            ),
            position=position,
            enclosure_id=None,
        )

    def nominal_parameters(self) -> dict:
        return {}
    
    
class PhysicalEnclosure(Component):
    """
    A physical enclosure (e.g. a wooden cube) that contains other components.

    Has no electrical ports — it exists purely as an affordance carrier so
    that actions like InvertEnclosure can be targeted at it via the normal
    component registry.

    State
    -----
    is_inverted
        True while the enclosure is upside-down (bottom face open), making
        internal components visible.

    Affordances
    -----------
    MOVABLE    — can be lifted and inverted (exposes bottom face).
    OBSERVABLE — always visible from the outside.
    """

    def __init__(
        self,
        component_id: str,
        display_name: str,
        contained_component_ids: list[str] | None = None,
        position: Optional[Position] = None,
    ):
        self.is_inverted: bool = False
        self.contained_component_ids: list[str] = list(contained_component_ids or [])
        super().__init__(
            component_id=component_id,
            display_name=display_name,
            ports=[],
            affordances=AffordanceSet(
                static={Affordance.REACHABLE, Affordance.MOVABLE}
            ),
            position=position,
            enclosure_id=None,
        )

    def nominal_parameters(self) -> dict:
        return {}


class Peephole(Component):
    """
    A peephole or access port on an enclosure face.

    Has no electrical ports.  Tracks its own open/closed state; the
    OPENABLE and CLOSEABLE affordances follow automatically.

    State
    -----
    is_open
        True while the peephole cover is removed, making specific internal
        components observable.

    Affordances
    -----------
    OBSERVABLE  — the physical hole is always visible from outside.
    OPENABLE    — present when is_open is False.
    CLOSEABLE   — present when is_open is True.
    """

    def __init__(
        self,
        component_id: str,
        display_name: str,
        position: Optional[Position] = None,
        enclosure_id: Optional[str] = None,
    ):
        self.is_open: bool = False
        super().__init__(
            component_id=component_id,
            display_name=display_name,
            ports=[],
            affordances=AffordanceSet(
                static={Affordance.REACHABLE},
                conditional=[
                    ConditionalAffordance(
                        Affordance.OPENABLE,
                        lambda comp, _ctx: not comp.is_open,
                        "openable when closed",
                    ),
                    ConditionalAffordance(
                        Affordance.CLOSEABLE,
                        lambda comp, _ctx: comp.is_open,
                        "closeable when open",
                    ),
                ],
            ),
            position=position,
            enclosure_id=enclosure_id,
        )

    def nominal_parameters(self) -> dict:
        return {}


class Fuse(Component):
    def __init__(
        self,
        component_id: str,
        display_name: str,
        rating_amps: float,
        is_blown: bool = False,
        position: Optional[Position] = None,
        enclosure_id: Optional[str] = None,
    ):
        self.rating_amps = rating_amps
        self.is_blown = is_blown
        super().__init__(
            component_id=component_id,
            display_name=display_name,
            ports=[
                ElectricalPort("p", PortRole.POSITIVE),
                ElectricalPort("n", PortRole.NEGATIVE),
            ],
            affordances=AffordanceSet(
                static={Affordance.MEASURABLE, Affordance.REPLACEABLE}
            ),
            position=position,
            enclosure_id=enclosure_id,
        )

    def nominal_parameters(self) -> dict[str, Any]:
        return {"rating_amps": self.rating_amps, "is_blown": self.is_blown}

    def current_parameters(self) -> dict[str, Any]:
        params = super().current_parameters()
        if "is_blown" not in self._fault_overlay:
            params["is_blown"] = self.is_blown
        return params
