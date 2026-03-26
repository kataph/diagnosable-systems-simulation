"""
All diagnostic and fault-injection actions in one place.

Usage::

    from diagnosable_systems_simulation.actions import (
        MeasureVoltage, ObserveComponent, ToggleSwitch,
        DisconnectCable, DegradeComponent,
    )
"""
from diagnosable_systems_simulation.actions.base import (
    Action, ActionCost, ActionResult,
)
from diagnosable_systems_simulation.actions.diagnostic_actions import (
    AdjustPotentiometer,
    ClosePeephole,
    InspectConnections,
    InvertEnclosure,
    MeasureVoltage,
    ObserveComponent,
    OpenPeephole,
    ReplaceComponent,
    RestoreEnclosure,
    TestContinuity,
    TestDiode,
    ToggleSwitch,
)
from diagnosable_systems_simulation.actions.fault_actions import (
    BlowFuse,
    DegradeComponent,
    DisconnectCable,
    ForceSwitch,
    ReconnectCable,
    ShortCircuit,
)

__all__ = [
    # Base
    "Action", "ActionCost", "ActionResult",
    # Diagnostic
    "AdjustPotentiometer", "ClosePeephole", "InspectConnections",
    "InvertEnclosure", "MeasureVoltage", "ObserveComponent", "OpenPeephole",
    "ReplaceComponent", "RestoreEnclosure", "TestContinuity", "TestDiode",
    "ToggleSwitch",
    # Fault injection
    "BlowFuse", "DegradeComponent", "DisconnectCable",
    "ForceSwitch", "ReconnectCable", "ShortCircuit",
]
