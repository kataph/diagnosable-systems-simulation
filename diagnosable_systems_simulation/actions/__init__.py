"""
All diagnostic and fault-injection actions in one place.

Usage::

    from diagnosable_systems_simulation.actions import (
        CloseSwitch, MeasureVoltage, ObserveComponent, OpenSwitch,
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
    CloseSwitch,
    OpenSwitch,
    TestContinuity,
    TestDiode,
    MeasureCurrent,
    OpenInspectionPanel,
    CloseInspectionPanel,
    TestPathContinuity,
    MoveLED,
    TestControlSubchain,
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
    "CloseSwitch", "InvertEnclosure", "MeasureVoltage", "ObserveComponent",
    "OpenPeephole", "OpenSwitch", "ReplaceComponent", "RestoreEnclosure",
    "TestContinuity", "TestDiode",    
    "MeasureCurrent",
    "OpenInspectionPanel",
    "CloseInspectionPanel",
    "TestPathContinuity",
    "MoveLED",
    "TestControlSubchain",

    # Fault injection
    "BlowFuse", "DegradeComponent", "DisconnectCable",
    "ForceSwitch", "ReconnectCable", "ShortCircuit",
]
