"""
Test that InspectConnections reports which PORT of connected cables is actually connected.

This test ensures that InspectConnections explicitly reports cable port names (p or n)
when describing which cables are connected to a component's ports. This prevents
ambiguity where the NL interface could misinterpret nominal wiring as polarity inversions.

Issue: When cables are daisy-chained (ctrl2.out_neg.n → ctrl3.in_neg.p), the old
InspectConnections would only report "Control Output Cable (−) 2" without specifying
which port (p or n) of that cable is actually involved. This led to misinterpretation.

Fix: Include port names in the report, e.g., "Control Output Cable (−) 2 (port n)"
"""
from diagnosable_systems_simulation.systems.ten_cubes.factory import build_ten_cubes_system
from diagnosable_systems_simulation.actions.diagnostic_actions import InspectConnections


def test_inspect_connections_reports_cable_ports():
    """Verify InspectConnections includes port names of connected cables."""
    sys = build_ten_cubes_system()

    # Inspect ctrl3_cable_in_neg: should show connections to ctrl2_cable_out_neg and ctrl3_cable_out_neg
    ctrl3_cable_in_neg = sys.component("ctrl3_cable_in_neg")
    result = sys.apply_action(InspectConnections(), {"subject": ctrl3_cable_in_neg})

    assert result.success, "InspectConnections should succeed"
    message = result.message

    # Verify port names are included in the output
    assert "(port n)" in message, "Should report port 'n' for ctrl2_cable_out_neg connection"
    assert "(port p)" in message, "Should report port 'p' for ctrl3_cable_out_neg connection"

    # Verify the specific connections mentioned
    assert "Control Output Cable (−) 2 (port n)" in message, \
        "Should explicitly state ctrl2_cable_out_neg's port n is connected to ctrl3_cable_in_neg.p"
    assert "Control Output Cable (−) 3 (port p)" in message, \
        "Should explicitly state ctrl3_cable_out_neg's port p is connected to ctrl3_cable_in_neg.n"


def test_inspect_connections_nominal_wiring_recognized():
    """Verify that nominal positive-to-positive, negative-to-negative wiring is correctly reported."""
    sys = build_ten_cubes_system()

    # Check ctrl2 negative output cable → ctrl3 negative input cable
    ctrl2_neg_out = sys.component("ctrl2_cable_out_neg")
    ctrl3_neg_in = sys.component("ctrl3_cable_in_neg")

    # In nominal wiring: output.n connects to input.p
    assert ctrl2_neg_out.ports[1].node_id == ctrl3_neg_in.ports[0].node_id, \
        "Nominal wiring requires ctrl2_cable_out_neg.n to connect to ctrl3_cable_in_neg.p"

    # Verify InspectConnections correctly reports this
    result = sys.apply_action(InspectConnections(), {"subject": ctrl3_neg_in})
    assert result.success
    message = result.message

    # The message should show the correct port connections
    assert "port 'p': Control Output Cable (−) 2 (port n)" in message, \
        "Should show input port p connects to output port n of previous cable"


if __name__ == "__main__":
    test_inspect_connections_reports_cable_ports()
    test_inspect_connections_nominal_wiring_recognized()
    print("✓ All tests passed")
