"""
SI unit helpers.

Provides simple multiplier constants and a human-readable formatter
so component parameters can be specified and displayed clearly.

Usage::

    from diagnosable_systems.utils.units import k, M, m, u
    r = Resistor(..., resistance=4.7 * k)   # 4700 Ω
"""

# Multipliers
T  = 1e12
G  = 1e9
M  = 1e6
k  = 1e3
m  = 1e-3
u  = 1e-6   # micro
n  = 1e-9
p  = 1e-12


def format_value(value: float, unit: str = "") -> str:
    """
    Format a float with an appropriate SI prefix.

    >>> format_value(4700, "Ω")
    '4.700 kΩ'
    >>> format_value(0.015, "A")
    '15.000 mA'
    """
    prefixes = [
        (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"),
        (1.0, ""),
        (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p"),
    ]
    if value == 0.0:
        return f"0 {unit}".strip()
    abs_val = abs(value)
    for factor, prefix in prefixes:
        if abs_val >= factor:
            return f"{value / factor:.3f} {prefix}{unit}".strip()
    return f"{value:.3e} {unit}".strip()
