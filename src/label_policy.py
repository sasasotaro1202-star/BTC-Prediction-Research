"""Canonical BTC target-label policy shared by production settlement and research.

The neutral band is deliberately identical across offline research and live
settlement so reported OOS performance measures the same target the production
system actually settles.
"""
NEUTRAL_BPS = 2.0
NEUTRAL_RETURN = NEUTRAL_BPS / 10000.0
CLASSES = ("DOWN", "FLAT", "UP")

def direction_from_return(future_return: float) -> str:
    value = float(future_return)
    if value > NEUTRAL_RETURN:
        return "UP"
    if value < -NEUTRAL_RETURN:
        return "DOWN"
    return "FLAT"

def direction_from_prices(base: float, actual: float) -> str:
    base = float(base)
    actual = float(actual)
    if base <= 0:
        raise ValueError("base price must be positive")
    return direction_from_return(actual / base - 1.0)
