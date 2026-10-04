"""Canonical BTC target-label policy shared by production settlement and research.

The neutral band is deliberately identical across offline research and live
settlement so reported OOS performance measures the same target the production
system actually settles.
"""
NEUTRAL_BPS = 2.0
NEUTRAL_RETURN = NEUTRAL_BPS / 10000.0
CLASSES = ("DOWN", "FLAT", "UP")
BINARY_CLASSES = ("DOWN", "UP")
BINARY_TARGET_VERSION = "binary_sign_v1"

def binary_direction_from_return(future_return: float) -> str:
    """Return a two-class direction label with no neutral/FLAT state.

    Target semantics are deliberately versioned separately from the production
    three-class policy. Zero/non-positive return is DOWN; strictly positive
    return is UP. This function is research-only until a separate binary OOS,
    robustness, calibration, and holdout gate approves promotion.
    """
    value = float(future_return)
    return "UP" if value > 0.0 else "DOWN"

def binary_direction_from_prices(base: float, actual: float) -> str:
    base = float(base)
    actual = float(actual)
    if base <= 0:
        raise ValueError("base price must be positive")
    return binary_direction_from_return(actual / base - 1.0)

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
