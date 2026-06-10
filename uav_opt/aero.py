import math
import numpy as np
from uav_opt.config import AeroConfig


def lift_coefficient_for_bank(bank_rad: float, cl_cruise: float) -> float:
    return cl_cruise / np.cos(bank_rad)


def alpha_from_cl(cl: float, aero: AeroConfig) -> float:
    return (cl - aero.Cl_0) / aero.Cl_alpha


def drag_coefficient(cl: float, aero: AeroConfig) -> float:
    return aero.Cd_0 + aero.K * cl**2


def max_bank_from_stall_margin(airspeed_mps: float, aero: AeroConfig) -> float:
    """
    Estimate maximum bank before stall margin is exhausted.
    """
    cl_max = 1.25
    stall_margin = 1.2

    q = 0.5 * aero.rho * airspeed_mps**2
    cl_cruise = aero.W * 9.81 / (aero.S * q)

    n_reserve = (cl_max / cl_cruise) / stall_margin**2
    n_reserve = max(n_reserve, 1.0001)

    return math.acos(1.0 / n_reserve)


def best_range_speed(aero: AeroConfig) -> float:
    return np.sqrt(
        2.0 * aero.W * 9.81 * np.sqrt(aero.K / aero.Cd_0) / (aero.rho * aero.S)
    )


def aero_energy_cost_wh(bank_rad: float, airspeed_mps: float, timestep_s: float, aero: AeroConfig) -> float:
    """
    Approximate aerodynamic energy cost in Wh over timestep_s.

    This preserves your original calculation style.
    """
    q = 0.5 * aero.rho * airspeed_mps**2
    cl_cruise = aero.W * 9.81 / (aero.S * q)

    cl_mission = lift_coefficient_for_bank(bank_rad, cl_cruise)
    alpha = alpha_from_cl(cl_mission, aero)
    cd = drag_coefficient(cl_mission, aero)

    power_w = 0.5 * aero.rho * airspeed_mps**3 * aero.S * cd / np.cos(alpha)
    energy_wh = power_w * timestep_s / 3600.0

    return float(energy_wh)
