"""
Geometry, angle, wind, and aero-cost utilities for maneuver planning.

Coordinate convention:
    x: East
    y: North

Heading/course convention:
    0 rad points North.
    Positive angle rotates clockwise.
    Therefore:
        vx = speed * sin(heading)
        vy = speed * cos(heading)

Wind direction convention:
    theta_wa is wind-from direction.
    The wind velocity vector points toward theta_wa + pi.
"""

from __future__ import annotations

import numpy as np


def norm_angle(angle_rad: float) -> float:
    """Wrap angle to [0, 2*pi)."""
    return float(angle_rad % (2.0 * np.pi))


def wrap_signed_error(target_rad: float, current_rad: float) -> float:
    """Signed angular difference target - current in (-pi, pi]."""
    return float((target_rad - current_rad + np.pi) % (2.0 * np.pi) - np.pi)


def local_delta_xy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    reference_heading_rad: float,
) -> tuple[float, float]:
    """
    Transform world delta from point 1 to point 2 into a local frame aligned
    with reference_heading_rad.

    This preserves your original local-frame convention.
    """
    dx = x2 - x1
    dy = y2 - y1

    theta_l = 2.0 * np.pi - reference_heading_rad

    local_x = np.cos(theta_l) * dx + np.sin(theta_l) * dy
    local_y = -np.sin(theta_l) * dx + np.cos(theta_l) * dy

    return float(local_x), float(local_y)


def find_nearest(array, value) -> int:
    array = np.asarray(array)
    return int(np.abs(array - value).argmin())


def compute_straight_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.hypot(x2 - x1, y2 - y1))


def wind_from_components(wind_speed_mps: float, theta_wa_rad: float) -> tuple[float, float]:
    """
    Convert wind speed and wind-from direction to x/y velocity components.
    """
    wx = wind_speed_mps * np.sin(theta_wa_rad + np.pi)
    wy = wind_speed_mps * np.cos(theta_wa_rad + np.pi)
    return float(wx), float(wy)


def wind_correction_angle(
    wind_speed_mps: float,
    true_airspeed_mps: float,
    course_rad: float,
    theta_wa_rad: float,
) -> float:
    """
    Wind correction angle used by the straight segment model.
    """
    if true_airspeed_mps <= 1e-9:
        return 0.0

    arg = (wind_speed_mps / true_airspeed_mps) * np.sin(course_rad - theta_wa_rad)
    arg = np.clip(arg, -1.0, 1.0)

    return float(np.arcsin(arg))


def ground_velocity_components(
    true_airspeed_mps: float,
    heading_rad: float,
    wind_speed_mps: float,
    theta_wa_rad: float,
) -> tuple[float, float]:
    """
    Ground velocity from aircraft heading, true airspeed, and wind.
    """
    wx, wy = wind_from_components(wind_speed_mps, theta_wa_rad)

    vx = true_airspeed_mps * np.sin(heading_rad) + wx
    vy = true_airspeed_mps * np.cos(heading_rad) + wy

    return float(vx), float(vy)


def yaw_from_course(
    course_rad: float,
    true_airspeed_mps: float,
    wind_speed_mps: float,
    theta_wa_rad: float,
) -> float:
    """
    Convert desired ground-track course to aircraft yaw/heading.
    """
    wca = wind_correction_angle(
        wind_speed_mps=wind_speed_mps,
        true_airspeed_mps=true_airspeed_mps,
        course_rad=course_rad,
        theta_wa_rad=theta_wa_rad,
    )
    return norm_angle(course_rad - wca)


def course_from_yaw(
    yaw_rad: float,
    true_airspeed_mps: float,
    wind_speed_mps: float,
    theta_wa_rad: float,
) -> float:
    """
    Compute ground-track course from yaw/heading.
    """
    vx, vy = ground_velocity_components(
        true_airspeed_mps=true_airspeed_mps,
        heading_rad=yaw_rad,
        wind_speed_mps=wind_speed_mps,
        theta_wa_rad=theta_wa_rad,
    )

    return norm_angle(np.pi / 2.0 - np.arctan2(vy, vx))


def _unpack_aero(aero) -> tuple[float, float, float, float, float, float, float]:
    """
    Accept either:
        - your old Aero numpy array/list
        - uav_opt.config.AeroConfig-like object
    """
    if hasattr(aero, "W"):
        return (
            float(aero.W),
            float(aero.S),
            float(aero.rho),
            float(aero.Cl_0),
            float(aero.Cd_0),
            float(aero.Cl_alpha),
            float(aero.K),
        )

    arr = np.asarray(aero, dtype=float)
    if arr.size < 7:
        raise ValueError("Aero must contain W, S, rho, Cl_0, Cd_0, Cl_alpha, K.")

    return tuple(float(v) for v in arr[:7])


def aero_cost_wh(bank_rad: float, true_airspeed_mps: float, dt_s: float, aero) -> float:
    """
    Returns Wh for one timestep.
    """
    W, S, rho, Cl_0, Cd_0, Cl_alpha, K = _unpack_aero(aero)

    q = 0.5 * rho * true_airspeed_mps**2
    cl_cruise = W * 9.81 / (S * q)

    cos_bank = np.cos(bank_rad)
    cos_bank = np.clip(cos_bank, 1e-6, None)

    cl_mission = cl_cruise / cos_bank
    alpha = (cl_mission - Cl_0) / Cl_alpha
    cd = Cd_0 + K * cl_mission**2

    cos_alpha = np.cos(alpha)
    cos_alpha = np.clip(cos_alpha, 1e-6, None)

    energy_wh = 0.5 * rho * true_airspeed_mps**3 * S * dt_s * cd / cos_alpha / 3600.0

    return float(energy_wh)
