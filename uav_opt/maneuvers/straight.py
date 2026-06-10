"""
Straight-line segment generation with wind correction.
"""

from __future__ import annotations

import numpy as np

from uav_opt.maneuvers.geometry import (
    norm_angle,
    wind_correction_angle,
    ground_velocity_components,
    aero_cost_wh,
)


def compute_straight(
    x1: float,
    y1: float,
    chi: float,
    V_TAS: float,
    V_w: float,
    theta_wa: float,
    distance: float,
    dt: float,
    aero,
) -> np.ndarray:
    """
    Compute a straight ground-track segment with wind correction.

    The aircraft holds a crabbed yaw such that ground track roughly follows chi.
    """
    x = float(x1)
    y = float(y1)
    t = 0.0

    distance = max(float(distance), 0.0)

    wca = wind_correction_angle(
        wind_speed_mps=V_w,
        true_airspeed_mps=V_TAS,
        course_rad=chi,
        theta_wa_rad=theta_wa,
    )

    psi = norm_angle(chi - wca)

    vx, vy = ground_velocity_components(V_TAS, psi, V_w, theta_wa)

    DYN = [
        [
            0.0,
            x,
            y,
            0.0,
            vx,
            vy,
            0.0,
            psi,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    ]

    remaining = distance

    while remaining > 1e-6:
        vx, vy = ground_velocity_components(V_TAS, psi, V_w, theta_wa)
        ground_speed = max(np.hypot(vx, vy), 1e-6)

        step_distance = ground_speed * dt

        if step_distance >= remaining:
            h = remaining / ground_speed
        else:
            h = dt

        x += vx * h
        y += vy * h
        t += h

        cost = aero_cost_wh(0.0, V_TAS, h, aero)

        DYN.append(
            [
                t,
                x,
                y,
                0.0,
                vx,
                vy,
                0.0,
                psi,
                0.0,
                0.0,
                cost,
                0.0,
            ]
        )

        remaining -= ground_speed * h

    return np.asarray(DYN, dtype=float)
