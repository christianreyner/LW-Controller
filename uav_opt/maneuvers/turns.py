"""
Turn trajectory generation.

This file replaces the turn-related portions of the old Trochoidal.py.

Main functions:
    compute_turn(...)
    compute_turn_simple(...)
"""

from __future__ import annotations

import numpy as np

from uav_opt.maneuvers.geometry import (
    norm_angle,
    wrap_signed_error,
    ground_velocity_components,
    aero_cost_wh,
)


def compute_turn(
    x1: float,
    y1: float,
    psi_1: float,
    psi_2: float,
    g: float,
    bank_angle: float,
    V_TAS: float,
    V_w: float,
    theta_wa: float,
    dt: float,
    aero,
    roll_rate: float,
    heading_tol: float = 1e-2,
    roll_in: float = 0.0,
    roll_out: float = 0.0,
) -> np.ndarray:
    """
    Coordinated turn with finite roll-rate.

    Conventions:
        - psi is heading, clockwise from North.
        - positive bank_angle increases heading.
        - V_TAS is true airspeed.
        - wind is constant.

    Returns DYN with columns:
        0 time
        1 x
        2 y
        3 z
        4 vx
        5 vy
        6 vz
        7 heading
        8 bank
        9 pitch
        10 energy/cost increment
        11 mode/status
    """

    pi = np.pi

    def wrap2pi(a):
        return a % (2.0 * pi)

    V_TAS_safe = max(float(V_TAS), 1e-6)
    rr = max(abs(float(roll_rate)), 1e-6)

    k = g / V_TAS_safe
    k_over_rr = k / rr

    def dpsi_roll_abs(phi_abs: float) -> float:
        """
        Heading gained while rolling from 0 to |phi_abs|.
        """
        phi_abs = max(float(phi_abs), 0.0)
        if phi_abs <= 0.0:
            return 0.0

        c = np.cos(phi_abs)
        c = max(c, 1e-12)

        return float(k_over_rr * (-np.log(c)))

    def inv_budget_to_phi(remaining_heading: float) -> float:
        """
        If there is no steady hold phase, compute peak bank from total heading.
        """
        remaining_heading = max(float(remaining_heading), 0.0)
        if remaining_heading <= 0.0:
            return 0.0

        arg = -remaining_heading / (2.0 * k_over_rr)
        c = np.exp(arg)
        c = np.clip(c, 1e-12, 1.0)

        return float(np.arccos(c))

    x = float(x1)
    y = float(y1)
    psi = float(psi_1)
    psi_target = wrap2pi(psi_2)

    turn_sense = 1.0 if bank_angle >= 0.0 else -1.0
    phi_cmd_abs = abs(float(bank_angle))
    phi_cmd_signed = turn_sense * phi_cmd_abs

    psi_wrapped = wrap2pi(psi)
    if turn_sense > 0.0:
        remaining = (psi_target - psi_wrapped) % (2.0 * pi)
    else:
        remaining = (psi_wrapped - psi_target) % (2.0 * pi)

    vx0, vy0 = ground_velocity_components(V_TAS, psi, V_w, theta_wa)

    DYN = [
        [
            0.0,
            x,
            y,
            0.0,
            vx0,
            vy0,
            0.0,
            wrap2pi(psi),
            roll_in,
            0.0,
            0.0,
            0.0,
        ]
    ]

    if remaining <= heading_tol or phi_cmd_abs <= 1e-9:
        DYN[-1][11] = 3.0
        return np.asarray(DYN, dtype=float)

    roll_in_budget = dpsi_roll_abs(max(phi_cmd_abs - abs(roll_in), 0.0))
    roll_out_budget = dpsi_roll_abs(max(phi_cmd_abs - abs(roll_out), 0.0))
    full_bank_budget = roll_in_budget + roll_out_budget

    if remaining > full_bank_budget:
        allow_steady = True
        phi_peak_abs = phi_cmd_abs
    else:
        allow_steady = False
        phi_peak_abs = min(inv_budget_to_phi(remaining), phi_cmd_abs)

    phi_peak_signed = turn_sense * phi_peak_abs

    t = 0.0
    phi = float(roll_in)

    def append_row(t_row, x_row, y_row, psi_row, phi_row, cost_dt, mode):
        vx, vy = ground_velocity_components(V_TAS, psi_row, V_w, theta_wa)
        DYN.append(
            [
                t_row,
                x_row,
                y_row,
                0.0,
                vx,
                vy,
                0.0,
                wrap2pi(psi_row),
                phi_row,
                0.0,
                aero_cost_wh(phi_row, V_TAS, cost_dt, aero),
                float(mode),
            ]
        )

    def integrate_roll_segment(phi_a, phi_b, mode_code):
        """
        Integrate roll from phi_a to phi_b at limited roll rate.
        """
        nonlocal x, y, psi, t, phi

        dphi = phi_b - phi_a
        if abs(dphi) <= 1e-12:
            phi = phi_b
            return

        duration = abs(dphi) / rr
        n_steps = max(1, int(np.ceil(duration / max(min(dt, 0.05), 1e-3))))
        h = duration / n_steps

        phi_current = phi_a

        for _ in range(n_steps):
            step_sign = np.sign(dphi)
            phi_next = phi_current + step_sign * rr * h

            if step_sign > 0:
                phi_next = min(phi_next, phi_b)
            else:
                phi_next = max(phi_next, phi_b)

            phi_mid = 0.5 * (phi_current + phi_next)

            omega_mid = k * np.tan(phi_mid)
            psi_next = psi + omega_mid * h
            psi_mid = 0.5 * (psi + psi_next)

            vx_mid, vy_mid = ground_velocity_components(V_TAS, psi_mid, V_w, theta_wa)
            x += vx_mid * h
            y += vy_mid * h

            psi = psi_next
            t += h
            phi_current = phi_next
            phi = phi_next

            append_row(t, x, y, psi, phi, h, mode_code)

    # 1. Roll in
    integrate_roll_segment(phi, phi_peak_signed, mode_code=0)

    # 2. Steady turn
    if allow_steady and abs(phi) > 1e-9:
        psi_wrapped_after = wrap2pi(psi)

        if turn_sense > 0.0:
            remaining_now = (psi_target - psi_wrapped_after) % (2.0 * pi)
        else:
            remaining_now = (psi_wrapped_after - psi_target) % (2.0 * pi)

        steady_needed = max(0.0, remaining_now - roll_out_budget)
        omega = k * np.tan(phi)

        if abs(omega) > 1e-12 and steady_needed > 0.0:
            steady_time = steady_needed / abs(omega)
        else:
            steady_time = 0.0

        if steady_time > 0.0:
            n_steps = max(1, int(np.ceil(steady_time / max(dt, 1e-3))))

            for step in range(n_steps):
                h = dt if step < n_steps - 1 else steady_time - dt * (n_steps - 1)
                if h <= 1e-12:
                    continue

                psi_prev = psi
                psi += omega * h
                psi_mid = 0.5 * (psi_prev + psi)

                vx_mid, vy_mid = ground_velocity_components(V_TAS, psi_mid, V_w, theta_wa)
                x += vx_mid * h
                y += vy_mid * h
                t += h

                append_row(t, x, y, psi, phi, h, mode=1)

    # 3. Roll out
    integrate_roll_segment(phi, roll_out, mode_code=2)

    # 4. Mark final status
    final_err = wrap_signed_error(psi_target, wrap2pi(psi))
    if abs(final_err) <= heading_tol:
        DYN[-1][11] = 3.0

    return np.asarray(DYN, dtype=float)


def compute_turn_simple(
    x1: float,
    y1: float,
    psi_1: float,
    psi_2: float,
    g: float,
    bank_angle: float,
    V_TAS: float,
    V_w: float,
    theta_wa: float,
    dt: float,
    aero=None,
    heading_tol: float = 1e-2,
) -> np.ndarray:
    """
    Constant-bank turn without roll dynamics.

    Kept mostly for debugging and quick initial guesses.
    """
    pi = np.pi

    def wrap2pi(a):
        return a % (2.0 * pi)

    x0 = float(x1)
    y0 = float(y1)
    psi0 = float(psi_1)
    psi_target = wrap2pi(psi_2)

    phi = float(bank_angle)
    V_TAS_safe = max(float(V_TAS), 1e-9)

    if phi >= 0.0:
        remaining = (psi_target - wrap2pi(psi0)) % (2.0 * pi)
    else:
        remaining = (wrap2pi(psi0) - psi_target) % (2.0 * pi)

    vx0, vy0 = ground_velocity_components(V_TAS, psi0, V_w, theta_wa)

    if remaining <= heading_tol:
        return np.asarray(
            [[0.0, x0, y0, 0.0, vx0, vy0, 0.0, wrap2pi(psi0), phi, 0.0, 0.0, 3.0]],
            dtype=float,
        )

    omega = g * np.tan(phi) / V_TAS_safe
    if abs(omega) < 1e-12 or dt <= 0.0:
        return np.asarray(
            [[0.0, x0, y0, 0.0, vx0, vy0, 0.0, wrap2pi(psi0), phi, 0.0, 0.0, 2.0]],
            dtype=float,
        )

    t_hit = remaining / abs(omega)

    n_full = int(np.floor(t_hit / dt))
    rem = t_hit - n_full * dt

    rows = []
    rows.append([0.0, x0, y0, 0.0, vx0, vy0, 0.0, wrap2pi(psi0), phi, 0.0, 0.0, 0.0])

    t = 0.0
    x = x0
    y = y0
    psi = psi0

    total_steps = n_full + (1 if rem > 1e-12 else 0)

    for step in range(total_steps):
        h = dt if step < n_full else rem
        if h <= 1e-12:
            continue

        psi_prev = psi
        psi += omega * h
        psi_mid = 0.5 * (psi_prev + psi)

        vx, vy = ground_velocity_components(V_TAS, psi_mid, V_w, theta_wa)
        x += vx * h
        y += vy * h
        t += h

        cost = 0.0
        if aero is not None:
            cost = aero_cost_wh(phi, V_TAS, h, aero)

        rows.append([t, x, y, 0.0, vx, vy, 0.0, wrap2pi(psi), phi, 0.0, cost, 0.0])

    rows[-1][11] = 3.0

    return np.asarray(rows, dtype=float)
