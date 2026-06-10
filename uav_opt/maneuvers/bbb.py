"""
BBB maneuver planner.

BBB means:
    Bend - Bend - Bend - Bend

Currently this is kept mostly as a fallback/special maneuver. Your selector
does not use it by default, matching your recent Trochoidal.py behavior.
"""

from __future__ import annotations

import numpy as np

from uav_opt.maneuvers.geometry import local_delta_xy
from uav_opt.maneuvers.turns import compute_turn
from uav_opt.maneuvers.sbb_bbs import bisection_method_y


def compute_final_position_BBB(
    x_turn,
    y_turn,
    initial_heading,
    psi_2,
    del_phi,
    reference_heading,
    g,
    bank_angle,
    V_TAS,
    V_w,
    theta_wa,
    dt,
    aero,
    x2,
    y2,
    roll_rate,
    roll_in=0.0,
):
    DYN_a = compute_turn(
        x_turn,
        y_turn,
        initial_heading,
        initial_heading - del_phi,
        g,
        -bank_angle,
        V_TAS,
        V_w,
        theta_wa,
        dt,
        aero,
        roll_rate=roll_rate,
    )

    xa, ya, ha = DYN_a[-1, 1], DYN_a[-1, 2], DYN_a[-1, 7]

    DYN_b = compute_turn(
        xa,
        ya,
        ha,
        psi_2,
        g,
        bank_angle,
        V_TAS,
        V_w,
        theta_wa,
        dt,
        aero,
        roll_rate=roll_rate,
    )

    xb, yb, hb = DYN_b[-1, 1], DYN_b[-1, 2], DYN_b[-1, 7]

    _, delta_y = local_delta_xy(xb, yb, x2, y2, reference_heading)

    if delta_y > 0.0:
        del_phi_b, _, _ = bisection_method_y(
            xb,
            yb,
            hb,
            psi_2,
            reference_heading,
            g,
            bank_angle,
            V_TAS,
            V_w,
            theta_wa,
            dt,
            aero,
            x2,
            y2,
            roll_rate=roll_rate,
        )
    else:
        del_phi_b = 0.0

    if bank_angle < 0.0:
        del_phi_b *= -1.0

    DYN_c = compute_turn(
        xb,
        yb,
        hb,
        hb + del_phi_b,
        g,
        bank_angle,
        V_TAS,
        V_w,
        theta_wa,
        dt,
        aero,
        roll_rate=roll_rate,
    )

    xc, yc, hc = DYN_c[-1, 1], DYN_c[-1, 2], DYN_c[-1, 7]

    DYN_d = compute_turn(
        xc,
        yc,
        hc,
        psi_2,
        g,
        -bank_angle,
        V_TAS,
        V_w,
        theta_wa,
        dt,
        aero,
        roll_rate=roll_rate,
    )

    return float(DYN_d[-1, 1]), float(DYN_d[-1, 2]), float(del_phi_b)


def bisection_method_BBB(
    del_phi_0,
    x_turn,
    y_turn,
    initial_heading,
    psi_2,
    reference_course,
    g,
    bank_angle,
    V_TAS,
    V_w,
    theta_wa,
    dt,
    aero,
    x2,
    y2,
    roll_rate,
    tol=0.1,
    max_iter=100,
    change_threshold=1e-5,
    roll_in=0.0,
):
    low = 0.0
    high = del_phi_0
    prev_mid = None

    dpb_m = 0.0

    for _ in range(max_iter):
        mid = 0.5 * (low + high)

        if bank_angle > 0.0:
            xm, ym, dpb_m = compute_final_position_BBB(
                x_turn,
                y_turn,
                initial_heading,
                psi_2,
                mid,
                reference_course,
                g,
                bank_angle,
                V_TAS,
                V_w,
                theta_wa,
                dt,
                aero,
                x2,
                y2,
                roll_rate=roll_rate,
                roll_in=roll_in,
            )

            xl, yl, dpb_l = compute_final_position_BBB(
                x_turn,
                y_turn,
                initial_heading,
                psi_2,
                low,
                reference_course,
                g,
                bank_angle,
                V_TAS,
                V_w,
                theta_wa,
                dt,
                aero,
                x2,
                y2,
                roll_rate=roll_rate,
                roll_in=roll_in,
            )
        else:
            xm, ym, dpb_m = compute_final_position_BBB(
                x_turn,
                y_turn,
                initial_heading,
                psi_2,
                -mid,
                reference_course,
                g,
                bank_angle,
                V_TAS,
                V_w,
                theta_wa,
                dt,
                aero,
                x2,
                y2,
                roll_rate=roll_rate,
                roll_in=roll_in,
            )

            xl, yl, dpb_l = compute_final_position_BBB(
                x_turn,
                y_turn,
                initial_heading,
                psi_2,
                -low,
                reference_course,
                g,
                bank_angle,
                V_TAS,
                V_w,
                theta_wa,
                dt,
                aero,
                x2,
                y2,
                roll_rate=roll_rate,
                roll_in=roll_in,
            )

        lxm, _ = local_delta_xy(xm, ym, x2, y2, reference_course)
        lxl, _ = local_delta_xy(xl, yl, x2, y2, reference_course)

        if abs(lxm) < tol:
            return mid, dpb_m

        if prev_mid is not None and abs(mid - prev_mid) < change_threshold:
            return mid, dpb_m

        if lxm * lxl < 0.0:
            high = mid
        else:
            low = mid

        prev_mid = mid

    return 0.5 * (low + high), dpb_m


def BBB_maneuver(del_phi_0, waypoints, constants, aero, roll_rate):
    """
    Plan BBB maneuver.

    Returns:
        DYN, success
    """
    x1, y1, x2, y2, psi_1, psi_2 = waypoints
    g, V_TAS, bank_angle, V_w, theta_wa, dt = constants

    turn_right = True

    goal_lateral, _ = local_delta_xy(x1, y1, x2, y2, psi_1)
    if goal_lateral < 0.0:
        bank_angle = -bank_angle
        turn_right = False

    del_phi_a, del_phi_b = bisection_method_BBB(
        del_phi_0,
        x1,
        y1,
        psi_1,
        psi_2,
        psi_2,
        g,
        bank_angle,
        V_TAS,
        V_w,
        theta_wa,
        dt,
        aero,
        x2,
        y2,
        roll_rate=roll_rate,
    )

    bad = (
        np.isclose(abs(del_phi_a), np.pi)
        or np.isclose(abs(del_phi_b), np.pi)
        or np.isclose(abs(del_phi_a), 0.0)
        or np.isclose(abs(del_phi_b), 0.0)
    )

    if bad:
        return np.zeros((1, 12)), False

    if not turn_right:
        del_phi_a *= -1.0

    DYN_a = compute_turn(
        x1,
        y1,
        psi_1,
        psi_1 - del_phi_a,
        g,
        -bank_angle,
        V_TAS,
        V_w,
        theta_wa,
        dt,
        aero,
        roll_rate=roll_rate,
    )

    xa, ya, ha = DYN_a[-1, 1], DYN_a[-1, 2], DYN_a[-1, 7]

    DYN_b = compute_turn(
        xa,
        ya,
        ha,
        psi_2,
        g,
        bank_angle,
        V_TAS,
        V_w,
        theta_wa,
        dt,
        aero,
        roll_rate=roll_rate,
        roll_out=bank_angle,
    )

    xb, yb, hb = DYN_b[-1, 1], DYN_b[-1, 2], DYN_b[-1, 7]

    DYN_c = compute_turn(
        xb,
        yb,
        hb,
        hb + del_phi_b,
        g,
        bank_angle,
        V_TAS,
        V_w,
        theta_wa,
        dt,
        aero,
        roll_rate=roll_rate,
        roll_in=bank_angle,
    )

    xc, yc, hc = DYN_c[-1, 1], DYN_c[-1, 2], DYN_c[-1, 7]

    DYN_d = compute_turn(
        xc,
        yc,
        hc,
        psi_2,
        g,
        -bank_angle,
        V_TAS,
        V_w,
        theta_wa,
        dt,
        aero,
        roll_rate=roll_rate,
    )

    DYN = np.vstack((DYN_a, DYN_b, DYN_c, DYN_d))

    return DYN, True
