"""
SBB/BBS maneuver planners.

SBB:
    Straight - Bend - Bend

BBS:
    Bend - Bend - Straight

This file also contains bisection helpers used by these planners.
"""

from __future__ import annotations

import numpy as np

from uav_opt.maneuvers.geometry import (
    local_delta_xy,
    course_from_yaw,
    yaw_from_course,
)
from uav_opt.maneuvers.turns import compute_turn
from uav_opt.maneuvers.straight import compute_straight


def compute_final_position(
    x_turn,
    y_turn,
    initial_heading,
    psi_2,
    del_phi,
    g,
    bank_angle,
    V_TAS,
    V_w,
    theta_wa,
    dt,
    aero,
    roll_rate,
    roll_in=0.0,
):
    """
    Simulate two connected bends and return final x/y.
    """
    if bank_angle < 0:
        DYN_a = compute_turn(
            x_turn,
            y_turn,
            initial_heading,
            initial_heading - del_phi,
            g,
            bank_angle,
            V_TAS,
            V_w,
            theta_wa,
            dt,
            aero,
            roll_rate=roll_rate,
            roll_in=roll_in,
        )

        xa, ya, ha = DYN_a[-1, 1], DYN_a[-1, 2], DYN_a[-1, 7]

        DYN_b = compute_turn(
            xa,
            ya,
            ha,
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
    else:
        DYN_a = compute_turn(
            x_turn,
            y_turn,
            initial_heading,
            initial_heading + del_phi,
            g,
            bank_angle,
            V_TAS,
            V_w,
            theta_wa,
            dt,
            aero,
            roll_rate=roll_rate,
            roll_in=roll_in,
        )

        xa, ya, ha = DYN_a[-1, 1], DYN_a[-1, 2], DYN_a[-1, 7]

        DYN_b = compute_turn(
            xa,
            ya,
            ha,
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

    return float(DYN_b[-1, 1]), float(DYN_b[-1, 2])


def bisection_method_x(
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
    tol=0.01,
    max_iter=100,
    change_threshold=1e-7,
    roll_in=0.0,
):
    """
    Find del_phi so final x-error in reference frame is approximately zero.
    """
    low = 0.0
    high = np.pi
    prev_mid = None

    xm = ym = None

    for _ in range(max_iter):
        mid = 0.5 * (low + high)

        xm, ym = compute_final_position(
            x_turn,
            y_turn,
            initial_heading,
            psi_2,
            mid,
            g,
            bank_angle,
            V_TAS,
            V_w,
            theta_wa,
            dt,
            aero,
            roll_rate=roll_rate,
            roll_in=roll_in,
        )

        xl, yl = compute_final_position(
            x_turn,
            y_turn,
            initial_heading,
            psi_2,
            low,
            g,
            bank_angle,
            V_TAS,
            V_w,
            theta_wa,
            dt,
            aero,
            roll_rate=roll_rate,
            roll_in=roll_in,
        )

        lxm, _ = local_delta_xy(xm, ym, x2, y2, reference_course)
        lxl, _ = local_delta_xy(xl, yl, x2, y2, reference_course)

        if abs(lxm) < tol:
            return mid, xm, ym

        if prev_mid is not None and abs(mid - prev_mid) < change_threshold:
            return mid, xm, ym

        if lxm * lxl < 0:
            high = mid
        else:
            low = mid

        prev_mid = mid

    return 0.5 * (low + high), xm, ym


def bisection_method_y(
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
    tol=0.01,
    max_iter=100,
    change_threshold=1e-7,
    roll_in=0.0,
):
    """
    Find del_phi so final y-error in reference frame is approximately zero.
    """
    low = 0.0
    high = np.pi
    prev_mid = None

    xm = ym = None

    for _ in range(max_iter):
        mid = 0.5 * (low + high)

        xm, ym = compute_final_position(
            x_turn,
            y_turn,
            initial_heading,
            psi_2,
            mid,
            g,
            bank_angle,
            V_TAS,
            V_w,
            theta_wa,
            dt,
            aero,
            roll_rate=roll_rate,
            roll_in=roll_in,
        )

        xl, yl = compute_final_position(
            x_turn,
            y_turn,
            initial_heading,
            psi_2,
            low,
            g,
            bank_angle,
            V_TAS,
            V_w,
            theta_wa,
            dt,
            aero,
            roll_rate=roll_rate,
            roll_in=roll_in,
        )

        _, lym = local_delta_xy(xm, ym, x2, y2, reference_course)
        _, lyl = local_delta_xy(xl, yl, x2, y2, reference_course)

        if abs(lym) < tol:
            return mid, xm, ym

        if prev_mid is not None and abs(mid - prev_mid) < change_threshold:
            return mid, xm, ym

        if lym * lyl < 0:
            high = mid
        else:
            low = mid

        prev_mid = mid

    return 0.5 * (low + high), xm, ym


def SBB_BBS_maneuver(waypoints, constants, aero, roll_rate):
    """
    Plan either SBB or BBS. If first handedness fails, retry the opposite.

    Returns:
        DYN, del_phi, success
    """
    x1, y1, x2, y2, chi_1, chi_2 = waypoints
    g, V_TAS, bank_angle0, V_w, theta_wa, dt = constants

    psi_1 = yaw_from_course(chi_1, V_TAS, V_w, theta_wa)
    psi_2 = yaw_from_course(chi_2, V_TAS, V_w, theta_wa)

    goal_lateral, _ = local_delta_xy(x1, y1, x2, y2, chi_1)
    first_turn_right = goal_lateral >= 0.0

    def attempt(turn_right: bool):
        bank_angle = bank_angle0 if turn_right else -bank_angle0

        DYN_turn = compute_turn(
            x1,
            y1,
            psi_1,
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

        x_turn = DYN_turn[-1, 1]
        y_turn = DYN_turn[-1, 2]
        h_turn = DYN_turn[-1, 7]

        del_phi, x_mid, y_mid = bisection_method_x(
            x_turn,
            y_turn,
            h_turn,
            psi_2,
            chi_1,
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
            roll_in=bank_angle,
        )

        if not turn_right:
            del_phi *= -1.0

        _, distance = local_delta_xy(x_mid, y_mid, x2, y2, chi_1)

        if distance > 0.0:
            # SBB
            DYN_straight = compute_straight(
                x1,
                y1,
                chi_1,
                V_TAS,
                V_w,
                theta_wa,
                abs(distance),
                dt,
                aero,
            )

            DYN_turn_1 = compute_turn(
                DYN_straight[-1, 1],
                DYN_straight[-1, 2],
                psi_1,
                psi_2 + del_phi,
                g,
                bank_angle,
                V_TAS,
                V_w,
                theta_wa,
                dt,
                aero,
                roll_rate=roll_rate,
            )

            DYN_turn_2 = compute_turn(
                DYN_turn_1[-1, 1],
                DYN_turn_1[-1, 2],
                DYN_turn_1[-1, 7],
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

            DYN = np.vstack((DYN_straight, DYN_turn_1, DYN_turn_2))

        else:
            # BBS
            del_phi, x_mid, y_mid = bisection_method_x(
                x_turn,
                y_turn,
                h_turn,
                psi_2,
                chi_2,
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
                roll_in=bank_angle,
            )

            if not turn_right:
                del_phi *= -1.0

            DYN_turn_1 = compute_turn(
                x1,
                y1,
                psi_1,
                psi_2 + del_phi,
                g,
                bank_angle,
                V_TAS,
                V_w,
                theta_wa,
                dt,
                aero,
                roll_rate=roll_rate,
            )

            DYN_turn_2 = compute_turn(
                DYN_turn_1[-1, 1],
                DYN_turn_1[-1, 2],
                DYN_turn_1[-1, 7],
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

            x_t2, y_t2, h_t2 = DYN_turn_2[-1, 1], DYN_turn_2[-1, 2], DYN_turn_2[-1, 7]
            chi_t2 = course_from_yaw(h_t2, V_TAS, V_w, theta_wa)

            _, distance = local_delta_xy(x_t2, y_t2, x2, y2, chi_t2)

            DYN_straight = compute_straight(
                x_t2,
                y_t2,
                chi_t2,
                V_TAS,
                V_w,
                theta_wa,
                abs(distance),
                dt,
                aero,
            )

            DYN = np.vstack((DYN_turn_1, DYN_turn_2, DYN_straight))

        final_dist = float(np.hypot(DYN[-1, 1] - x2, DYN[-1, 2] - y2))
        bad_angle = np.isclose((del_phi % (2.0 * np.pi)), np.pi, atol=1e-2)
        ok = (not bad_angle) and final_dist <= 2.0

        return DYN, del_phi, ok

    DYN, del_phi, ok = attempt(first_turn_right)
    if ok:
        return DYN, del_phi, True

    return attempt(not first_turn_right)


def SBB_BBS_inv_maneuver(waypoints, constants, aero, roll_rate):
    """
    Inverse SBB/BBS maneuver. If first handedness fails, retry opposite.
    """
    x1, y1, x2, y2, chi_1, chi_2 = waypoints
    g, V_TAS, bank_angle0, V_w, theta_wa, dt = constants

    psi_1 = yaw_from_course(chi_1, V_TAS, V_w, theta_wa)
    psi_2 = yaw_from_course(chi_2, V_TAS, V_w, theta_wa)

    goal_lateral, _ = local_delta_xy(x1, y1, x2, y2, chi_1)
    first_turn_right = goal_lateral >= 0.0

    def attempt(turn_right: bool):
        bank_angle = bank_angle0 if turn_right else -bank_angle0

        del_phi, x_mid, y_mid = bisection_method_x(
            x1,
            y1,
            psi_1,
            psi_2,
            chi_2,
            g,
            -bank_angle,
            V_TAS,
            V_w,
            theta_wa,
            dt,
            aero,
            x2,
            y2,
            roll_rate=roll_rate,
        )

        if not turn_right:
            del_phi *= -1.0

        _, distance = local_delta_xy(x_mid, y_mid, x2, y2, chi_2)

        if distance > 0.0:
            # BBS
            DYN_turn_a = compute_turn(
                x1,
                y1,
                psi_1,
                psi_1 - del_phi,
                g,
                -bank_angle,
                V_TAS,
                V_w,
                theta_wa,
                dt,
                aero,
                roll_rate=roll_rate,
            )

            DYN_turn_b = compute_turn(
                DYN_turn_a[-1, 1],
                DYN_turn_a[-1, 2],
                DYN_turn_a[-1, 7],
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

            xb, yb, hb = DYN_turn_b[-1, 1], DYN_turn_b[-1, 2], DYN_turn_b[-1, 7]
            chi_tb = course_from_yaw(hb, V_TAS, V_w, theta_wa)

            _, distance = local_delta_xy(xb, yb, x2, y2, chi_tb)

            DYN_straight = compute_straight(
                xb,
                yb,
                chi_tb,
                V_TAS,
                V_w,
                theta_wa,
                abs(distance),
                dt,
                aero,
            )

            DYN = np.vstack((DYN_turn_a, DYN_turn_b, DYN_straight))

        else:
            # SBB
            del_phi, x_mid, y_mid = bisection_method_x(
                x1,
                y1,
                psi_1,
                psi_2,
                chi_1,
                g,
                -bank_angle,
                V_TAS,
                V_w,
                theta_wa,
                dt,
                aero,
                x2,
                y2,
                roll_rate=roll_rate,
            )

            if not turn_right:
                del_phi *= -1.0

            _, distance = local_delta_xy(x_mid, y_mid, x2, y2, chi_1)

            DYN_straight = compute_straight(
                x1,
                y1,
                chi_1,
                V_TAS,
                V_w,
                theta_wa,
                abs(distance),
                dt,
                aero,
            )

            xs, ys, hs = DYN_straight[-1, 1], DYN_straight[-1, 2], DYN_straight[-1, 7]

            DYN_turn_a = compute_turn(
                xs,
                ys,
                hs,
                hs - del_phi,
                g,
                -bank_angle,
                V_TAS,
                V_w,
                theta_wa,
                dt,
                aero,
                roll_rate=roll_rate,
            )

            DYN_turn_b = compute_turn(
                DYN_turn_a[-1, 1],
                DYN_turn_a[-1, 2],
                DYN_turn_a[-1, 7],
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

            DYN = np.vstack((DYN_straight, DYN_turn_a, DYN_turn_b))

        final_dist = float(np.hypot(DYN[-1, 1] - x2, DYN[-1, 2] - y2))
        bad_angle = np.isclose((del_phi % (2.0 * np.pi)), np.pi, atol=1e-2)
        ok = (not bad_angle) and final_dist <= 2.0

        return DYN, del_phi, ok

    DYN, del_phi, ok = attempt(first_turn_right)
    if ok:
        return DYN, del_phi, True

    return attempt(not first_turn_right)
