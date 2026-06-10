"""
BSB maneuver planner.

BSB means:
    Bang - Singular - Bang

This is usually the preferred efficient connection when enough straight
distance exists between two coordinated turns.
"""

from __future__ import annotations

import numpy as np

from uav_opt.maneuvers.geometry import (
    norm_angle,
    wrap_signed_error,
    local_delta_xy,
    compute_straight_distance,
    yaw_from_course,
)
from uav_opt.maneuvers.turns import compute_turn
from uav_opt.maneuvers.straight import compute_straight


def BSB_maneuver(
    waypoints,
    constants,
    aero,
    roll_rate: float,
    max_iter: int = 50,
    heading_tol: float = 1e-2,
    relax: float = 0.1,
):
    """
    Plan Bang-Singular-Bang maneuver.

    Args:
        waypoints:
            [x1, y1, x2, y2, chi_1, chi_2]
        constants:
            [g, V_TAS, bank_angle, V_w, theta_wa, dt]
        aero:
            Aero array/config.
        roll_rate:
            Max roll rate rad/s.

    Returns:
        DYN, total_cost, success
    """
    x1, y1, x2, y2, chi_1, chi_2 = waypoints
    g, V_TAS, bank_angle_in, V_w, theta_wa, dt = constants

    psi_1 = yaw_from_course(chi_1, V_TAS, V_w, theta_wa)
    psi_2 = yaw_from_course(chi_2, V_TAS, V_w, theta_wa)

    goal_lateral, _ = local_delta_xy(x1, y1, x2, y2, chi_1)

    turn_right = True
    bank_angle = bank_angle_in

    if goal_lateral < 0.0:
        bank_angle = -bank_angle_in
        turn_right = False

    # Quick direct-turn seed.
    DYN_seed = compute_turn(
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
    )

    xs = DYN_seed[-1, 1]
    ys = DYN_seed[-1, 2]

    chi_tan_guess = norm_angle(np.pi / 2.0 - np.arctan2(y2 - ys, x2 - xs))

    chi_diff = norm_angle(chi_2 - chi_tan_guess)

    # Feasibility check: tangent must be reachable with selected turn direction.
    if not ((chi_diff < np.pi and turn_right) or (chi_diff > np.pi and not turn_right)):
        return DYN_seed, float(np.sum(DYN_seed[:, 10])), False

    last_best = None
    best_obj = np.inf

    # ------------------------------------------------------------------
    # Fixed-point tangent-course iteration.
    # ------------------------------------------------------------------
    for _ in range(max_iter):
        psi_tan = yaw_from_course(chi_tan_guess, V_TAS, V_w, theta_wa)

        DYN_t1 = compute_turn(
            x1,
            y1,
            psi_1,
            psi_tan,
            g,
            bank_angle,
            V_TAS,
            V_w,
            theta_wa,
            dt,
            aero,
            roll_rate=roll_rate,
        )

        x_t1, y_t1, h_t1 = DYN_t1[-1, 1], DYN_t1[-1, 2], DYN_t1[-1, 7]

        DYN_t2 = compute_turn(
            x_t1,
            y_t1,
            h_t1,
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

        x_t2, y_t2, h_t2 = DYN_t2[-1, 1], DYN_t2[-1, 2], DYN_t2[-1, 7]

        chi_tan_new = norm_angle(np.pi / 2.0 - np.arctan2(y2 - y_t2, x2 - x_t2))
        chi_diff = norm_angle(chi_2 - chi_tan_new)

        if not ((chi_diff < np.pi and turn_right) or (chi_diff > np.pi and not turn_right)):
            break

        err = wrap_signed_error(chi_tan_new, chi_tan_guess)

        obj = (x2 - x_t2) ** 2 + (y2 - y_t2) ** 2
        if obj < best_obj:
            best_obj = obj
            last_best = (chi_tan_guess, DYN_t1, DYN_t2, x_t1, y_t1, h_t1, x_t2, y_t2, h_t2)

        if abs(err) <= heading_tol:
            straight_dist = compute_straight_distance(x_t2, y_t2, x2, y2)

            DYN_straight = compute_straight(
                x_t1,
                y_t1,
                chi_tan_guess,
                V_TAS,
                V_w,
                theta_wa,
                straight_dist,
                dt,
                aero,
            )

            xs, ys, hs = DYN_straight[-1, 1], DYN_straight[-1, 2], DYN_straight[-1, 7]

            DYN_t2_new = compute_turn(
                xs,
                ys,
                hs,
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

            DYN = np.vstack((DYN_t1, DYN_straight, DYN_t2_new))
            total_cost = float(np.sum(DYN[:, 10]))

            return DYN, total_cost, True

        chi_tan_guess = norm_angle(chi_tan_guess + relax * err)

    # ------------------------------------------------------------------
    # Fallback: partial roll-out case.
    #
    # This preserves the original Trochoidal.py behavior:
    #   - if the residual behaves like an insufficient initial straight,
    #     build Singular -> Bang -> Bang
    #   - otherwise build Bang -> Bang -> Singular
    # ------------------------------------------------------------------

    chi_tan_new_checked = float(chi_tan_guess)
    best_partial = None
    best_obj = np.inf
    best_err = np.inf

    for _outer in range(10):
        lo_lim = 0.0
        hi_lim = bank_angle
        roll_out = bank_angle / 2.0
        straight_dist = 99.0

        DYN_t1 = None
        DYN_t2 = None
        chi_tan_new = chi_tan_guess

        for it in range(max_iter + 1):
            psi_tan = yaw_from_course(chi_tan_guess, V_TAS, V_w, theta_wa)

            # turn_1: initial -> tangent, but do not necessarily roll out to zero
            DYN_t1 = compute_turn(
                x1,
                y1,
                psi_1,
                psi_tan,
                g,
                bank_angle,
                V_TAS,
                V_w,
                theta_wa,
                dt,
                aero,
                roll_rate=roll_rate,
                roll_out=roll_out,
            )

            x_t1, y_t1, h_t1 = DYN_t1[-1, 1], DYN_t1[-1, 2], DYN_t1[-1, 7]

            # turn_2: continue from partial roll-out state
            DYN_t2 = compute_turn(
                x_t1,
                y_t1,
                h_t1,
                psi_2,
                g,
                bank_angle,
                V_TAS,
                V_w,
                theta_wa,
                dt,
                aero,
                roll_rate=roll_rate,
                roll_in=roll_out,
            )

            x_t2, y_t2, h_t2 = DYN_t2[-1, 1], DYN_t2[-1, 2], DYN_t2[-1, 7]

            # tangent course from end of second bend to target
            chi_tan_new = norm_angle(np.pi / 2.0 - np.arctan2(y2 - y_t2, x2 - x_t2))
            chi_diff = norm_angle(chi_2 - chi_tan_new)

            tangent_check = (
                (chi_diff < np.pi and turn_right)
                or
                (chi_diff > np.pi and not turn_right)
            )

            straight_dist_old = straight_dist
            straight_dist = compute_straight_distance(x_t2, y_t2, x2, y2)

            residual = abs(1.0 - straight_dist / max(straight_dist_old, 1e-9))

            if tangent_check:
                chi_tan_new_checked = chi_tan_new
                lim_range = abs(hi_lim - lo_lim)

                if residual > 0.01 or lim_range > np.deg2rad(5.0):
                    hi_lim = roll_out
                    roll_out = 0.5 * (hi_lim + lo_lim)
                else:
                    break
            else:
                lo_lim = roll_out
                roll_out = 0.5 * (hi_lim + lo_lim)

            # If the two bends already end close enough to the target,
            # no straight segment is needed.
            if straight_dist < 0.5:
                DYN = np.vstack((DYN_t1, DYN_t2))
                total_cost = float(np.sum(DYN[:, 10]))
                return DYN, total_cost, True

        if DYN_t1 is None or DYN_t2 is None:
            continue

        err = wrap_signed_error(chi_tan_new_checked, chi_tan_guess)

        if abs(err) <= heading_tol and straight_dist < 0.5:
            DYN = np.vstack((DYN_t1, DYN_t2))
            total_cost = float(np.sum(DYN[:, 10]))
            return DYN, total_cost, True

        obj = (x2 - x_t2) ** 2 + (y2 - y_t2) ** 2

        if obj < best_obj:
            best_obj = obj
            best_err = err
            best_partial = (
                chi_tan_guess,
                chi_tan_new,
                straight_dist,
                roll_out,
                DYN_t1,
                DYN_t2,
            )

        chi_tan_guess = norm_angle(chi_tan_guess + 0.2 * err)

    # ------------------------------------------------------------------
    # Final fallback assembly.
    #
    # This is the part I got wrong before.
    # The old code had two possible assemblies:
    #
    #   1. Singular -> Bang -> Bang
    #   2. Bang -> Bang -> Singular
    #
    # My previous refactor only kept case 2.
    # ------------------------------------------------------------------

    if best_partial is None:
        return DYN_seed, float(np.sum(DYN_seed[:, 10])), False

    chi_tan_guess, chi_tan_new, straight_dist, roll_out, DYN_t1, DYN_t2 = best_partial

    if abs(best_err) <= heading_tol:
        # Most probably SB/SBB-like:
        # build straight first, then two bends.

        DYN_straight = compute_straight(
            x1,
            y1,
            chi_tan_new,
            V_TAS,
            V_w,
            theta_wa,
            straight_dist,
            dt,
            aero,
        )

        xs, ys, hs = DYN_straight[-1, 1], DYN_straight[-1, 2], DYN_straight[-1, 7]

        psi_tan = yaw_from_course(chi_tan_guess, V_TAS, V_w, theta_wa)

        DYN_t1_new = compute_turn(
            xs,
            ys,
            psi_1,
            psi_tan,
            g,
            bank_angle,
            V_TAS,
            V_w,
            theta_wa,
            dt,
            aero,
            roll_rate=roll_rate,
            roll_out=roll_out,
        )

        x_t1, y_t1, h_t1 = DYN_t1_new[-1, 1], DYN_t1_new[-1, 2], DYN_t1_new[-1, 7]

        DYN_t2_new = compute_turn(
            x_t1,
            y_t1,
            h_t1,
            psi_2,
            g,
            bank_angle,
            V_TAS,
            V_w,
            theta_wa,
            dt,
            aero,
            roll_rate=roll_rate,
            roll_in=roll_out,
        )

        DYN = np.vstack((DYN_straight, DYN_t1_new, DYN_t2_new))

    else:
        # Most probably BS:
        # build two bends first, then straight.

        x_t2, y_t2, h_t2 = DYN_t2[-1, 1], DYN_t2[-1, 2], DYN_t2[-1, 7]

        DYN_straight = compute_straight(
            x_t2,
            y_t2,
            chi_tan_new,
            V_TAS,
            V_w,
            theta_wa,
            straight_dist,
            dt,
            aero,
        )

        DYN = np.vstack((DYN_t1, DYN_t2, DYN_straight))

    total_cost = float(np.sum(DYN[:, 10]))
    return DYN, total_cost, True
