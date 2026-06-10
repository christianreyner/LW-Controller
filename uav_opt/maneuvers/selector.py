"""
Optimal maneuver selector.

This file replaces the high-level optimal_path(...) function in Trochoidal.py.

It evaluates maneuver candidates and returns the lowest-cost DYN.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

from uav_opt.maneuvers.bsb import BSB_maneuver
from uav_opt.maneuvers.sbb_bbs import (
    SBB_BBS_maneuver,
    SBB_BBS_inv_maneuver,
)
from uav_opt.maneuvers.bbb import BBB_maneuver
from uav_opt.maneuvers.turns import compute_turn


def optimal_path(waypoints, constants, Aero, roll_rate):
    """
    Select best maneuver.

    Args:
        waypoints:
            [x1, y1, x2, y2, chi_1, chi_2]
        constants:
            [g, V_TAS, bank_angle, V_w, theta_wa, dt]
        Aero:
            Aero array/config.
        roll_rate:
            Max roll rate rad/s.

    Returns:
        DYN array.
    """
    plot = False
    candidates = []

    dt = constants[5]

    def fix_time_column(DYN):
        """
        Make time column monotonic after stacking multiple segments.
        """
        if DYN is None or len(DYN) == 0:
            return DYN

        DYN = np.array(DYN, copy=True, dtype=float)
        DYN[:, 0] = np.arange(DYN.shape[0]) * dt
        return DYN

    # ------------------------------------------------------------------
    # 1. Try BSB first.
    # ------------------------------------------------------------------
    DYN_bsb, cost_bsb, ok_bsb = BSB_maneuver(
        waypoints,
        constants,
        Aero,
        roll_rate=roll_rate,
    )

    if ok_bsb and len(DYN_bsb) > 0:
        DYN_bsb = fix_time_column(DYN_bsb)
        cost_bsb = float(np.sum(DYN_bsb[:, 10]))
        candidates.append(("BSB", cost_bsb, DYN_bsb))

        if plot:
            plot_trajectory(DYN_bsb, "BSB", waypoints, constants)

    else:
        # ------------------------------------------------------------------
        # 2. Try SBB/BBS.
        # ------------------------------------------------------------------
        DYN_sbb, del_phi_1, ok_sbb = SBB_BBS_maneuver(
            waypoints,
            constants,
            Aero,
            roll_rate=roll_rate,
        )

        if ok_sbb and len(DYN_sbb) > 0:
            DYN_sbb = fix_time_column(DYN_sbb)
            cost_sbb = float(np.sum(DYN_sbb[:, 10]))
            candidates.append(("SBB/BBS", cost_sbb, DYN_sbb))

            if plot:
                plot_trajectory(DYN_sbb, "SBB/BBS", waypoints, constants)

        # ------------------------------------------------------------------
        # 3. Try inverse SBB/BBS.
        # ------------------------------------------------------------------
        DYN_inv, del_phi_2, ok_inv = SBB_BBS_inv_maneuver(
            waypoints,
            constants,
            Aero,
            roll_rate=roll_rate,
        )

        if ok_inv and len(DYN_inv) > 0:
            DYN_inv = fix_time_column(DYN_inv)
            cost_inv = float(np.sum(DYN_inv[:, 10]))
            candidates.append(("SBB/BBS inverse", cost_inv, DYN_inv))

            if plot:
                plot_trajectory(DYN_inv, "SBB/BBS inverse", waypoints, constants)

        # ------------------------------------------------------------------
        # Optional BBB fallback.
        #
        # Your previous current version had this commented out. I keep the same
        # default behavior by disabling it. Enable if you want.
        # ------------------------------------------------------------------
        use_bbb = False

        if use_bbb and not candidates:
            del_phi = max(abs(del_phi_1), abs(del_phi_2))

            DYN_bbb, ok_bbb = BBB_maneuver(
                del_phi,
                waypoints,
                constants,
                Aero,
                roll_rate=roll_rate,
            )

            if ok_bbb and len(DYN_bbb) > 0:
                DYN_bbb = fix_time_column(DYN_bbb)
                cost_bbb = float(np.sum(DYN_bbb[:, 10]))
                candidates.append(("BBB", cost_bbb, DYN_bbb))

                if plot:
                    plot_trajectory(DYN_bbb, "BBB", waypoints, constants)

    # ------------------------------------------------------------------
    # 4. Fallback: direct turn.
    # ------------------------------------------------------------------
    if not candidates:
        print("No maneuver solution found. Falling back to direct turn.")

        x1, y1, x2, y2, psi_1, psi_2 = waypoints
        g, V_TAS, bank_angle, V_w, theta_wa, dt = constants

        DYN_fallback = compute_turn(
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
            Aero,
            roll_rate=roll_rate,
        )

        return fix_time_column(DYN_fallback)

    # ------------------------------------------------------------------
    # 5. Select lowest cost.
    # ------------------------------------------------------------------
    candidates = sorted(candidates, key=lambda item: item[1])

    best_name, best_cost, best_DYN = candidates[0]

    if len(candidates) >= 2:
        second_name, second_cost, _ = candidates[1]
        print(
            f"Best maneuver: {best_name} (Cost: {best_cost:.3f}); "
            f"Second best: {second_name} (Cost: {second_cost:.3f})"
        )
    else:
        print(f"Best maneuver: {best_name} (Cost: {best_cost:.3f})")

    if plot:
        plt.legend(loc="upper right", fontsize=16)
        plt.grid(True)
        plt.show()

    return best_DYN


def draw_wind_field(ax, V_w, theta_wa_rad, nx=11, ny=11):
    """
    Draw wind vector field in current axes.
    """
    if V_w <= 1e-9:
        return

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span = max(x1 - x0, y1 - y0)

    X, Y = np.meshgrid(
        np.linspace(x0, x1, nx),
        np.linspace(y0, y1, ny),
    )

    U = np.full_like(X, V_w * np.sin(theta_wa_rad + np.pi), dtype=float)
    V = np.full_like(Y, V_w * np.cos(theta_wa_rad + np.pi), dtype=float)

    target_frac = 0.10
    denom = max(1e-6, target_frac * span)
    scale = max(1.0, V_w / denom)

    ax.quiver(
        X,
        Y,
        U,
        V,
        angles="xy",
        scale_units="xy",
        scale=scale,
        width=0.0025,
        headwidth=10,
        headlength=12,
        headaxislength=9,
        minlength=0,
        pivot="middle",
        color="#d62728",
        alpha=0.4,
        zorder=0,
    )


def plot_trajectory(
    DYN,
    tag,
    waypoints,
    constants,
    show_quiver=True,
    keep_previous=False,
    auto_fit=True,
):
    """
    Plot trajectory candidate.
    """
    x1, y1, x2, y2, psi_1, psi_2 = waypoints
    g, V_TAS, bank_angle, V_w, theta_wa, dt = constants

    traj = np.asarray(DYN, dtype=float)

    if traj.size == 0:
        if not keep_previous:
            plt.figure(figsize=(8, 7))
        ax = plt.gca()
        ax.cla()
        ax.set_title("Optimal Path")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.35)
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.5, 0.5, "Empty trajectory", transform=ax.transAxes, ha="center", va="center")
        plt.tight_layout()
        plt.show()
        return

    xs = traj[:, 1]
    ys = traj[:, 2]

    if not keep_previous:
        plt.figure(figsize=(8, 7))

    ax = plt.gca()
    ax.cla()

    ax.set_title(f"Optimal Path: {tag}", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (m)", fontsize=13)
    ax.set_ylabel("Y (m)", fontsize=13)
    ax.grid(True, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")

    if auto_fit:
        xmin, xmax = min(xs.min(), x1, x2), max(xs.max(), x1, x2)
        ymin, ymax = min(ys.min(), y1, y2), max(ys.max(), y1, y2)

        dx = max(xmax - xmin, 1.0)
        dy = max(ymax - ymin, 1.0)

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)

        span = max(dx, dy) * 1.24

        ax.set_xlim(cx - span / 2.0, cx + span / 2.0)
        ax.set_ylim(cy - span / 2.0, cy + span / 2.0)
    else:
        ax.set_xlim(-100, 200)
        ax.set_ylim(-50, 300)

    if show_quiver and V_w > 1e-6:
        draw_wind_field(ax, V_w=V_w, theta_wa_rad=theta_wa)

    ax.plot(xs, ys, color="#1f77b4", lw=2.6, alpha=0.95, zorder=3)

    ax.scatter([x1, x2], [y1, y2], color="#d62728", s=60, zorder=4)
    ax.text(x1, y1, " WP1", fontsize=12, color="#d62728", va="center")
    ax.text(x2, y2, " WP2", fontsize=12, color="#d62728", va="center")

    def add_heading_arrow(x, y, psi, length=12.0, color="#333", z=5):
        dx = length * np.sin(psi)
        dy = length * np.cos(psi)
        arr = FancyArrow(
            x,
            y,
            dx,
            dy,
            width=0.9,
            head_width=3.6,
            head_length=3.8,
            length_includes_head=True,
            color=color,
            alpha=0.9,
            zorder=z,
        )
        ax.add_patch(arr)

    add_heading_arrow(x1, y1, psi_1)
    add_heading_arrow(x2, y2, psi_2)

    total_dist = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    total_time = float(traj[-1, 0]) if traj.shape[1] > 0 else 0.0
    total_cost = float(np.sum(traj[:, 10])) if traj.shape[1] > 10 else 0.0

    ax.text(
        0.02,
        0.98,
        f"Length: {total_dist:.1f} m\n"
        f"Time:   {total_time:.2f} s\n"
        f"Cost:   {total_cost:.3f} Wh",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb"),
        fontfamily="monospace",
        fontsize=10,
        zorder=8,
    )

    plt.tight_layout()
    plt.show()


def evaluate_and_plot_costs(Aero, roll_rate):
    """
    Utility sweep preserved from old Trochoidal.py.
    """
    g = 9.81
    V_TAS = 17.8
    V_w = 5.0
    theta_wa = np.radians(270.0)
    dt = 0.01

    x1, y1 = 0.0, 0.0
    x2, y2 = 100.0, 0.0
    psi_1, psi_2 = 0.0, np.pi

    bank_values = np.radians(np.arange(10.0, 45.0, 1.0))

    BSB_costs = []
    SBB_costs = []
    SBB_inv_costs = []
    BBB_costs = []

    waypoints = [x1, y1, x2, y2, psi_1, psi_2]

    for bank in bank_values:
        print(f"Evaluating bank {np.rad2deg(bank):.1f} deg")

        constants = [g, V_TAS, bank, V_w, theta_wa, dt]

        DYN_bsb, cb, okb = BSB_maneuver(
            waypoints,
            constants,
            Aero,
            roll_rate=roll_rate,
        )
        BSB_costs.append(float(np.sum(DYN_bsb[:, 10])) if okb else np.nan)

        DYN_sbb, _, oks = SBB_BBS_maneuver(
            waypoints,
            constants,
            Aero,
            roll_rate=roll_rate,
        )
        SBB_costs.append(float(np.sum(DYN_sbb[:, 10])) if oks else np.nan)

        DYN_inv, _, oki = SBB_BBS_inv_maneuver(
            waypoints,
            constants,
            Aero,
            roll_rate=roll_rate,
        )
        SBB_inv_costs.append(float(np.sum(DYN_inv[:, 10])) if oki else np.nan)

        BBB_costs.append(np.nan)

    plt.figure(figsize=(7, 5))
    plt.plot(np.rad2deg(bank_values), BSB_costs, label="BSB", marker="o")
    plt.plot(np.rad2deg(bank_values), SBB_costs, label="SBB/BBS", marker="s")
    plt.plot(np.rad2deg(bank_values), SBB_inv_costs, label="SBB/BBS inverse", marker="^")
    plt.plot(np.rad2deg(bank_values), BBB_costs, label="BBB", marker="x")
    plt.xlabel("Bank angle (deg)")
    plt.ylabel("Cost (Wh)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return bank_values, BSB_costs, SBB_costs, SBB_inv_costs, BBB_costs
