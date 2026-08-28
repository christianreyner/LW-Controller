#!/usr/bin/env python3
"""
Wind-aware bank-angle solver with roll dynamics and diagnostic plotting.

The primary function is:

    desired_bank_to_point_with_wind_roll(...)

It computes a commanded bank angle that attempts to make the aircraft's
ground trajectory pass through or close to a target point.

Debugging features:
- Plot the selected aircraft trajectory.
- Plot wings-level and bank-limit trajectories for comparison.
- Mark the target and selected closest-approach point.
- Plot bank angle, heading, and target distance versus time.
- Plot signed closest-approach error versus commanded bank.
- Plot all bisection evaluations.
- Identify closest approaches that occur at the simulation boundaries.

Conventions:
- x axis: East [m]
- y axis: North [m]
- psi: heading [rad], clockwise from North
- phi: bank angle [rad], positive for a right turn
- Wind direction is the direction FROM which the wind blows
- theta_wa uses the same clockwise-from-North convention
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. Data structures
# ============================================================

@dataclass
class TurnWithWindParams:
    """
    Parameters for a constant-bank coordinated turn with wind.
    """

    x0: float
    y0: float
    psi0: float
    g: float
    V_TAS: float
    Vw_x: float
    Vw_y: float


@dataclass
class TurnWithWindRollParams:
    """
    Parameters for a coordinated turn with wind and roll dynamics.
    """

    x0: float
    y0: float
    psi0: float
    phi0: float
    g: float
    V_TAS: float
    Vw_x: float
    Vw_y: float
    p_max: float


# ============================================================
# 2. General utility functions
# ============================================================

def wind_components_from_direction(V_w, theta_wa):
    """
    Convert wind speed and wind-from direction to East/North components.

    Parameters
    ----------
    V_w : float
        Wind speed [m/s].
    theta_wa : float
        Direction FROM which the wind blows [rad], clockwise from North.

    Returns
    -------
    Vw_x, Vw_y : float
        Wind vector components [m/s], East and North respectively.
    """
    Vw_x = V_w * np.sin(theta_wa + np.pi)
    Vw_y = V_w * np.cos(theta_wa + np.pi)
    return float(Vw_x), float(Vw_y)


def interpolate_trajectory_state(t_query, t, x, y, psi, phi):
    """
    Linearly interpolate a trajectory state at a requested time.
    """
    t_query = float(np.clip(t_query, t[0], t[-1]))

    return {
        "t": t_query,
        "x": float(np.interp(t_query, t, x)),
        "y": float(np.interp(t_query, t, y)),
        "psi": float(np.interp(t_query, t, psi)),
        "phi": float(np.interp(t_query, t, phi)),
    }


def parabolic_minimum_time(t3, f3):
    """
    Estimate the minimum of a parabola passing through three samples.

    Parameters
    ----------
    t3 : array-like, shape (3,)
        Three times.
    f3 : array-like, shape (3,)
        Function values at the three times.

    Returns
    -------
    t_vertex : float or None
        Estimated minimum time. None means the fit is invalid or concave.
    """
    tm1, t0, tp1 = np.asarray(t3, dtype=float)
    fm1, f0, fp1 = np.asarray(f3, dtype=float)

    denom = (
        (tm1 - t0)
        * (tm1 - tp1)
        * (t0 - tp1)
    )

    if abs(denom) <= 1e-14:
        return None

    a = (
        fm1 * (t0 - tp1)
        + f0 * (tp1 - tm1)
        + fp1 * (tm1 - t0)
    ) / denom

    b = (
        fm1 * (tp1**2 - t0**2)
        + f0 * (tm1**2 - tp1**2)
        + fp1 * (t0**2 - tm1**2)
    ) / denom

    if a <= 0.0:
        return None

    t_vertex = -b / (2.0 * a)

    if tm1 <= t_vertex <= tp1:
        return float(t_vertex)

    return None


# ============================================================
# 3. Analytic constant-bank trajectory
# ============================================================

def position_with_wind_const_bank(t, phi, p: TurnWithWindParams):
    """
    Analytic position for a constant-bank turn in constant wind.

    Model
    -----
        dpsi/dt = (g / V_TAS) * tan(phi)

        v_air = V_TAS * [sin(psi), cos(psi)]
        v_gnd = v_air + v_wind
    """
    t = np.asarray(t, dtype=float)

    x0 = p.x0
    y0 = p.y0
    psi0 = p.psi0
    g = p.g
    V = p.V_TAS
    Vw_x = p.Vw_x
    Vw_y = p.Vw_y

    if abs(phi) < 1e-9:
        vx = V * np.sin(psi0) + Vw_x
        vy = V * np.cos(psi0) + Vw_y

        x = x0 + vx * t
        y = y0 + vy * t
        return x, y

    Omega = (g / V) * np.tan(phi)

    if abs(Omega) < 1e-9:
        vx = V * np.sin(psi0) + Vw_x
        vy = V * np.cos(psi0) + Vw_y

        x = x0 + vx * t
        y = y0 + vy * t
        return x, y

    psi_t = psi0 + Omega * t

    x = (
        x0
        - V * (np.cos(psi_t) - np.cos(psi0)) / Omega
        + Vw_x * t
    )

    y = (
        y0
        + V * (np.sin(psi_t) - np.sin(psi0)) / Omega
        + Vw_y * t
    )

    return x, y


def heading_at_time(phi, t, p: TurnWithWindParams):
    """
    Heading for a constant-bank turn.
    """
    t = np.asarray(t, dtype=float)

    if abs(phi) < 1e-9:
        return np.full_like(t, p.psi0, dtype=float)

    Omega = (p.g / p.V_TAS) * np.tan(phi)
    return p.psi0 + Omega * t


# ============================================================
# 4. Numerical trajectory with roll dynamics
# ============================================================

def trajectory_with_wind_roll(
    phi_cmd,
    p: TurnWithWindRollParams,
    t_final,
    dt=0.05,
    max_turns=1.0,
):
    """
    Simulate the aircraft trajectory with roll dynamics and wind.

    The trajectory ends when either:
      1. t_final is reached, or
      2. max_turns of accumulated heading rotation is reached.

    Set max_turns=None to disable the turn limiter.
    """
    if t_final <= 0.0:
        raise ValueError("t_final must be greater than zero.")

    if dt <= 0.0:
        raise ValueError("dt must be greater than zero.")

    if p.V_TAS <= 0.0:
        raise ValueError("V_TAS must be greater than zero.")

    # Full time grid up to t_final.
    n_intervals = max(1, int(np.ceil(t_final / dt)))
    t = np.linspace(0.0, t_final, n_intervals + 1)
    dt_steps = np.diff(t)

    n = len(t)

    # --------------------------------------------------------
    # Bank profile
    # --------------------------------------------------------
    dphi_total = phi_cmd - p.phi0

    if abs(dphi_total) < 1e-12 or p.p_max <= 0.0:
        phi = np.full(n, p.phi0, dtype=float)
    else:
        roll_direction = np.sign(dphi_total)
        phi_ramp = p.phi0 + roll_direction * p.p_max * t

        if roll_direction > 0.0:
            phi = np.minimum(phi_ramp, phi_cmd)
        else:
            phi = np.maximum(phi_ramp, phi_cmd)

    # --------------------------------------------------------
    # Heading
    # --------------------------------------------------------
    psi = np.empty(n, dtype=float)
    psi[0] = p.psi0

    phi_for_psi = phi[1:].copy()
    phi_for_psi[np.abs(phi_for_psi) < 1e-9] = 0.0

    psi_dot = (p.g / p.V_TAS) * np.tan(phi_for_psi)
    delta_psi = psi_dot * dt_steps

    psi[1:] = p.psi0 + np.cumsum(delta_psi)

    # --------------------------------------------------------
    # Stop after max_turns of accumulated heading rotation
    # --------------------------------------------------------
    if max_turns is not None and max_turns > 0.0:
        accumulated_turn = np.concatenate(
            ([0.0], np.cumsum(np.abs(delta_psi)))
        )

        maximum_rotation = max_turns * 2.0 * np.pi

        reached = np.nonzero(
            accumulated_turn >= maximum_rotation
        )[0]

        if reached.size > 0:
            stop_index = int(reached[0])

            # Keep the sample where the turn limit was reached.
            t = t[:stop_index + 1]
            phi = phi[:stop_index + 1]
            psi = psi[:stop_index + 1]
            dt_steps = np.diff(t)
            n = len(t)

    # --------------------------------------------------------
    # Ground velocity
    # --------------------------------------------------------
    vx_air = p.V_TAS * np.sin(psi)
    vy_air = p.V_TAS * np.cos(psi)

    vx_ground = vx_air + p.Vw_x
    vy_ground = vy_air + p.Vw_y

    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------
    x = np.empty(n, dtype=float)
    y = np.empty(n, dtype=float)

    x[0] = p.x0
    y[0] = p.y0

    if n > 1:
        x[1:] = p.x0 + np.cumsum(
            vx_ground[1:] * dt_steps
        )
        y[1:] = p.y0 + np.cumsum(
            vy_ground[1:] * dt_steps
        )

    return t, x, y, psi, phi


# ============================================================
# 5. Closest-approach analysis
# ============================================================

def analyze_closest_approach_from_trajectory(
    t,
    x,
    y,
    psi,
    phi,
    p: TurnWithWindRollParams,
    xt,
    yt,
):
    """
    Analyze the earliest local closest approach in a stored trajectory.

    The function first identifies local minima in squared distance. If no
    local minimum exists, it uses the global discrete minimum.

    A parabolic fit is used to refine the closest-approach time when the
    selected point is an interior sample.

    Returns a dictionary containing:
    - signed cross-track error
    - closest distance
    - closest-approach time
    - interpolated state
    - all local-minimum indices
    - whether the minimum is at a simulation boundary
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    psi = np.asarray(psi, dtype=float)
    phi = np.asarray(phi, dtype=float)

    dx = x - xt
    dy = y - yt
    d2 = dx**2 + dy**2
    distance = np.sqrt(d2)

    n = len(t)

    if n >= 3:
        is_local_minimum = (
            (d2[1:-1] <= d2[:-2])
            & (d2[1:-1] <= d2[2:])
        )
        local_min_indices = np.nonzero(is_local_minimum)[0] + 1
    else:
        local_min_indices = np.array([], dtype=int)

    used_global_fallback = local_min_indices.size == 0

    if used_global_fallback:
        i_min = int(np.argmin(d2))
    else:
        # Preserve the original behavior: use the earliest local minimum.
        i_min = int(local_min_indices[0])

    t_min = float(t[i_min])

    if 0 < i_min < n - 1:
        refined_time = parabolic_minimum_time(
            t[i_min - 1:i_min + 2],
            d2[i_min - 1:i_min + 2],
        )

        if refined_time is not None:
            t_min = refined_time

    state = interpolate_trajectory_state(
        t_min,
        t,
        x,
        y,
        psi,
        phi,
    )

    rx = xt - state["x"]
    ry = yt - state["y"]

    d_min = float(np.hypot(rx, ry))

    vx = p.V_TAS * np.sin(state["psi"]) + p.Vw_x
    vy = p.V_TAS * np.cos(state["psi"]) + p.Vw_y

    cross_z = vx * ry - vy * rx

    if d_min <= 1e-12:
        err = 0.0
    else:
        err = float(np.sign(cross_z) * d_min)

    time_epsilon = max(1e-9, 0.51 * np.min(np.diff(t)))

    at_start = t_min <= t[0] + time_epsilon
    at_end = t_min >= t[-1] - time_epsilon

    if at_start:
        minimum_location = "start_boundary"
    elif at_end:
        minimum_location = "end_boundary"
    else:
        minimum_location = "interior"

    return {
        "err": err,
        "t_min": t_min,
        "d_min": d_min,
        "state": state,
        "vx_ground": float(vx),
        "vy_ground": float(vy),
        "cross_z": float(cross_z),
        "distance": distance,
        "distance_squared": d2,
        "local_min_indices": local_min_indices,
        "selected_index": i_min,
        "used_global_fallback": used_global_fallback,
        "minimum_location": minimum_location,
        "at_start": at_start,
        "at_end": at_end,
    }


def closest_approach_error(
    phi,
    p: TurnWithWindParams,
    xt,
    yt,
    t_max=200.0,
    n_grid=400,
    max_turns=1.0,
):
    """
    Closest-approach error for an analytic constant-bank trajectory.
    """
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")

    if abs(phi) < 1e-9:
        Omega = 0.0
    else:
        Omega = (p.g / p.V_TAS) * np.tan(phi)

    if (
        max_turns is not None
        and max_turns > 0.0
        and abs(Omega) > 1e-6
    ):
        one_turn_time = 2.0 * np.pi / abs(Omega)
        t_end = min(t_max, max_turns * one_turn_time)
    else:
        t_end = t_max

    t_grid = np.linspace(0.0, t_end, n_grid)

    x, y = position_with_wind_const_bank(t_grid, phi, p)

    dx = x - xt
    dy = y - yt
    d2 = dx**2 + dy**2

    if n_grid >= 3:
        is_local_minimum = (
            (d2[1:-1] <= d2[:-2])
            & (d2[1:-1] <= d2[2:])
        )
        candidates = np.nonzero(is_local_minimum)[0] + 1
    else:
        candidates = np.array([], dtype=int)

    if candidates.size == 0:
        i_min = int(np.argmin(d2))
    else:
        i_min = int(candidates[0])

    t_min = float(t_grid[i_min])

    if 0 < i_min < n_grid - 1:
        refined_time = parabolic_minimum_time(
            t_grid[i_min - 1:i_min + 2],
            d2[i_min - 1:i_min + 2],
        )

        if refined_time is not None:
            t_min = refined_time

    x_min, y_min = position_with_wind_const_bank(t_min, phi, p)

    x_min = float(np.asarray(x_min))
    y_min = float(np.asarray(y_min))

    rx = xt - x_min
    ry = yt - y_min
    d_min = float(np.hypot(rx, ry))

    psi_min = float(heading_at_time(phi, t_min, p))

    vx = p.V_TAS * np.sin(psi_min) + p.Vw_x
    vy = p.V_TAS * np.cos(psi_min) + p.Vw_y

    cross_z = vx * ry - vy * rx

    if d_min <= 1e-12:
        err = 0.0
    else:
        err = float(np.sign(cross_z) * d_min)

    return err, t_min, d_min


def closest_approach_error_with_roll(
    phi_cmd,
    p: TurnWithWindRollParams,
    xt,
    yt,
    t_max=200.0,
    dt=0.05,
    max_turns=1.0,
    return_details=False,
):
    t, x, y, psi, phi = trajectory_with_wind_roll(
        phi_cmd=phi_cmd,
        p=p,
        t_final=t_max,
        dt=dt,
        max_turns=max_turns,
    )

    closest = analyze_closest_approach_from_trajectory(
        t=t,
        x=x,
        y=y,
        psi=psi,
        phi=phi,
        p=p,
        xt=xt,
        yt=yt,
    )

    if return_details:
        details = {
            "phi_cmd": float(phi_cmd),
            "t": t,
            "x": x,
            "y": y,
            "psi": psi,
            "phi": phi,
            "closest": closest,
            "turns_completed": float(
                np.sum(np.abs(np.diff(psi))) / (2.0 * np.pi)
            ),
            "stopped_by_turn_limit": bool(
                max_turns is not None
                and max_turns > 0.0
                and np.sum(np.abs(np.diff(psi)))
                    >= max_turns * 2.0 * np.pi - 1e-6
            ),
        }

        return (
            closest["err"],
            closest["t_min"],
            closest["d_min"],
            details,
        )

    return (
        closest["err"],
        closest["t_min"],
        closest["d_min"],
    )


# ============================================================
# 6. Constant-bank solver
# ============================================================

def desired_bank_to_point_with_wind(
    x0,
    y0,
    psi0,
    xt,
    yt,
    g,
    V_TAS,
    V_w,
    theta_wa,
    t_max=200.0,
    phi_max=np.deg2rad(60.0),
    tol_pos=1.0,
    max_iter=40,
):
    """
    Compute a constant bank angle without roll dynamics.
    """
    Vw_x, Vw_y = wind_components_from_direction(V_w, theta_wa)

    p = TurnWithWindParams(
        x0=float(x0),
        y0=float(y0),
        psi0=float(psi0),
        g=float(g),
        V_TAS=float(V_TAS),
        Vw_x=Vw_x,
        Vw_y=Vw_y,
    )

    history = []

    def evaluate(phi):
        err, t_min, d_min = closest_approach_error(
            phi,
            p,
            xt,
            yt,
            t_max=t_max,
            max_turns=1.0,
        )

        history.append({
            "phi": float(phi),
            "err": float(err),
            "t_min": float(t_min),
            "d_min": float(d_min),
        })

        return err, t_min, d_min

    err0, t0, d0 = evaluate(0.0)

    if abs(err0) <= tol_pos:
        return 0.0, {
            "phi_solution": 0.0,
            "err": err0,
            "t_min": t0,
            "d_min": d0,
            "bracket": (0.0, 0.0),
            "converged": True,
            "note": "wings_level_sufficient",
            "history": history,
        }

    phi_left = -abs(phi_max)
    phi_right = abs(phi_max)

    err_left, t_left, d_left = evaluate(phi_left)
    err_right, t_right, d_right = evaluate(phi_right)

    if abs(err_left) <= tol_pos:
        return phi_left, {
            "phi_solution": phi_left,
            "err": err_left,
            "t_min": t_left,
            "d_min": d_left,
            "bracket": (phi_left, phi_left),
            "converged": True,
            "note": "left_boundary_solution",
            "history": history,
        }

    if abs(err_right) <= tol_pos:
        return phi_right, {
            "phi_solution": phi_right,
            "err": err_right,
            "t_min": t_right,
            "d_min": d_right,
            "bracket": (phi_right, phi_right),
            "converged": True,
            "note": "right_boundary_solution",
            "history": history,
        }

    if err_left * err_right > 0.0:
        if abs(err_left) <= abs(err_right):
            phi_solution = phi_left
            err_solution = err_left
            t_solution = t_left
            d_solution = d_left
        else:
            phi_solution = phi_right
            err_solution = err_right
            t_solution = t_right
            d_solution = d_right

        return phi_solution, {
            "phi_solution": phi_solution,
            "err": err_solution,
            "t_min": t_solution,
            "d_min": d_solution,
            "bracket": (phi_left, phi_right),
            "converged": False,
            "reason": "no_sign_change",
            "history": history,
        }

    a = phi_left
    b = phi_right
    fa = err_left
    fb = err_right

    phi_solution = 0.5 * (a + b)
    err_solution = np.nan
    t_solution = np.nan
    d_solution = np.nan
    converged = False

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc, tc, dc = evaluate(c)

        phi_solution = c
        err_solution = fc
        t_solution = tc
        d_solution = dc

        if abs(fc) <= tol_pos:
            converged = True
            break

        if fa * fc < 0.0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    info = {
        "phi_solution": phi_solution,
        "err": err_solution,
        "t_min": t_solution,
        "d_min": d_solution,
        "bracket": (a, b),
        "converged": converged,
        "history": history,
    }

    if not converged:
        info["reason"] = "max_iterations"

    return phi_solution, info


# ============================================================
# 7. Roll-aware solver
# ============================================================

def desired_bank_to_point_with_wind_roll(
    x0,
    y0,
    psi0,
    phi0,
    xt,
    yt,
    g,
    V_TAS,
    V_w,
    theta_wa,
    p_max_roll=np.deg2rad(15.0),
    t_max=200.0,
    dt=0.05,
    phi_max=np.deg2rad(60.0),
    tol_pos=1.0,
    max_iter=40,
    debug_plot=False,
    plot_on_failure=False,
    plot_scan_points=121,
    plot_filename=None,
    show_plot=True,
):
    """
    Compute commanded bank with roll dynamics and constant wind.

    Parameters
    ----------
    debug_plot : bool
        If True, always generate a diagnostic plot.

    plot_on_failure : bool
        If True, generate a diagnostic plot when the solver does not
        converge.

    plot_scan_points : int
        Number of bank commands evaluated in the diagnostic error scan.

    plot_filename : str or None
        If provided, save the diagnostic figure to this file.

    show_plot : bool
        If True, call plt.show() after generating the figure.

    Returns
    -------
    phi_cmd : float
        Selected commanded bank [rad].

    info : dict
        Solver and trajectory diagnostics. The selected trajectory is
        available under:

            info["trajectory"]
    """
    if V_TAS <= 0.0:
        raise ValueError("V_TAS must be greater than zero.")

    if g <= 0.0:
        raise ValueError("g must be greater than zero.")

    if t_max <= 0.0:
        raise ValueError("t_max must be greater than zero.")

    if dt <= 0.0:
        raise ValueError("dt must be greater than zero.")

    if phi_max <= 0.0:
        raise ValueError("phi_max must be greater than zero.")

    Vw_x, Vw_y = wind_components_from_direction(V_w, theta_wa)

    p = TurnWithWindRollParams(
        x0=float(x0),
        y0=float(y0),
        psi0=float(psi0),
        phi0=float(phi0),
        g=float(g),
        V_TAS=float(V_TAS),
        Vw_x=float(Vw_x),
        Vw_y=float(Vw_y),
        p_max=float(abs(p_max_roll)),
    )

    history = []

    def evaluate(phi_cmd):
        err, t_min, d_min, details = closest_approach_error_with_roll(
            phi_cmd=phi_cmd,
            p=p,
            xt=xt,
            yt=yt,
            t_max=t_max,
            dt=dt,
            return_details=True,
        )

        closest = details["closest"]

        history.append({
            "phi": float(phi_cmd),
            "err": float(err),
            "t_min": float(t_min),
            "d_min": float(d_min),
            "minimum_location": closest["minimum_location"],
            "used_global_fallback": closest["used_global_fallback"],
        })

        return err, t_min, d_min

    # --------------------------------------------------------
    # 1. Wings-level command
    # --------------------------------------------------------
    err0, t0, d0 = evaluate(0.0)

    if abs(err0) <= tol_pos:
        phi_solution = 0.0

        info = {
            "phi_solution": phi_solution,
            "err": err0,
            "t_min": t0,
            "d_min": d0,
            "bracket": (0.0, 0.0),
            "converged": True,
            "note": "wings_level_sufficient",
            "history": history,
        }

        return _finalize_roll_solution(
            phi_solution=phi_solution,
            info=info,
            p=p,
            xt=xt,
            yt=yt,
            t_max=t_max,
            dt=dt,
            phi_max=phi_max,
            tol_pos=tol_pos,
            debug_plot=debug_plot,
            plot_on_failure=plot_on_failure,
            plot_scan_points=plot_scan_points,
            plot_filename=plot_filename,
            show_plot=show_plot,
        )

    # --------------------------------------------------------
    # 2. Evaluate bank limits
    # --------------------------------------------------------
    phi_left = -abs(phi_max)
    phi_right = abs(phi_max)

    err_left, t_left, d_left = evaluate(phi_left)
    err_right, t_right, d_right = evaluate(phi_right)

    if abs(err_left) <= tol_pos:
        info = {
            "phi_solution": phi_left,
            "err": err_left,
            "t_min": t_left,
            "d_min": d_left,
            "bracket": (phi_left, phi_left),
            "converged": True,
            "note": "left_boundary_solution",
            "history": history,
        }

        return _finalize_roll_solution(
            phi_solution=phi_left,
            info=info,
            p=p,
            xt=xt,
            yt=yt,
            t_max=t_max,
            dt=dt,
            phi_max=phi_max,
            tol_pos=tol_pos,
            debug_plot=debug_plot,
            plot_on_failure=plot_on_failure,
            plot_scan_points=plot_scan_points,
            plot_filename=plot_filename,
            show_plot=show_plot,
        )

    if abs(err_right) <= tol_pos:
        info = {
            "phi_solution": phi_right,
            "err": err_right,
            "t_min": t_right,
            "d_min": d_right,
            "bracket": (phi_right, phi_right),
            "converged": True,
            "note": "right_boundary_solution",
            "history": history,
        }

        return _finalize_roll_solution(
            phi_solution=phi_right,
            info=info,
            p=p,
            xt=xt,
            yt=yt,
            t_max=t_max,
            dt=dt,
            phi_max=phi_max,
            tol_pos=tol_pos,
            debug_plot=debug_plot,
            plot_on_failure=plot_on_failure,
            plot_scan_points=plot_scan_points,
            plot_filename=plot_filename,
            show_plot=show_plot,
        )

    # --------------------------------------------------------
    # 3. Check initial bracket
    # --------------------------------------------------------
    if err_left * err_right > 0.0:
        if abs(err_left) <= abs(err_right):
            phi_solution = phi_left
            err_solution = err_left
            t_solution = t_left
            d_solution = d_left
        else:
            phi_solution = phi_right
            err_solution = err_right
            t_solution = t_right
            d_solution = d_right

        info = {
            "phi_solution": phi_solution,
            "err": err_solution,
            "t_min": t_solution,
            "d_min": d_solution,
            "bracket": (phi_left, phi_right),
            "converged": False,
            "reason": "suboptimal",
            "history": history,
        }

        return _finalize_roll_solution(
            phi_solution=phi_solution,
            info=info,
            p=p,
            xt=xt,
            yt=yt,
            t_max=t_max,
            dt=dt,
            phi_max=phi_max,
            tol_pos=tol_pos,
            debug_plot=debug_plot,
            plot_on_failure=plot_on_failure,
            plot_scan_points=plot_scan_points,
            plot_filename=plot_filename,
            show_plot=show_plot,
        )

    # --------------------------------------------------------
    # 4. Bisection
    # --------------------------------------------------------
    a = phi_left
    b = phi_right
    fa = err_left
    fb = err_right

    phi_solution = 0.5 * (a + b)
    err_solution = np.nan
    t_solution = np.nan
    d_solution = np.nan
    converged = False

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc, tc, dc = evaluate(c)

        phi_solution = c
        err_solution = fc
        t_solution = tc
        d_solution = dc

        if abs(fc) <= tol_pos:
            converged = True
            break

        if fa * fc < 0.0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    info = {
        "phi_solution": phi_solution,
        "err": err_solution,
        "t_min": t_solution,
        "d_min": d_solution,
        "bracket": (a, b),
        "converged": converged,
        "history": history,
    }

    if not converged:
        info["reason"] = "max_iterations"

    return _finalize_roll_solution(
        phi_solution=phi_solution,
        info=info,
        p=p,
        xt=xt,
        yt=yt,
        t_max=t_max,
        dt=dt,
        phi_max=phi_max,
        tol_pos=tol_pos,
        debug_plot=debug_plot,
        plot_on_failure=plot_on_failure,
        plot_scan_points=plot_scan_points,
        plot_filename=plot_filename,
        show_plot=show_plot,
    )


def _finalize_roll_solution(
    phi_solution,
    info,
    p,
    xt,
    yt,
    t_max,
    dt,
    phi_max,
    tol_pos,
    debug_plot,
    plot_on_failure,
    plot_scan_points,
    plot_filename,
    show_plot,
):
    """
    Store the selected trajectory and optionally generate a debug plot.
    """
    _, _, _, trajectory = closest_approach_error_with_roll(
        phi_cmd=phi_solution,
        p=p,
        xt=xt,
        yt=yt,
        t_max=t_max,
        dt=dt,
        return_details=True,
    )

    info["trajectory"] = trajectory

    info["debug_context"] = {
        "params": p,
        "xt": float(xt),
        "yt": float(yt),
        "t_max": float(t_max),
        "dt": float(dt),
        "phi_max": float(phi_max),
        "tol_pos": float(tol_pos),
    }

    closest = trajectory["closest"]

    if closest["at_end"]:
        info["warning"] = (
            "Selected closest approach occurs at t_max. "
            "The aircraft may still be approaching the target; "
            "increase t_max."
        )
    elif closest["at_start"]:
        info["warning"] = (
            "Selected closest approach occurs at t=0. "
            "The aircraft immediately moves away from the target."
        )
    elif closest["used_global_fallback"]:
        info["warning"] = (
            "No interior local minimum was detected. "
            "The global sampled minimum was used."
        )

    should_plot = debug_plot or (
        plot_on_failure and not info["converged"]
    )

    if should_plot:
        figure = plot_roll_solution_debug(
            info=info,
            scan_points=plot_scan_points,
            filename=plot_filename,
            show=show_plot,
        )
        info["debug_figure"] = figure

    return float(phi_solution), info


# ============================================================
# 8. Diagnostic plotting
# ============================================================

def generate_bank_error_scan(info, scan_points=121):
    """
    Evaluate signed closest-approach error across the full bank range.

    This is useful for detecting:
    - multiple roots
    - discontinuous jumps caused by switching local minima
    - roots that exist even though the two endpoints have the same sign
    - closest approaches at t_max
    """
    context = info["debug_context"]

    p = context["params"]
    xt = context["xt"]
    yt = context["yt"]
    t_max = context["t_max"]
    dt = context["dt"]
    phi_max = context["phi_max"]

    scan_points = max(3, int(scan_points))

    phi_scan = np.linspace(-phi_max, phi_max, scan_points)

    err_scan = np.empty(scan_points)
    d_min_scan = np.empty(scan_points)
    t_min_scan = np.empty(scan_points)
    boundary_scan = np.zeros(scan_points, dtype=bool)
    fallback_scan = np.zeros(scan_points, dtype=bool)

    for i, phi_cmd in enumerate(phi_scan):
        err, t_min, d_min, details = closest_approach_error_with_roll(
            phi_cmd=phi_cmd,
            p=p,
            xt=xt,
            yt=yt,
            t_max=t_max,
            dt=dt,
            return_details=True,
        )

        err_scan[i] = err
        d_min_scan[i] = d_min
        t_min_scan[i] = t_min

        closest = details["closest"]

        boundary_scan[i] = (
            closest["at_start"]
            or closest["at_end"]
        )

        fallback_scan[i] = closest["used_global_fallback"]

    return {
        "phi": phi_scan,
        "err": err_scan,
        "d_min": d_min_scan,
        "t_min": t_min_scan,
        "boundary": boundary_scan,
        "fallback": fallback_scan,
    }


def plot_roll_solution_debug(
    info,
    scan_points=121,
    filename=None,
    show=True,
):
    """
    Plot trajectory and solver diagnostics.

    Parameters
    ----------
    info : dict
        Information returned by desired_bank_to_point_with_wind_roll().

    scan_points : int
        Number of bank commands used for the error scan.

    filename : str or None
        Save the figure if a filename is supplied.

    show : bool
        Call plt.show() if True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    if "trajectory" not in info:
        raise ValueError(
            "info does not contain trajectory data. "
            "Use desired_bank_to_point_with_wind_roll()."
        )

    context = info["debug_context"]

    p = context["params"]
    xt = context["xt"]
    yt = context["yt"]
    t_max = context["t_max"]
    dt = context["dt"]
    phi_max = context["phi_max"]
    tol_pos = context["tol_pos"]

    selected = info["trajectory"]
    closest = selected["closest"]

    t = selected["t"]
    x = selected["x"]
    y = selected["y"]
    psi = selected["psi"]
    phi = selected["phi"]

    phi_solution = info["phi_solution"]

    # Comparison trajectories
    comparison_commands = [
        ("Wings level", 0.0, "0.45", "--"),
        ("Left limit", -phi_max, "tab:orange", ":"),
        ("Right limit", phi_max, "tab:green", ":"),
    ]

    comparison_trajectories = []

    for name, phi_cmd, color, linestyle in comparison_commands:
        if abs(phi_cmd - phi_solution) < 1e-10:
            continue

        _, _, _, details = closest_approach_error_with_roll(
            phi_cmd=phi_cmd,
            p=p,
            xt=xt,
            yt=yt,
            t_max=t_max,
            dt=dt,
            return_details=True,
        )

        comparison_trajectories.append(
            (name, phi_cmd, color, linestyle, details)
        )

    scan = generate_bank_error_scan(
        info,
        scan_points=scan_points,
    )

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)

    ax_trajectory = fig.add_subplot(grid[:, 0])
    ax_bank = fig.add_subplot(grid[0, 1])
    ax_distance = fig.add_subplot(grid[1, 1])
    ax_error = fig.add_subplot(grid[0, 2])
    ax_tmin = fig.add_subplot(grid[1, 2])

    # --------------------------------------------------------
    # Trajectory plot
    # --------------------------------------------------------
    for name, phi_cmd, color, linestyle, details in comparison_trajectories:
        ax_trajectory.plot(
            details["x"],
            details["y"],
            color=color,
            linestyle=linestyle,
            linewidth=1.2,
            alpha=0.75,
            label=f"{name}: {np.rad2deg(phi_cmd):.1f}°",
        )

    ax_trajectory.plot(
        x,
        y,
        color="tab:blue",
        linewidth=2.5,
        label=(
            f"Selected: {np.rad2deg(phi_solution):.2f}°"
        ),
    )

    ax_trajectory.scatter(
        p.x0,
        p.y0,
        s=80,
        color="black",
        marker="o",
        zorder=6,
        label="Initial position",
    )

    ax_trajectory.scatter(
        xt,
        yt,
        s=180,
        color="red",
        marker="*",
        edgecolor="black",
        linewidth=0.8,
        zorder=8,
        label="Target",
    )

    closest_state = closest["state"]

    ax_trajectory.scatter(
        closest_state["x"],
        closest_state["y"],
        s=90,
        color="cyan",
        marker="o",
        edgecolor="black",
        linewidth=0.8,
        zorder=8,
        label="Selected closest approach",
    )

    ax_trajectory.plot(
        [closest_state["x"], xt],
        [closest_state["y"], yt],
        color="red",
        linewidth=1.5,
        linestyle="--",
        label=f"Miss distance: {closest['d_min']:.2f} m",
    )

    # Initial heading arrow
    trajectory_scale = max(
        np.ptp(np.concatenate([x, np.array([xt])])),
        np.ptp(np.concatenate([y, np.array([yt])])),
        1.0,
    )

    heading_arrow_length = 0.10 * trajectory_scale

    heading_dx = heading_arrow_length * np.sin(p.psi0)
    heading_dy = heading_arrow_length * np.cos(p.psi0)

    ax_trajectory.arrow(
        p.x0,
        p.y0,
        heading_dx,
        heading_dy,
        width=0.004 * trajectory_scale,
        head_width=0.025 * trajectory_scale,
        head_length=0.035 * trajectory_scale,
        color="black",
        length_includes_head=True,
        zorder=7,
    )

    # Wind arrow: displacement caused by five seconds of wind
    wind_arrow_time = 5.0
    wind_dx = p.Vw_x * wind_arrow_time
    wind_dy = p.Vw_y * wind_arrow_time

    if np.hypot(wind_dx, wind_dy) > 1e-9:
        ax_trajectory.arrow(
            p.x0,
            p.y0,
            wind_dx,
            wind_dy,
            width=0.003 * trajectory_scale,
            head_width=0.020 * trajectory_scale,
            head_length=0.030 * trajectory_scale,
            color="magenta",
            length_includes_head=True,
            zorder=7,
        )

        ax_trajectory.text(
            p.x0 + wind_dx,
            p.y0 + wind_dy,
            "  wind × 5 s",
            color="magenta",
            fontsize=9,
        )

    ax_trajectory.set_xlabel("East, x [m]")
    ax_trajectory.set_ylabel("North, y [m]")
    ax_trajectory.set_title("Ground trajectories")
    ax_trajectory.axis("equal")
    ax_trajectory.grid(True, alpha=0.3)
    ax_trajectory.legend(fontsize=8, loc="best")

    # --------------------------------------------------------
    # Bank and heading
    # --------------------------------------------------------
    ax_bank.plot(
        t,
        np.rad2deg(phi),
        color="tab:blue",
        linewidth=2.0,
        label="Bank",
    )

    ax_bank.axhline(
        np.rad2deg(phi_solution),
        color="tab:blue",
        linestyle="--",
        alpha=0.6,
        label="Commanded bank",
    )

    ax_bank.axvline(
        closest["t_min"],
        color="red",
        linestyle=":",
        label="Closest approach",
    )

    ax_bank.set_xlabel("Time [s]")
    ax_bank.set_ylabel("Bank [deg]", color="tab:blue")
    ax_bank.tick_params(axis="y", labelcolor="tab:blue")
    ax_bank.grid(True, alpha=0.3)

    ax_heading = ax_bank.twinx()

    ax_heading.plot(
        t,
        np.rad2deg(np.unwrap(psi)),
        color="tab:orange",
        linewidth=1.5,
        alpha=0.85,
        label="Heading",
    )

    ax_heading.set_ylabel(
        "Unwrapped heading [deg]",
        color="tab:orange",
    )
    ax_heading.tick_params(axis="y", labelcolor="tab:orange")

    ax_bank.set_title("Bank and heading")

    bank_handles, bank_labels = ax_bank.get_legend_handles_labels()
    heading_handles, heading_labels = ax_heading.get_legend_handles_labels()

    ax_bank.legend(
        bank_handles + heading_handles,
        bank_labels + heading_labels,
        fontsize=8,
        loc="best",
    )

    # --------------------------------------------------------
    # Distance versus time
    # --------------------------------------------------------
    distance = closest["distance"]

    ax_distance.plot(
        t,
        distance,
        color="tab:purple",
        linewidth=2.0,
        label="Distance to target",
    )

    local_indices = closest["local_min_indices"]

    if local_indices.size > 0:
        ax_distance.scatter(
            t[local_indices],
            distance[local_indices],
            color="tab:orange",
            marker="x",
            s=45,
            label="Detected local minima",
            zorder=6,
        )

    ax_distance.scatter(
        closest["t_min"],
        closest["d_min"],
        color="red",
        s=70,
        zorder=7,
        label="Selected minimum",
    )

    ax_distance.axvline(
        closest["t_min"],
        color="red",
        linestyle=":",
    )

    ax_distance.axhline(
        tol_pos,
        color="green",
        linestyle="--",
        alpha=0.7,
        label="Position tolerance",
    )

    ax_distance.set_xlabel("Time [s]")
    ax_distance.set_ylabel("Distance to target [m]")
    ax_distance.set_title(
        "Target distance versus time\n"
        f"Selected minimum: {closest['minimum_location']}"
    )
    ax_distance.grid(True, alpha=0.3)
    ax_distance.legend(fontsize=8, loc="best")

    # --------------------------------------------------------
    # Signed error versus commanded bank
    # --------------------------------------------------------
    phi_scan_deg = np.rad2deg(scan["phi"])

    ax_error.plot(
        phi_scan_deg,
        scan["err"],
        color="black",
        linewidth=1.5,
        label="Signed closest-approach error",
    )

    ax_error.axhline(
        0.0,
        color="black",
        linewidth=1.0,
    )

    ax_error.axhspan(
        -tol_pos,
        tol_pos,
        color="green",
        alpha=0.12,
        label="Position tolerance",
    )

    history_phi = np.array(
        [item["phi"] for item in info["history"]],
        dtype=float,
    )

    history_err = np.array(
        [item["err"] for item in info["history"]],
        dtype=float,
    )

    ax_error.scatter(
        np.rad2deg(history_phi),
        history_err,
        color="tab:orange",
        edgecolor="black",
        linewidth=0.4,
        s=40,
        zorder=7,
        label="Solver evaluations",
    )

    ax_error.scatter(
        np.rad2deg(phi_solution),
        info["err"],
        color="red",
        marker="*",
        edgecolor="black",
        linewidth=0.7,
        s=170,
        zorder=8,
        label="Selected command",
    )

    boundary_indices = np.nonzero(scan["boundary"])[0]

    if boundary_indices.size > 0:
        ax_error.scatter(
            phi_scan_deg[boundary_indices],
            scan["err"][boundary_indices],
            facecolors="none",
            edgecolors="magenta",
            linewidth=1.0,
            s=50,
            label="Minimum at time boundary",
        )

    ax_error.set_xlabel("Commanded bank [deg]")
    ax_error.set_ylabel("Signed error [m]")
    ax_error.set_title("Bank-command error function")
    ax_error.grid(True, alpha=0.3)
    ax_error.legend(fontsize=8, loc="best")

    # --------------------------------------------------------
    # Closest-approach time versus bank
    # --------------------------------------------------------
    ax_tmin.plot(
        phi_scan_deg,
        scan["t_min"],
        color="tab:green",
        linewidth=1.5,
        label="Closest-approach time",
    )

    ax_tmin.axhline(
        t_max,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="t_max",
    )

    ax_tmin.scatter(
        np.rad2deg(phi_solution),
        info["t_min"],
        color="red",
        marker="*",
        edgecolor="black",
        linewidth=0.7,
        s=150,
        zorder=8,
        label="Selected command",
    )

    if boundary_indices.size > 0:
        ax_tmin.scatter(
            phi_scan_deg[boundary_indices],
            scan["t_min"][boundary_indices],
            facecolors="none",
            edgecolors="magenta",
            linewidth=1.0,
            s=50,
            label="Boundary minimum",
        )

    ax_tmin.set_xlabel("Commanded bank [deg]")
    ax_tmin.set_ylabel("Closest-approach time [s]")
    ax_tmin.set_title("Selected closest-approach time")
    ax_tmin.grid(True, alpha=0.3)
    ax_tmin.legend(fontsize=8, loc="best")

    # --------------------------------------------------------
    # Figure title
    # --------------------------------------------------------
    if info["converged"]:
        status = "CONVERGED"
    else:
        status = f"NOT CONVERGED: {info.get('reason', 'unknown')}"

    title = (
        f"Wind-aware bank solver debug — {status}\n"
        f"phi_cmd={np.rad2deg(phi_solution):.3f} deg, "
        f"signed error={info['err']:.3f} m, "
        f"distance={info['d_min']:.3f} m, "
        f"t_min={info['t_min']:.3f} s"
    )

    if "warning" in info:
        title += f"\nWarning: {info['warning']}"

    fig.suptitle(title, fontsize=13)

    if filename is not None:
        fig.savefig(filename, dpi=160, bbox_inches="tight")

    if show:
        plt.show()

    return fig


# ============================================================
# 9. Example
# ============================================================

if __name__ == "__main__":
    # Initial aircraft state
    x0 = 697075.2887876498
    y0 = 6084791.098685207
    psi0 = 3.8823103881361867 #np.deg2rad(0.0)       # North
    phi0 = 0.6994298696517944 #np.deg2rad(0.0)

    # Target
    xt = 697037.7793601836
    yt = 6084771.4720707415

    # Aircraft/environment
    g = 9.80665
    V_TAS = 17.5 * 1.069

    # Wind from West, therefore blowing toward East
    V_w = 8.75
    theta_wa = np.deg2rad(30.0)

    phi_cmd, info = desired_bank_to_point_with_wind_roll(
        x0=x0,
        y0=y0,
        psi0=psi0,
        phi0=phi0,
        xt=xt,
        yt=yt,
        g=g,
        V_TAS=V_TAS,
        V_w=V_w,
        theta_wa=theta_wa,
        p_max_roll=np.deg2rad(15.0),
        t_max=5.0,
        dt=0.005,
        phi_max=np.deg2rad(40.0),
        tol_pos=0.1,
        max_iter=40,

        # Plot every call:
        debug_plot=True,

        # Alternatively, set debug_plot=False and this to True:
        plot_on_failure=True,

        plot_scan_points=121,
        plot_filename=None,
        show_plot=True,
    )

    print()
    print("Solver result")
    print("-------------")
    print(f"Bank command : {np.rad2deg(phi_cmd):.6f} deg")
    print(f"Converged    : {info['converged']}")
    print(f"Signed error : {info['err']:.6f} m")
    print(f"Miss distance: {info['d_min']:.6f} m")
    print(f"Closest time : {info['t_min']:.6f} s")

    if "reason" in info:
        print(f"Failure reason: {info['reason']}")

    if "warning" in info:
        print(f"Warning       : {info['warning']}")
