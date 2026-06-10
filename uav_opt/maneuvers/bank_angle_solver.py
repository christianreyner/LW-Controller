#!/usr/bin/env python3
"""
Wind-aware bank angle solver + example with plotting.

This script defines a function `desired_bank_to_point_with_wind(...)` that
computes a steady bank angle (no roll dynamics) which, in the presence of
constant wind, makes the aircraft's *ground track* pass as close as possible
to a target point.

Conventions:
- x axis: East [m]
- y axis: North [m]
- psi: heading [rad], clockwise from North
- Wind input: speed V_w [m/s] and "from" direction theta_wa [rad],
  same convention as psi (i.e., direction FROM which the wind blows).
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ============================================================
# 1. Data structure for turn parameters
# ============================================================

@dataclass
class TurnWithWindParams:
    """
    Parameters describing the initial state and environment for a
    steady, coordinated, level turn with wind.
    """
    x0: float       # initial x position [m] (East)
    y0: float       # initial y position [m] (North)
    psi0: float     # initial heading [rad], clockwise from North
    g: float        # gravity [m/s^2]
    V_TAS: float    # true airspeed [m/s]
    Vw_x: float     # wind x-component [m/s] (East)
    Vw_y: float     # wind y-component [m/s] (North)
      
@dataclass
class TurnWithWindRollParams:
    """
    Parameters for a coordinated turn with wind and simple roll dynamics.
    """
    x0: float       # initial x position [m] (East)
    y0: float       # initial y position [m] (North)
    psi0: float     # initial heading [rad], clockwise from North
    phi0: float     # initial bank angle [rad]
    g: float        # gravity [m/s^2]
    V_TAS: float    # true airspeed [m/s]
    Vw_x: float     # wind x-component [m/s] (East)
    Vw_y: float     # wind y-component [m/s] (North)
    p_max: float    # max roll rate magnitude [rad/s] (for roll-in)


# ============================================================
# 2. Analytic ground track with wind and constant bank
# ============================================================

def position_with_wind_const_bank(t, phi, p: TurnWithWindParams):
    """
    Analytic (x(t), y(t)) for a coordinated, level, constant-bank turn
    at constant TAS, in constant wind.

    Model:
      - Heading dynamics:   dψ/dt = (g / V_TAS) * tan(phi)
      - Air-relative speed: v_air = V_TAS [sin(ψ), cos(ψ)]
      - Ground speed:       v_g   = v_air + v_wind

    Parameters
    ----------
    t : float or array-like
        Time(s) [s] at which to evaluate the position.
    phi : float
        Bank angle [rad]; sign >0 for right turn, <0 for left.
    p : TurnWithWindParams
        Turn and wind parameters.

    Returns
    -------
    x, y : ndarray
        Positions at time(s) t [m].
    """
    t = np.asarray(t, dtype=float)
    x0, y0, psi0 = p.x0, p.y0, p.psi0
    g, V = p.g, p.V_TAS
    Vw_x, Vw_y = p.Vw_x, p.Vw_y

    # Handle "almost zero" bank (straight line) explicitly
    if abs(phi) < 1e-9:
        psi = psi0
        vx = V * np.sin(psi) + Vw_x
        vy = V * np.cos(psi) + Vw_y
        x = x0 + vx * t
        y = y0 + vy * t
        return x, y

    k = g / V
    Omega = k * np.tan(phi)  # signed heading rate [rad/s]

    # If Omega is extremely small, treat as straight to avoid numerical issues
    if abs(Omega) < 1e-9:
        psi = psi0
        vx = V * np.sin(psi) + Vw_x
        vy = V * np.cos(psi) + Vw_y
        x = x0 + vx * t
        y = y0 + vy * t
        return x, y

    psi_t = psi0 + Omega * t

    # Analytic integration (air-relative contribution)
    # ∫ sin(ψ0 + Ω τ) dτ = [-cos(ψ0 + Ω τ)] / Ω
    # ∫ cos(ψ0 + Ω τ) dτ = [ sin(ψ0 + Ω τ)] / Ω
    x = (
        x0
        + V * (-(np.cos(psi_t) - np.cos(psi0)) / Omega)
        + Vw_x * t
    )
    y = (
        y0
        + V * ((np.sin(psi_t) - np.sin(psi0)) / Omega)
        + Vw_y * t
    )

    return x, y


def heading_at_time(phi, t, p: TurnWithWindParams):
    """
    Heading ψ(t) for a constant-bank turn.

    Parameters
    ----------
    phi : float
        Bank angle [rad].
    t : float or array-like
        Time(s) [s].
    p : TurnWithWindParams
        Turn parameters.

    Returns
    -------
    psi : float or ndarray
        Heading(s) [rad], clockwise from North.
    """
    t = np.asarray(t, dtype=float)
    if abs(phi) < 1e-9:
        return np.full_like(t, p.psi0)

    k = p.g / p.V_TAS
    Omega = k * np.tan(phi)
    return p.psi0 + Omega * t

def trajectory_with_wind_roll(
    phi_cmd,
    p: TurnWithWindRollParams,
    t_final,
    dt=0.05,
):
    """
    Vectorized numerical trajectory with wind and simple roll dynamics
    (roll-in only, no roll-out), using fixed step Euler integration.

    Bank dynamics:
        dphi/dt = ±p_max until phi reaches phi_cmd, then holds.

    Heading:
        dpsi/dt = (g / V_TAS) * tan(phi)

    Position:
        v_air = V_TAS [sin(psi), cos(psi)]
        v_g   = v_air + v_wind
        x,y integrated with Euler, using v_g(t_{k+1}) as in the loop version.
    """
    # Time grid
    n_steps = max(2, int(np.ceil(t_final / dt)) + 1)
    t = dt * np.arange(n_steps)   # [0, dt, 2dt, ...]
    # Effective dt (constant)
    dt_eff = dt

    # --------------------------
    # Bank profile φ(t): ramp with saturation at phi_cmd
    # --------------------------
    dphi_total = phi_cmd - p.phi0
    phi = np.empty(n_steps)

    if abs(dphi_total) < 1e-12 or p.p_max <= 0.0:
        # No roll or no roll authority: constant bank
        phi[:] = p.phi0
    else:
        sgn = np.sign(dphi_total)
        # Ideal ramp
        phi_ramp = p.phi0 + sgn * p.p_max * t
        # Saturate at phi_cmd
        if sgn > 0:
            phi = np.minimum(phi_ramp, phi_cmd)
        else:
            phi = np.maximum(phi_ramp, phi_cmd)

    # --------------------------
    # Heading ψ(t)
    # --------------------------
    psi = np.empty(n_steps)
    psi[0] = p.psi0

    g_over_V = p.g / p.V_TAS

    # Use φ at t_{k+1} like in the loop version: ψ_{k+1} depends on φ_{k+1}
    phi_for_psi = phi[1:].copy()
    # Optional threshold to mimic your if |phi| < 1e-9: psi_dot = 0
    phi_for_psi[np.abs(phi_for_psi) < 1e-9] = 0.0

    psi_dot = g_over_V * np.tan(phi_for_psi)    # size n_steps-1
    psi[1:] = p.psi0 + np.cumsum(psi_dot * dt_eff)

    # --------------------------
    # Velocities and positions
    # --------------------------
    V = p.V_TAS
    vx_air = V * np.sin(psi)
    vy_air = V * np.cos(psi)

    vx_g = vx_air + p.Vw_x
    vy_g = vy_air + p.Vw_y

    x = np.empty(n_steps)
    y = np.empty(n_steps)
    x[0] = p.x0
    y[0] = p.y0

    # Euler: x_{k+1} = x_k + v_g(t_{k+1}) * dt
    x[1:] = p.x0 + np.cumsum(vx_g[1:] * dt_eff)
    y[1:] = p.y0 + np.cumsum(vy_g[1:] * dt_eff)

    return t, x, y, psi, phi

# ============================================================
# 3. Closest-approach error for a given bank
# ============================================================

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
    For a given bank φ, compute the signed lateral error at the *earliest*
    local closest-approach point (within at most `max_turns` revolutions).

    Parameters
    ----------
    phi : float
        Bank angle [rad].
    p : TurnWithWindParams
        Turn and wind parameters.
    xt, yt : float
        Target coordinates [m].
    t_max : float, optional
        Hard upper bound on time horizon [s].
    n_grid : int, optional
        Number of grid points for the coarse time search.
    max_turns : float or None, optional
        Maximum number of heading revolutions considered.
        - If None: use full [0, t_max].
        - If >0: limit to min(t_max, max_turns * 2π/|Ω|).

    Returns
    -------
    err : float
        Signed cross-track error [m] at closest approach.
        > 0 : target is to the LEFT of velocity vector
        < 0 : target is to the RIGHT.
    t_min : float
        Time of (earliest) closest approach [s].
    d_min : float
        Scalar distance [m] at that point.
    """
    g, V = p.g, p.V_TAS

    # Heading rate
    if abs(phi) < 1e-9:
        Omega = 0.0
    else:
        Omega = (g / V) * np.tan(phi)

    # Adaptive time horizon: at most `max_turns` heading revolutions
    if max_turns is not None and max_turns > 0.0 and abs(Omega) > 1e-6:
        T_one_turn = 2.0 * np.pi / abs(Omega)
        t_end = min(t_max, max_turns * T_one_turn)
    else:
        t_end = t_max

    # Coarse grid in time
    t_grid = np.linspace(0.0, t_end, n_grid)
    xg, yg = position_with_wind_const_bank(t_grid, phi, p)
    dx = xg - xt
    dy = yg - yt
    d2 = dx**2 + dy**2

    # Find indices of all *local* minima in d^2
    # d2[i] <= d2[i-1] and d2[i] <= d2[i+1]
    if n_grid >= 3:
        is_min = (d2[1:-1] <= d2[:-2]) & (d2[1:-1] <= d2[2:])
        idx_candidates = np.nonzero(is_min)[0] + 1
    else:
        idx_candidates = np.array([], dtype=int)

    if idx_candidates.size == 0:
        # Fallback: no local minima detected, use global discrete min
        i_min = int(np.argmin(d2))
    else:
        # Choose the earliest local minimum in time
        i_min = int(idx_candidates[0])

    # Parabolic refinement around selected minimum (if interior)
    if 0 < i_min < n_grid - 1:
        tm1, t0, tp1 = t_grid[i_min - 1 : i_min + 2]
        d2m1, d20, d2p1 = d2[i_min - 1 : i_min + 2]

        denom = (tm1 - t0) * (tm1 - tp1) * (t0 - tp1)
        if abs(denom) > 1e-14:
            a = (
                d2m1 * (t0 - tp1)
                + d20 * (tp1 - tm1)
                + d2p1 * (tm1 - t0)
            ) / denom

            b = (
                d2m1 * (tp1**2 - t0**2)
                + d20 * (tm1**2 - tp1**2)
                + d2p1 * (t0**2 - tm1**2)
            ) / denom

            if a > 0:
                t_vertex = -b / (2 * a)
                if tm1 <= t_vertex <= tp1:
                    t_min = float(t_vertex)
                else:
                    t_min = float(t0)
            else:
                t_min = float(t0)
        else:
            t_min = float(t0)
    else:
        t_min = float(t_grid[i_min])

    # Evaluate position and distance at refined t_min
    x_min, y_min = position_with_wind_const_bank(t_min, phi, p)
    dx = x_min - xt
    dy = y_min - yt
    d_min = float(np.hypot(dx, dy))

    # Ground velocity at t_min
    if abs(phi) < 1e-9:
        psi = p.psi0
    else:
        psi = p.psi0 + Omega * t_min

    vx = V * np.sin(psi) + p.Vw_x
    vy = V * np.cos(psi) + p.Vw_y

    # Vector from aircraft to target
    rx = xt - x_min
    ry = yt - y_min

    # 2D cross product v × r (z-component)
    cross_z = vx * ry - vy * rx

    err = float(np.sign(cross_z) * d_min)
    return err, float(t_min), d_min

def closest_approach_error_with_roll(
    phi_cmd,
    p: TurnWithWindRollParams,
    xt,
    yt,
    t_max=200.0,
    dt=0.05,
):
    """
    Compute signed cross-track error at earliest local closest-approach point
    for a given commanded bank phi_cmd, using roll dynamics.

    Parameters
    ----------
    phi_cmd : float
        Commanded steady bank [rad] after roll-in.
    p : TurnWithWindRollParams
        Turn and wind parameters including initial bank & roll-rate limit.
    xt, yt : float
        Target coordinates [m].
    t_max : float, optional
        End of simulation horizon [s].
    dt : float, optional
        Integration step [s].

    Returns
    -------
    err : float
        Signed cross-track error [m] at closest approach.
    t_min : float
        Time of earliest closest approach [s].
    d_min : float
        Minimum distance [m].
    """
    # Simulate trajectory with roll-in
    t, x, y, psi, phi = trajectory_with_wind_roll(phi_cmd, p, t_final=t_max, dt=dt)

    # Distances to target
    dx = x - xt
    dy = y - yt
    d2 = dx**2 + dy**2

    n = len(t)
    if n < 3:
        i_min = int(np.argmin(d2))
        t_min = float(t[i_min])
        d_min = float(np.sqrt(d2[i_min]))
    else:
        # local minima indices
        is_min = (d2[1:-1] <= d2[:-2]) & (d2[1:-1] <= d2[2:])
        idx_candidates = np.nonzero(is_min)[0] + 1

        if idx_candidates.size == 0:
            i_min = int(np.argmin(d2))
            t_min = float(t[i_min])
        else:
            i_min = int(idx_candidates[0])   # earliest local minimum

            # Optional parabolic refinement
            tm1, t0, tp1 = t[i_min - 1 : i_min + 2]
            d2m1, d20, d2p1 = d2[i_min - 1 : i_min + 2]
            denom = (tm1 - t0) * (tm1 - tp1) * (t0 - tp1)
            if abs(denom) > 1e-14:
                a = (
                    d2m1 * (t0 - tp1)
                    + d20 * (tp1 - tm1)
                    + d2p1 * (tm1 - t0)
                ) / denom

                b = (
                    d2m1 * (tp1**2 - t0**2)
                    + d20 * (tm1**2 - tp1**2)
                    + d2p1 * (t0**2 - tm1**2)
                ) / denom

                if a > 0:
                    t_vertex = -b / (2 * a)
                    if tm1 <= t_vertex <= tp1:
                        t_min = float(t_vertex)
                    else:
                        t_min = float(t0)
                else:
                    t_min = float(t0)
            else:
                t_min = float(t0)

        # recompute distance at refined time
        # small one-step interpolation: find closest index, no need re-simulate
        i_closest = int(np.argmin(np.abs(t - t_min)))
        d_min = float(np.sqrt(d2[i_closest]))

    # Ground velocity at t_min (interpolated from stored psi, phi)
    i_closest = int(np.argmin(np.abs(t - t_min)))
    psi_t = psi[i_closest]

    V = p.V_TAS
    vx = V * np.sin(psi_t) + p.Vw_x
    vy = V * np.cos(psi_t) + p.Vw_y

    # Position at t_min
    x_t = x[i_closest]
    y_t = y[i_closest]

    # Vector from aircraft to target
    rx = xt - x_t
    ry = yt - y_t

    # Cross product z component: v × r
    cross_z = vx * ry - vy * rx
    err = float(np.sign(cross_z) * d_min)

    return err, float(t_min), d_min

# ============================================================
# 4. Bank angle solver with wind
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
    Compute steady bank angle φ (no roll dynamics) that, in constant wind,
    makes the aircraft's ground track pass as close as possible to (xt, yt).

    Error function:
      - For each φ, find time of closest approach to target (over [0, t_max]).
      - Return signed cross-track error at that point.
    Then use a bracketed 1D root-finder (bisection) on φ.

    Parameters
    ----------
    x0, y0 : float
        Initial position [m].
    psi0 : float
        Initial heading [rad], clockwise from North.
    xt, yt : float
        Target coordinates [m].
    g : float
        Gravity [m/s^2].
    V_TAS : float
        True airspeed [m/s].
    V_w : float
        Wind speed [m/s].
    theta_wa : float
        Wind-from direction [rad], clockwise from North.
        (E.g. 270° means wind from West, blowing East.)
    t_max : float, optional
        Time horizon [s] for closest-approach search.
    phi_max : float, optional
        Maximum absolute bank angle [rad] used as a search bound.
    tol_pos : float, optional
        Acceptable lateral error [m] for convergence.
    max_iter : int, optional
        Maximum number of bisection iterations.

    Returns
    -------
    phi_cmd : float
        Desired bank angle [rad]; >0 right turn, <0 left turn.
    info : dict
        Diagnostic information:
        - 'phi_solution': final φ [rad]
        - 'err': final signed cross-track error [m]
        - 't_min': time of closest approach [s]
        - 'd_min': distance at closest approach [m]
        - 'bracket': final bracket (φ_left, φ_right) [rad]
        - 'converged': bool
        - optional 'reason': string if not converged (e.g., 'no_sign_change')
    """
    # Wind components from "wind from" direction
    # theta_wa is the direction FROM which the wind blows, so the
    # actual wind vector points at theta_wa + π.
    Vw_x = V_w * np.sin(theta_wa + np.pi)
    Vw_y = V_w * np.cos(theta_wa + np.pi)

    # Package parameters
    p = TurnWithWindParams(
        x0=float(x0),
        y0=float(y0),
        psi0=float(psi0),
        g=float(g),
        V_TAS=float(V_TAS),
        Vw_x=float(Vw_x),
        Vw_y=float(Vw_y),
    )

    # 1) Check wings-level solution
    err0, t0_min, d0_min = closest_approach_error(0.0, p, xt, yt, t_max=t_max, max_turns=1.0)
    if abs(err0) <= tol_pos:
        info = dict(
            phi_solution=0.0,
            err=err0,
            t_min=t0_min,
            d_min=d0_min,
            bracket=(0.0, 0.0),
            converged=True,
            note="wings_level_sufficient",
        )
        return 0.0, info

    # 2) Set up search bracket for bank angle
    phi_L = -phi_max
    phi_R = +phi_max

    err_L, tL, dL = closest_approach_error(phi_L, p, xt, yt, t_max=t_max, max_turns=1.0)
    err_R, tR, dR = closest_approach_error(phi_R, p, xt, yt, t_max=t_max, max_turns=1.0)

    # 3) Check if error changes sign in [phi_L, phi_R]
    if err_L * err_R > 0:
        # No sign change: within the allowed bank range, the target
        # is always on same side of the track. Choose best of ends.
        if abs(err_L) < abs(err_R):
            phi_sol = phi_L
            err_sol, t_sol, d_sol = err_L, tL, dL
        else:
            phi_sol = phi_R
            err_sol, t_sol, d_sol = err_R, tR, dR

        info = dict(
            phi_solution=phi_sol,
            err=err_sol,
            t_min=t_sol,
            d_min=d_sol,
            bracket=(phi_L, phi_R),
            converged=False,
            reason="no_sign_change",
        )
        return phi_sol, info

    # 4) Bisection within [phi_L, phi_R]
    a, b = phi_L, phi_R
    fa, fb = err_L, err_R

    phi_sol = None
    err_sol = None
    t_sol = None
    d_sol = None
    converged = False

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc, tc, dc = closest_approach_error(c, p, xt, yt, t_max=t_max, max_turns=1.0)

        phi_sol, err_sol, t_sol, d_sol = c, fc, tc, dc

        if abs(fc) <= tol_pos:
            converged = True
            break

        # Maintain sign change in the bracket
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    info = dict(
        phi_solution=phi_sol,
        err=err_sol,
        t_min=t_sol,
        d_min=d_sol,
        bracket=(a, b),
        converged=converged,
    )
    return phi_sol, info

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
    p_max_roll=np.deg2rad(15.0),   # e.g. 15 deg/s
    t_max=200.0,
    dt=0.05,
    phi_max=np.deg2rad(60.0),
    tol_pos=1.0,
    max_iter=40,
):
    """
    Compute commanded steady bank phi_cmd that, with roll dynamics (no roll-out)
    and constant wind, makes the ground track pass as close as possible to (xt, yt).

    Bank dynamics:
      - Start from phi0 at t=0.
      - Roll with ±p_max_roll until phi reaches phi_cmd.
      - Then hold phi = phi_cmd (no roll-out).

    Parameters
    ----------
    x0, y0, psi0, phi0 : float
        Initial position [m], heading [rad], and bank [rad].
    xt, yt : float
        Target coordinates [m].
    g, V_TAS : float
        Gravity [m/s^2], true airspeed [m/s].
    V_w : float
        Wind speed [m/s].
    theta_wa : float
        Wind-from direction [rad], clockwise from North.
    p_max_roll : float, optional
        Max roll rate magnitude [rad/s].
    t_max : float, optional
        Time horizon [s].
    dt : float, optional
        Integration time step [s].
    phi_max : float, optional
        Max |phi_cmd| considered [rad].
    tol_pos : float, optional
        Lateral error tolerance [m].
    max_iter : int, optional
        Bisection iterations.

    Returns
    -------
    phi_cmd : float
        Commanded bank angle [rad].
    info : dict
        Diagnostics, analogous to the constant-bank version.
    """
    # Wind components
    Vw_x = V_w * np.sin(theta_wa + np.pi)
    Vw_y = V_w * np.cos(theta_wa + np.pi)

    p = TurnWithWindRollParams(
        x0=float(x0),
        y0=float(y0),
        psi0=float(psi0),
        phi0=float(phi0),
        g=float(g),
        V_TAS=float(V_TAS),
        Vw_x=float(Vw_x),
        Vw_y=float(Vw_y),
        p_max=float(p_max_roll),
    )

    # 1) Check wings-level command: phi_cmd = 0
    err0, t0_min, d0_min = closest_approach_error_with_roll(
        0.0, p, xt, yt, t_max=t_max, dt=dt
    )
    if abs(err0) <= tol_pos:
        info = dict(
            phi_solution=0.0,
            err=err0,
            t_min=t0_min,
            d_min=d0_min,
            bracket=(0.0, 0.0),
            converged=True,
            note="wings_level_sufficient",
        )
        return 0.0, info

    # 2) Bracket
    phi_L = -phi_max
    phi_R = +phi_max

    err_L, tL, dL = closest_approach_error_with_roll(
        phi_L, p, xt, yt, t_max=t_max, dt=dt
    )
    err_R, tR, dR = closest_approach_error_with_roll(
        phi_R, p, xt, yt, t_max=t_max, dt=dt
    )

    if err_L * err_R > 0:
        # No sign change: pick end with smaller |err|
        if abs(err_L) < abs(err_R):
            phi_sol, err_sol, t_sol, d_sol = phi_L, err_L, tL, dL
        else:
            phi_sol, err_sol, t_sol, d_sol = phi_R, err_R, tR, dR

        info = dict(
            phi_solution=phi_sol,
            err=err_sol,
            t_min=t_sol,
            d_min=d_sol,
            bracket=(phi_L, phi_R),
            converged=False,
            reason="no_sign_change",
        )
        return phi_sol, info

    # 3) Bisection
    a, b = phi_L, phi_R
    fa, fb = err_L, err_R

    phi_sol = None
    err_sol = None
    t_sol = None
    d_sol = None
    converged = False

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc, tc, dc = closest_approach_error_with_roll(
            c, p, xt, yt, t_max=t_max, dt=dt
        )

        phi_sol, err_sol, t_sol, d_sol = c, fc, tc, dc

        if abs(fc) <= tol_pos:
            converged = True
            break

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    info = dict(
        phi_solution=phi_sol,
        err=err_sol,
        t_min=t_sol,
        d_min=d_sol,
        bracket=(a, b),
        converged=converged,
    )
    return phi_sol, info

# ============================================================
# 5. Example + plotting (stops at target hit / closest approach)
# ============================================================

def run_example(constants, start, end, psi0):
    # -------------------------
    # Scenario setup
    # -------------------------
    g, V_TAS, bank_angle, V_w, theta_wa, dt = constants

    # Initial state
    x0, y0 = start

    # Target position [m]
    xt, yt = end

    # Solver parameters
    t_search = 300.0       # [s] search horizon for closest approach
    phi_max = np.deg2rad(89.0)     # max bank for search
    tol_pos = 0.5         # [m] lateral error tolerance ("hit" radius)

    # -------------------------
    # Solve for wind-aware bank
    # -------------------------
    phi_sol, info = desired_bank_to_point_with_wind(
        x0=x0,
        y0=y0,
        psi0=psi0,
        xt=xt,
        yt=yt,
        g=g,
        V_TAS=V_TAS,
        V_w=V_w,
        theta_wa=theta_wa,
        t_max=t_search,
        phi_max=phi_max,
        tol_pos=tol_pos,
        max_iter=40,
    )

    phi_deg = np.rad2deg(phi_sol)

    print("=== Wind-aware bank angle solution ===")
    print(f"Converged:         {info['converged']}")
    print(f"Bank angle φ:      {phi_deg:.2f} deg")
    print(f"Closest distance:  {info['d_min']:.1f} m")
    print(f"Signed error:      {info['err']:.1f} m")
    print(f"Time of c.a.:      {info['t_min']:.1f} s")
    print(f"Bracket [deg]:     "
          f"[{np.rad2deg(info['bracket'][0]):.1f}, "
          f"{np.rad2deg(info['bracket'][1]):.1f}]")

    # -------------------------
    # Build parameters object for trajectory plotting
    # -------------------------
    Vw_x = V_w * np.sin(theta_wa + np.pi)
    Vw_y = V_w * np.cos(theta_wa + np.pi)

    p = TurnWithWindParams(
        x0=x0,
        y0=y0,
        psi0=psi0,
        g=g,
        V_TAS=V_TAS,
        Vw_x=Vw_x,
        Vw_y=Vw_y,
    )

    # -------------------------
    # Trajectory only up to target hit / closest approach
    # -------------------------
    # We stop at the time of closest approach given by the solver.
    # This is effectively the "hit time" within tol_pos.
    t_hit = min(info["t_min"], t_search)
    # Choose a reasonable number of points; scales with t_hit
    n_pts = max(200, int(5 * t_hit))
    t_traj = np.linspace(0.0, t_hit, n_pts)

    # (1) Optimal bank with wind, truncated at t_hit
    x_opt, y_opt = position_with_wind_const_bank(t_traj, phi_sol, p)

    # (2) Wings-level reference (φ = 0), also truncated at t_hit
    x_straight, y_straight = position_with_wind_const_bank(t_traj, 0.0, p)

    # Point of closest approach on optimal trajectory (should be ~ last point)
    x_ca, y_ca = position_with_wind_const_bank(t_hit, phi_sol, p)

    # -------------------------
    # Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

#     ax.plot(x_straight, y_straight, "0.7", label="Wings-level (with wind)")
    ax.plot(x_opt, y_opt, "b-", label=f"Wind-aware φ ≈ {phi_deg:.1f}°")

    ax.plot([x0], [y0], "go", label="Start")
    ax.plot([xt], [yt], "rx", ms=10, mew=2, label="Target")
    ax.plot([x_ca], [y_ca], "m*", ms=12, label="Closest approach / hit")

    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("x East [m]")
    ax.set_ylabel("y North [m]")
    ax.set_title(
        "Ground track with wind (trajectory stops at target hit / closest approach)"
    )
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()
    
def run_example_with_roll(constants, start, end, psi0, phi0, roll_rate):
    g, V_TAS, bank_angle, V_w, theta_wa, dt = constants
    x0, y0 = start
    xt, yt = end

    t_search = 300.0
    phi_max = np.deg2rad(89.0)
    tol_pos = 0.5
    p_max_roll = roll_rate

    # Solve bank command with roll dynamics
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
        p_max_roll=p_max_roll,
        t_max=t_search,
        dt=dt,
        phi_max=phi_max,
        tol_pos=tol_pos,
        max_iter=40,
    )

    phi_deg = np.rad2deg(phi_cmd)
    print("=== Wind-aware bank (with roll dynamics, no roll-out) ===")
    print(f"Converged:         {info['converged']}")
    print(f"Commanded φ:       {phi_deg:.2f} deg")
    print(f"Closest distance:  {info['d_min']:.1f} m")
    print(f"Signed error:      {info['err']:.1f} m")
    print(f"Time of c.a.:      {info['t_min']:.1f} s")
    print(
        f"Bracket [deg]:     "
        f"[{np.rad2deg(info['bracket'][0]):.1f}, "
        f"{np.rad2deg(info['bracket'][1]):.1f}]"
    )

    # Rebuild params and simulate trajectory to t_hit
    Vw_x = V_w * np.sin(theta_wa + np.pi)
    Vw_y = V_w * np.cos(theta_wa + np.pi)
    p_roll = TurnWithWindRollParams(
        x0=x0,
        y0=y0,
        psi0=psi0,
        phi0=phi0,
        g=g,
        V_TAS=V_TAS,
        Vw_x=Vw_x,
        Vw_y=Vw_y,
        p_max=p_max_roll,
    )

    t_hit = min(info["t_min"], t_search)
    t_traj, x_traj, y_traj, psi_traj, phi_traj = trajectory_with_wind_roll(
        phi_cmd, p_roll, t_final=t_hit, dt=dt
    )

    # Wings-level reference with same roll model (phi_cmd = 0)
    t_ws, x_ws, y_ws, _, _ = trajectory_with_wind_roll(
        0.0, p_roll, t_final=t_hit, dt=dt
    )

    x_ca = x_traj[-1]
    y_ca = y_traj[-1]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x_ws, y_ws, "0.7", label="Wings-level (with wind & roll model)")
    ax.plot(x_traj, y_traj, "b-", label=f"Wind-aware φ_cmd ≈ {phi_deg:.1f}°")
    ax.plot([x0], [y0], "go", label="Start")
    ax.plot([xt], [yt], "rx", ms=10, mew=2, label="Target")
    ax.plot([x_ca], [y_ca], "m*", ms=12, label="Closest approach / hit")

    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("x East [m]")
    ax.set_ylabel("y North [m]")
    ax.set_title("Ground track with wind and roll dynamics (no roll-out)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
