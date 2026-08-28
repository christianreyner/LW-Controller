import numpy as np
import matplotlib.pyplot as plt
import time

from uav_opt.config import AircraftConfig, L1Config
from uav_opt.aero import aero_energy_cost_wh
from uav_opt.wind import wind_correction, wind_to_xy_velocity, air_and_ground_velocity_xy
from uav_opt.angles import wrap_2pi, signed_angle_to_point
from uav_opt.path_utils import (
    fill_sparse_points,
    extrapolate_end,
    find_closest_point,
    find_l1_point_by_straight_distance,
    find_l1_point_by_path_distance,
    stack_with_next_segment,
)
from uav_opt.simulator_helper import *
from uav_opt.maneuvers.bank_angle_solver import desired_bank_to_point_with_wind_roll


def compute_l1_circular_distance(
    ground_speed_mps: float,
    period_s: float,
    max_bank_rad: float,
    g: float = 9.81,
) -> float:
    R = ground_speed_mps**2 / (g * np.tan(max_bank_rad))
    numerator = 2.0 * ground_speed_mps**2
    denominator = (4.0 * np.pi**2 / period_s**2) + (ground_speed_mps**2 / (2.0 * R**2))
    return float(np.sqrt(numerator / denominator))


def l1_bank_command(
    aircraft_position: np.ndarray,
    ground_track_rad: float,
    l1_point: np.ndarray,
    ground_speed_mps: float,
    damping: float,
    l1_distance_m: float,
) -> float:
    """
    L1 bank command.

    Important:
    L1 guidance should use the angle between the GROUND VELOCITY direction
    and the L1 lookahead point, not the aircraft heading/nose direction.

    heading_rad      = air-relative nose direction
    ground_track_rad = actual movement direction over ground
    """
    l1_angle = signed_angle_to_point(
        aircraft_position,
        ground_track_rad,
        l1_point,
    )

    # Optional but ArduPilot-like: prevent using target points behind aircraft.
    l1_angle = float(np.clip(l1_angle, -np.pi / 2.0, np.pi / 2.0))

    lateral_accel = (
        4.0
        * damping**2
        * ground_speed_mps**2
        * np.sin(l1_angle)
        / max(l1_distance_m, 1e-6)
    )

    return float(np.arctan(lateral_accel / 9.81))

def compute_desired_roll_for_l1_point(
    aircraft_position: np.ndarray,
    heading: float,
    roll: float,
    l1_point: np.ndarray,
    ground_speed: float,
    ground_track: float,
    l1_distance: float,
    aircraft: AircraftConfig,
    l1: L1Config,
    t_max: float,
    use_wind_aware_bank_solver: bool,
) -> tuple[float, bool]:
    """
    Compute desired roll for the internal L1 preview simulation.

    If enabled, try the wind-aware bank solver first.
    If it fails or does not converge, fall back to standard L1 guidance.

    Important:
    - Wind-aware solver uses aircraft heading and current roll.
    - L1 fallback uses ground track, not aircraft heading.
    """
    if use_wind_aware_bank_solver:
        try:
            bank, info = desired_bank_to_point_with_wind_roll(
                x0=float(aircraft_position[0]),
                y0=float(aircraft_position[1]),
                psi0=float(heading),
                phi0=float(roll),
                xt=float(l1_point[0]),
                yt=float(l1_point[1]),
                g=9.81,
                V_TAS=float(aircraft.airspeed_mps),
                V_w=float(l1.wind_speed_mps),
                theta_wa=float(l1.wind_from_direction_rad),
                p_max_roll=float(aircraft.roll_rate_rad_s),
                t_max=float(t_max),
                dt=float(l1.dt_solver),
                phi_max=float(aircraft.max_bank_rad),
                tol_pos=float(l1.tol_pos),
                max_iter=int(l1.max_iter),
            )
            if info.get("converged", False):
                return float(bank), True

        except Exception:
            pass
    return l1_bank_command(
        aircraft_position=aircraft_position,
        ground_track_rad=ground_track,
        l1_point=l1_point,
        ground_speed_mps=ground_speed,
        damping=l1.damping,
        l1_distance_m=l1_distance,
    ), False
    
def simulate_l1_path(
    subarrays: list[tuple[np.ndarray, np.ndarray]],
    initial_position: np.ndarray,
    initial_heading: float,
    aircraft: AircraftConfig,
    l1: L1Config,
    sim_time_s: float = 500.0,
):
    """
    Lightweight internal simulation for preview.

    DYN columns:
    0 step/time-index
    1 x
    2 y
    3 z
    4 vx
    5 vy
    6 vz
    7 heading
    8 roll
    9 pitch
    10 energy cost increment
    11 segment index
    """
    dt = l1.dt_s
    max_steps = int(sim_time_s / dt)

    pos = np.asarray(initial_position, dtype=float).copy()
    heading = float(initial_heading)
    roll = 0.0
    pitch = 0.0

    # Override heading only at t=0 so the aircraft starts with correct crab
    if len(subarrays) > 0:
        x_seg_raw, y_seg_raw = subarrays[0]

        if len(x_seg_raw) == 2:
            x_seg0, y_seg0 = fill_sparse_points((x_seg_raw, y_seg_raw))
            x_track0, y_track0 = extrapolate_end((x_seg0, y_seg0))
        else:
            x_seg0 = np.asarray(x_seg_raw, dtype=float)
            y_seg0 = np.asarray(y_seg_raw, dtype=float)
            x_track0, y_track0 = stack_with_next_segment(
                subarrays,
                0,
                x_seg0,
                y_seg0,
            )

        if len(x_track0) >= 2:
            dx = float(x_track0[1] - x_track0[0])
            dy = float(y_track0[1] - y_track0[0])
            course0 = float(np.arctan2(dx, dy))

            _, heading, _ = wind_correction(
                airspeed_mps=aircraft.airspeed_mps,
                course_rad=course0,
                wind_speed_mps=l1.wind_speed_mps,
                wind_from_direction_rad=l1.wind_from_direction_rad,
            )
            
    dyn_rows = []

    y_step = 0

    for segment_index, (x_seg_raw, y_seg_raw) in enumerate(subarrays):
        if y_step >= max_steps:
            break

        if len(x_seg_raw) == 2:
            x_seg, y_seg = fill_sparse_points((x_seg_raw, y_seg_raw))
            x_track, y_track = extrapolate_end((x_seg, y_seg))
        else:
            x_seg = np.asarray(x_seg_raw, dtype=float)
            y_seg = np.asarray(y_seg_raw, dtype=float)
            x_track, y_track = stack_with_next_segment(
                subarrays,
                segment_index,
                x_seg,
                y_seg,
            )

        closest_index = 0
        segment_stop_index = max(len(x_seg) - 8, 1)
        
        succ_count = 0
        fail_count = 0
        t0 = time.time()
        while closest_index < segment_stop_index and y_step < max_steps:
            air_velocity, ground_velocity, ground_speed, ground_track = air_and_ground_velocity_xy(
                airspeed_mps=aircraft.airspeed_mps,
                heading_rad=heading,
                wind_speed_mps=l1.wind_speed_mps,
                wind_from_direction_rad=l1.wind_from_direction_rad,
            )

            ground_speed = max(float(ground_speed), 0.1)

            aircraft_position = pos.copy()

            closest_point, closest_index = find_closest_point(
                aircraft_position,
                x_track,
                y_track,
                closest_index,
            )

            if l1.use_wind_aware_bank_solver:
                # Using the circular l1
                l1_circ = compute_l1_circular_distance(
                    ground_speed,
                    l1.period_s,
                    aircraft.max_bank_rad,
                )

                l1_ref = compute_l1_circular_distance(
                    aircraft.airspeed_mps,
                    l1.period_s,
                    aircraft.max_bank_rad,
                )

                if l1_circ > l1_ref:
                    l1_distance = l1_ref
                else:
                    l1_distance = l1_circ
                    
                l1_time = l1_distance / ground_speed
            else:
                # Using the fixed l1 period
                l1_distance = ardupilot_l1_distance(
                    ground_speed_mps=ground_speed,
                    damping=l1.damping,
                    period_s=l1.period_s,
                )
                l1_time = l1_distance / ground_speed
                
            l1_point, _ = find_l1_point_by_path_distance(
                closest_index,
                x_track,
                y_track,
                l1_distance,
            )

            desired_roll, wind_aware = compute_desired_roll_for_l1_point(
                aircraft_position=aircraft_position,
                heading=heading,
                roll=roll,
                l1_point=l1_point,
                ground_speed=ground_speed,
                ground_track=ground_track,
                l1_distance=l1_distance,
                aircraft=aircraft,
                l1=l1,
                t_max=l1_time*5, #ensuring enough time
                use_wind_aware_bank_solver=l1.use_wind_aware_bank_solver,
            )
            if wind_aware:
                succ_count += 1
            else:
                fail_count += 1

            desired_roll = float(
                np.clip(
                    desired_roll,
                    -aircraft.max_bank_rad,
                    aircraft.max_bank_rad,
                )
            )

            max_delta_roll = aircraft.roll_rate_rad_s * dt
            roll += float(
                np.clip(
                    desired_roll - roll,
                    -max_delta_roll,
                    max_delta_roll,
                )
            )
            roll = float(np.clip(roll, -aircraft.max_bank_rad, aircraft.max_bank_rad))

            lateral_accel = np.tan(roll) * 9.81
            heading = wrap_2pi(
                heading + lateral_accel * dt / max(aircraft.airspeed_mps, 1e-6)
            )

            air_velocity, ground_velocity, ground_speed, ground_track = air_and_ground_velocity_xy(
                airspeed_mps=aircraft.airspeed_mps,
                heading_rad=heading,
                wind_speed_mps=l1.wind_speed_mps,
                wind_from_direction_rad=l1.wind_from_direction_rad,
            )

            vx = float(ground_velocity[0])
            vy = float(ground_velocity[1])

            cost = aero_energy_cost_wh(
                bank_rad=roll,
                airspeed_mps=aircraft.airspeed_mps,
                timestep_s=dt,
                aero=aircraft.aero,
            )

            dyn_rows.append(
                [
                    y_step,
                    pos[0],
                    pos[1],
                    0.0,
                    vx,
                    vy,
                    0.0,
                    heading,
                    roll,
                    pitch,
                    cost,
                    segment_index,
                ]
            )

            pos[0] += vx * dt
            pos[1] += vy * dt

            y_step += 1
            
        elapsed_ms = (time.time() - t0) * 1000
        count = succ_count + fail_count
        avg_ms = elapsed_ms / count if count else 0

        print(f"Counter: {succ_count}, {fail_count}, iter time: {avg_ms:.2f}, total time: {elapsed_ms:.2f}")

    return np.asarray(dyn_rows, dtype=float)
    
    
def simulate_conventional_path(
    waypoints: np.ndarray,
    initial_heading_rad: float,
    aircraft: AircraftConfig,
    l1: L1Config,
    sim_time_s: float = 500.0,
    overshoot_m: float = 0.0,
    leadin_m: float = 0.0,
) -> np.ndarray:
    """
    ArduPilot-style conventional waypoint-following simulation.

    This is not old flightsim1(), but it intentionally follows the important
    AP_L1_Control::update_waypoint() structure:

        active leg = previous waypoint -> next waypoint
        L1_dist = (1/pi) * damping * period * ground speed
        Nu = cross-track capture angle + velocity angle relative to active leg
        lateral acceleration = K_L1 * Vg^2 / L1_dist * sin(Nu)
        desired bank = atan(lat_accel / g)

    DYN columns:
    0 step/time-index
    1 x
    2 y
    3 z
    4 vx
    5 vy
    6 vz
    7 heading
    8 roll
    9 pitch
    10 energy cost increment
    11 active waypoint index
    """
    orig = np.asarray(waypoints, dtype=float)
    wp = get_overshoot(
        np.asarray(waypoints, dtype=float),
        overshoot_m=overshoot_m,
        leadin_m=leadin_m,
    )
 
    plt.figure(figsize=(8, 6))

    # Path line
    plt.plot(wp[:, 0], wp[:, 1], "-", color="tab:blue", alpha=0.7)

    # All expanded points
    plt.scatter(wp[:, 0], wp[:, 1], color="tab:blue", s=40, label="Expanded waypoints")

    # Original waypoints
    plt.scatter(orig[:, 0], orig[:, 1], color="red", s=80, marker="x", label="Original waypoints")

    for i, (x, y) in enumerate(orig):
        plt.text(x + 2, y + 2, f"orig {i}", color="red")

    plt.axis("equal")
    plt.grid(True, linestyle=":")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.show()   
    
    if len(wp) < 2:
        return np.empty((0, 12), dtype=float)

    dt = float(l1.dt_s)
    max_steps = int(sim_time_s / dt)

    pos = wp[0, :2].astype(float).copy()
    
    # initial heading
    dx, dy = wp[1, :2] - wp[0, :2]
    course = np.arctan2(dx, dy)

    _, heading, _ = wind_correction(
        airspeed_mps=aircraft.airspeed_mps,
        course_rad=course,
        wind_speed_mps=l1.wind_speed_mps,
        wind_from_direction_rad=l1.wind_from_direction_rad,
    )

    roll = 0.0
    pitch = 0.0

    target_index = 1

    last_nu = 0.0
    xtrack_i = 0.0
    xtrack_i_gain = float(getattr(l1, "xtrack_i_gain", 0.0))

    dyn_rows = []

    for step in range(max_steps):
        if target_index >= len(wp):
            break

        air_velocity, ground_velocity, ground_speed, ground_track = (
            air_and_ground_velocity_xy(
                airspeed_mps=aircraft.airspeed_mps,
                heading_rad=heading,
                wind_speed_mps=l1.wind_speed_mps,
                wind_from_direction_rad=l1.wind_from_direction_rad,
            )
        )

        ground_speed = max(float(ground_speed), 0.1)

        current_l1_distance = ardupilot_l1_distance(
            ground_speed_mps=ground_speed,
            damping=l1.damping,
            period_s=l1.period_s,
        )

        # Advance waypoint if we are already inside the active acceptance zone
        # or have crossed the finish line.
        while target_index < len(wp):
            acceptance_radius = active_waypoint_acceptance_radius(
                wp=wp,
                target_index=target_index,
                wp_radius_m=l1.waypoint_radius_m,
                l1_distance_m=current_l1_distance,
            )

            reached = waypoint_reached_ardupilot_style(
                position_xy=pos,
                prev_wp_xy=wp[target_index - 1, :2],
                next_wp_xy=wp[target_index, :2],
                acceptance_radius_m=acceptance_radius,
            )

            if not reached:
                break

            target_index += 1

        if target_index >= len(wp):
            break

        prev_wp_xy = wp[target_index - 1, :2]
        next_wp_xy = wp[target_index, :2]

        (
            desired_roll,
            l1_distance,
            nu,
            crosstrack_error,
            along_track,
            last_nu,
        ) = ardupilot_l1_waypoint_bank_command(
            position_xy=pos,
            heading_rad=heading,
            prev_wp_xy=prev_wp_xy,
            next_wp_xy=next_wp_xy,
            ground_velocity_xy=ground_velocity,
            damping=l1.damping,
            period_s=l1.period_s,
            last_nu=last_nu,
            xtrack_i=xtrack_i,
            xtrack_i_gain=xtrack_i_gain,
            dt_s=dt,
        )

        desired_roll = float(
            np.clip(
                desired_roll,
                -aircraft.max_bank_rad,
                aircraft.max_bank_rad,
            )
        )

        # Roll Lag in s
        roll_tau = 0.2

        roll_rate_cmd = (desired_roll - roll) / roll_tau
        roll_rate_cmd = np.clip(
            roll_rate_cmd,
            -aircraft.roll_rate_rad_s,
            aircraft.roll_rate_rad_s,
        )

        roll += roll_rate_cmd * dt

        roll = float(
            np.clip(
                roll,
                -aircraft.max_bank_rad,
                aircraft.max_bank_rad,
            )
        )

        lateral_accel = np.tan(roll) * 9.81

        heading = wrap_2pi(
            heading + lateral_accel * dt / max(aircraft.airspeed_mps, 1e-6)
        )

        # Recompute velocity after heading changed.
        air_velocity, ground_velocity, ground_speed, ground_track = (
            air_and_ground_velocity_xy(
                airspeed_mps=aircraft.airspeed_mps,
                heading_rad=heading,
                wind_speed_mps=l1.wind_speed_mps,
                wind_from_direction_rad=l1.wind_from_direction_rad,
            )
        )
        
        vx = float(ground_velocity[0])
        vy = float(ground_velocity[1])

        cost = aero_energy_cost_wh(
            bank_rad=roll,
            airspeed_mps=aircraft.airspeed_mps,
            timestep_s=dt,
            aero=aircraft.aero,
        )

        dyn_rows.append(
            [
                step,
                pos[0],
                pos[1],
                0.0,
                vx,
                vy,
                0.0,
                heading,
                roll,
                pitch,
                cost,
                target_index,
            ]
        )

        pos[0] += vx * dt
        pos[1] += vy * dt

    return np.asarray(dyn_rows, dtype=float)
    
import numpy as np


def get_overshoot(
    waypoints: np.ndarray,
    overshoot_m: float = 0.0,
    leadin_m: float = 0.0,
) -> np.ndarray:
    wp = np.asarray(waypoints, dtype=float)

    if wp.ndim != 2 or wp.shape[1] < 2:
        raise ValueError("waypoints must be an Nx2 or NxM array")

    n = len(wp)
    if n < 2:
        return wp.copy()

    if overshoot_m == 0.0 and leadin_m == 0.0:
        return wp.copy()

    def shape_pair(a: np.ndarray, b: np.ndarray, top: bool):
        a = a.copy()
        b = b.copy()

        ay = a[1]
        by = b[1]

        if top:
            if ay < by:
                a[1] = by + overshoot_m
                b[1] = by + leadin_m
            elif ay > by + overshoot_m:
                b[1] = by + leadin_m
            else:
                a[1] = ay + overshoot_m
                b[1] = by + leadin_m
        else:
            if ay > by:
                a[1] = by - overshoot_m
                b[1] = by - leadin_m
            elif ay < by - overshoot_m:
                b[1] = by - leadin_m
            else:
                a[1] = ay - overshoot_m
                b[1] = by - leadin_m

        return a, b

    result = [wp[0].copy()]
    top = True

    # Process pairs: (1,2), (3,4), (5,6), ...
    for i in range(1, n - 1, 2):
        a, b = shape_pair(wp[i], wp[i + 1], top)

        result.append(wp[i].copy())  # original waypoint
        result.append(a)             # overshoot copy
        result.append(b)             # leadin copy
        result.append(wp[i + 1].copy())  # original next waypoint

        top = not top

    # If there's an unpaired last waypoint, keep it
    if n % 2 == 0:
        result.append(wp[-1].copy())

    return np.asarray(result, dtype=float)
    
def plot_plan_and_simulation(
    original_waypoints_utm: np.ndarray,
    optimal_path_utm: np.ndarray,
    optimal_sim_dyn: np.ndarray | None = None,
    conventional_sim_dyn_utm: np.ndarray | None = None,
    title: str = "Path Preview",
) -> None:
    """
    Plot:
    - original waypoints
    - optimal geometric path
    - conventional waypoint-following simulation
    - optimal-path L1 simulation
    """
    plt.figure(figsize=(9, 7))

    plt.plot(
        original_waypoints_utm[:, 0],
        original_waypoints_utm[:, 1],
        "ko--",
        label="Waypoints",
    )

    if conventional_sim_dyn_utm is not None and len(conventional_sim_dyn_utm) > 0:
        plt.plot(
            conventional_sim_dyn_utm[:, 1],
            conventional_sim_dyn_utm[:, 2],
            color="gray",
            linestyle="-",
            linewidth=1.8,
            label="Conventional path simulation",
        )

    if optimal_sim_dyn is not None and len(optimal_sim_dyn) > 0:
        plt.plot(
            optimal_sim_dyn[:, 1],
            optimal_sim_dyn[:, 2],
            "r-",
            linewidth=1.5,
            label="Optimal path L1 simulation",
        )

    plt.plot(
        optimal_path_utm[:, 0],
        optimal_path_utm[:, 1],
        "b-",
        linewidth=2,
        label="Optimal path",
    )

    plt.title(title)
    plt.xlabel("UTM X / Easting (m)")
    plt.ylabel("UTM Y / Northing (m)")
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
    )

    plt.tight_layout()
    plt.show()
