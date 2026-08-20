import csv
import os
import time
from datetime import datetime, timezone
import numpy as np
from pymavlink import mavutil

from uav_opt.config import AircraftConfig, L1Config
from uav_opt.mavlink_client import (
    MavlinkClient,
    roll_to_pwm,
    airspeed_to_throttle_pwm,
)
from uav_opt.simulator_helper import *
from uav_opt.simulator import compute_l1_circular_distance, l1_bank_command
from uav_opt.wind import wind_correction,  wind_to_xy_velocity, air_and_ground_velocity_xy
from uav_opt.path_utils import (
    fill_sparse_points,
    extrapolate_end,
    find_closest_point,
    find_l1_point_by_straight_distance,
    stack_with_next_segment,
)

from uav_opt.maneuvers.bank_angle_solver import desired_bank_to_point_with_wind_roll


class SITLPathFollower:
    """
    Online path follower for ArduPilot SITL/vehicle.

    Sends RC roll/throttle override in FBWB.

    The target path is already in UTM coordinates, same as aircraft_state().
    """

    def __init__(
        self,
        mav: MavlinkClient,
        aircraft: AircraftConfig,
        l1: L1Config,
        use_wind_aware_bank_solver: bool,
    ):
        self.mav = mav
        self.aircraft = aircraft
        self.l1 = l1
        self.use_wind_aware_bank_solver = use_wind_aware_bank_solver

        # ArduPlane-style cross-track integrator state.
        # The integrator is in radians of L1 guidance-angle correction.
        self.xtrack_i = 0.0
        self._xtrack_i_time = None

        # These defaults can later be moved into L1Config.
        self.xtrack_i_gain = getattr(self.l1, "xtrack_i_gain", 0.0)
        self.xtrack_i_limit = getattr(self.l1, "xtrack_i_limit", 0.2)
        self.xtrack_i_enable_angle = getattr(
            self.l1,
            "xtrack_i_enable_angle",
            np.deg2rad(20.0),
        )

        # Previous gain used to detect runtime gain changes.
        self._last_xtrack_i_gain = self.xtrack_i_gain

    def follow(
        self,
        subarrays: list[tuple[np.ndarray, np.ndarray]],
        max_time_s: float = 500.0,
        message_rate_hz: float = 10.0,
        log_path: str = "path_follower_log.csv",
    ) -> None:

        log_file = open(log_path, "w", newline="", buffering=1)

        log_fields = [
            "timestamp_utc",
            "iteration",
            "segment",
            "closest_index",
            "stop_index",

            "position_x_m",
            "position_y_m",
            "cross_track_error_m",
            "cross_track_error_abs_m",
            "nu1_deg",
            "xtrack_i_deg",
            "integral_active",
            "integral_lateral_offset_m",

            "heading_deg",
            "ground_track_deg",
            "ground_speed_mps",
            "airspeed_mps",

            "l1_distance_m",
            "l1_time_s",
            "desired_roll_deg",
            "actual_roll_deg",
            "roll_pwm",
            "throttle_pwm",

            "solver_status",

            "runtime_ms",
            "target_period_ms",
            "loop_utilization_pct",
            "frequency_hz",

            "max_check_ms",
            "state_ms",
            "velocity_ms",
            "closest_ms",
            "l1_distance_ms",
            "xtrack_ms",
            "l1_point_ms",
            "bank_ms",
            "clip_ms",
            "pwm_ms",
            "send_ms",
            "print_ms",
            "sleep_commanded_ms",
            "sleep_actual_ms",
            "total_loop_ms",
        ]

        csv_writer = csv.DictWriter(log_file, fieldnames=log_fields)
        csv_writer.writeheader()

        print("Starting online SITL path follower.")

        self.mav.set_message_rate(
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            message_rate_hz,
        )
        self.mav.set_message_rate(
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
            message_rate_hz,
        )
        self.mav.set_message_rate(
            mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD,
            message_rate_hz,
        )

        self.mav.set_mode("FBWB")
        self.mav.wait_mode("FBWB", timeout=5.0)

        rc_limits = self.mav.get_rc_channel_limits()
        rc_map = rc_limits["map"]

        roll_limits = rc_limits["roll"]
        throttle_limits = rc_limits["throttle"]

        airspeed_min = self.mav.get_param("AIRSPEED_MIN")
        airspeed_max = self.mav.get_param("AIRSPEED_MAX")
        e2t = self.mav.get_e2t(timeout=2.0)
        print(e2t)

        throttle_pwm = airspeed_to_throttle_pwm(
            target_airspeed_mps=self.aircraft.airspeed_mps/e2t,
            min_airspeed_mps=airspeed_min,
            max_airspeed_mps=airspeed_max,
            pwm_min=throttle_limits["min"],
            pwm_max=throttle_limits["max"],
        )

        print(f"Computed throttle PWM for target airspeed: {throttle_pwm}")

        start_time = time.time()
        iteration = 0

        for segment_index, (x_seg_raw, y_seg_raw) in enumerate(subarrays):
            if time.time() - start_time > max_time_s:
                print("Follower max_time_s reached.")
                break

            print(f"Following segment {segment_index + 1}/{len(subarrays)}")

            # Reset cross-track integral when starting a new path segment.
            # self.xtrack_i = 0.0
            # self._xtrack_i_time = None

            if len(x_seg_raw) == 2:
                x_seg, y_seg = fill_sparse_points((x_seg_raw, y_seg_raw))
                x_track, y_track = extrapolate_end((x_seg, y_seg))
            else:
                x_seg = np.asarray(x_seg_raw, dtype=float)
                y_seg = np.asarray(y_seg_raw, dtype=float)
                x_track, y_track = stack_with_next_segment(subarrays, segment_index, x_seg, y_seg)

            closest_index = 0
            stop_index = max(len(x_seg) - 8, 1)
            while closest_index < stop_index:
                loop_t0 = time.perf_counter()

                # ----------------------------
                # 1. max_time_s check
                # ----------------------------
                t0 = time.perf_counter()
                if time.time() - start_time > max_time_s:
                    print("Follower max_time_s reached inside segment.")
                    return
                t_max_check = time.perf_counter() - t0

                # ----------------------------
                # 2. MAVLink state receive
                # ----------------------------
                t0 = time.perf_counter()
                state = self.mav.aircraft_state(timeout=2.0)
                t_state = time.perf_counter() - t0

                # ----------------------------
                # 3. Air/ground velocity
                # ----------------------------
                t0 = time.perf_counter()
                air_velocity, ground_velocity, ground_speed, ground_track = air_and_ground_velocity_xy(
                    airspeed_mps=self.aircraft.airspeed_mps,
                    heading_rad=state.heading_rad,
                    wind_speed_mps=self.l1.wind_speed_mps,
                    wind_from_direction_rad=self.l1.wind_from_direction_rad,
                )

                ground_speed = max(float(ground_speed), 0.1)
                t_velocity = time.perf_counter() - t0

                # ----------------------------
                # 4. Find closest path point
                # ----------------------------
                t0 = time.perf_counter()
                closest_point, closest_index = find_closest_point(
                    aircraft_position_xy=state.position_utm,
                    x_path=x_track,
                    y_path=y_track,
                    previous_index=closest_index,
                )
                cross_track_error, normal_left = (
                    signed_cross_track_error_and_normal(
                        aircraft_position_xy=state.position_utm,
                        x_path=x_track,
                        y_path=y_track,
                        closest_index=closest_index,
                    )
                )

                t_closest = time.perf_counter() - t0

                # ----------------------------
                # 5. Compute L1 distance
                # ----------------------------
                t0 = time.perf_counter()
                if self.use_wind_aware_bank_solver:
                    l1_circ = compute_l1_circular_distance(
                        ground_speed_mps=ground_speed,
                        period_s=self.l1.period_s,
                        max_bank_rad=self.aircraft.max_bank_rad,
                    )

                    l1_ref = compute_l1_circular_distance(
                        self.aircraft.airspeed_mps,
                        self.l1.period_s,
                        self.aircraft.max_bank_rad,
                    )

                    if l1_circ > l1_ref:
                        l1_distance = l1_ref
                    else:
                        l1_distance = l1_circ
                else:
                    l1_distance = ardupilot_l1_distance(
                        ground_speed_mps=ground_speed,
                        damping=self.l1.damping,
                        period_s=self.l1.period_s,
                    )
                l1_time = l1_distance / ground_speed
                
                t_l1_distance = time.perf_counter() - t0

                # ----------------------------
                # 6. Compute cross-track error
                #    and update L1 integrator
                # ----------------------------
                t0 = time.perf_counter()

                if np.isfinite(cross_track_error):
                    nu1_rad, xtrack_i_rad = self._update_cross_track_integrator(
                        cross_track_error_m=cross_track_error,
                        l1_distance_m=l1_distance,
                        nominal_loop_period_s=1.0 / message_rate_hz,
                    )
                else:
                    nu1_rad = 0.0
                    xtrack_i_rad = self.xtrack_i

                t_xtrack = time.perf_counter() - t0

                # ----------------------------
                # 7. Find L1 lookahead point
                # ----------------------------
                t0 = time.perf_counter()

                l1_point, l1_index = find_l1_point_by_straight_distance(
                    closest_index=closest_index,
                    x_path=x_track,
                    y_path=y_track,
                    l1_distance_m=l1_distance,
                )

                t_l1_point = time.perf_counter() - t0

                # ----------------------------
                # 8. Apply cross-track integral
                # ----------------------------
                #
                # xtrack_i_rad is an angular correction to the L1
                # position component. Convert it into an equivalent
                # lateral displacement at the L1 distance.
                #
                # Positive cross-track error is left of the path.
                # Therefore, the integral correction shifts the
                # lookahead point to the right, toward the path.
                #
                integral_lateral_offset_m = (
                    l1_distance
                    * np.sin(xtrack_i_rad)
                )

                integral_active = (
                    np.isfinite(nu1_rad)
                    and abs(nu1_rad) < self.xtrack_i_enable_angle
                    and self.xtrack_i_gain != 0.0
                )

                l1_point_corrected = (
                    np.asarray(l1_point, dtype=float)
                    - normal_left * integral_lateral_offset_m
                )

                # ----------------------------
                # 9. Compute desired bank
                # ----------------------------
                t0 = time.perf_counter()

                desired_roll, solver_status = self._compute_desired_bank(
                    state=state,
                    l1_point=l1_point_corrected,
                    ground_speed=ground_speed,
                    ground_track=ground_track,
                    l1_distance=l1_distance,
                    t_max=l1_time * 5,
                )

                t_desired_bank = time.perf_counter() - t0

                # ----------------------------
                # 10. Clip desired roll
                # ----------------------------
                t0 = time.perf_counter()
                desired_roll = float(
                    np.clip(
                        desired_roll,
                        -self.aircraft.max_bank_rad,
                        self.aircraft.max_bank_rad,
                    )
                )
                t_clip = time.perf_counter() - t0

                # ----------------------------
                # 11. Convert roll to PWM
                # ----------------------------
                t0 = time.perf_counter()
                roll_pwm = roll_to_pwm(
                    desired_roll_rad=desired_roll,
                    max_roll_rad=self.aircraft.max_bank_rad,
                    pwm_min=roll_limits["min"],
                    pwm_max=roll_limits["max"],
                )
                t_pwm = time.perf_counter() - t0

                # ----------------------------
                # 12. Send RC override
                # ----------------------------
                t0 = time.perf_counter()
                self.mav.send_rc_override(
                    roll=roll_pwm,
                    throttle=throttle_pwm,
                    rc_map=rc_map,
                )
                t_send = time.perf_counter() - t0

                # Time before print and before sleep
                loop_dt_before_print = time.perf_counter() - loop_t0
                freq_before_print = 1.0 / loop_dt_before_print if loop_dt_before_print > 1e-6 else float("inf")

                # ----------------------------
                # 13. Print timing
                # ----------------------------
                t0 = time.perf_counter()

                print(
                    f"seg={segment_index:02d} "
                    f"idx={closest_index:04d}/{stop_index:04d} "
                    f"xtrack={cross_track_error:7.2f}m "
                    f"nu1={np.rad2deg(nu1_rad):6.2f}deg "
                    f"xtrack_i={np.rad2deg(xtrack_i_rad):6.2f}deg "
                    f"solver={solver_status} "
                    f"cmd_roll={np.rad2deg(desired_roll):6.1f}deg "
                    f"actual_roll={np.rad2deg(state.roll_rad):6.1f}deg "
                    f"freq_no_sleep={freq_before_print:5.1f}Hz "
                    f"dt_no_sleep={loop_dt_before_print*1000:7.1f}ms"
                )

                print(
                    "TIMING ms | "
                    f"max_check={t_max_check*1000:7.2f} "
                    f"state={t_state*1000:7.2f} "
                    f"vel={t_velocity*1000:7.2f} "
                    f"closest={t_closest*1000:7.2f} "
                    f"xtrack={t_xtrack*1000:7.2f} "
                    f"l1_dist={t_l1_distance*1000:7.2f} "
                    f"l1_point={t_l1_point*1000:7.2f} "
                    f"bank={t_desired_bank*1000:7.2f} "
                    f"clip={t_clip*1000:7.2f} "
                    f"pwm={t_pwm*1000:7.2f} "
                    f"send={t_send*1000:7.2f}"
                )

                t_print = time.perf_counter() - t0

                iteration += 1

                # ----------------------------
                # 12. Maintain approximate loop rate
                # ----------------------------
                t0 = time.perf_counter()
                target_dt = 1.0 / message_rate_hz
                elapsed_before_sleep = time.perf_counter() - loop_t0
                sleep_s = max(0.0, target_dt - elapsed_before_sleep)
                time.sleep(sleep_s)
                t_sleep_actual = time.perf_counter() - t0

                loop_dt_total = time.perf_counter() - loop_t0
                loop_utilization_pct = 100.0 * loop_dt_before_print / target_dt
                freq_total = 1.0 / loop_dt_total if loop_dt_total > 1e-6 else float("inf")

                print(
                    "LOOP ms   | "
                    f"print={t_print*1000:7.2f} "
                    f"sleep_cmd={sleep_s*1000:7.2f} "
                    f"sleep_actual={t_sleep_actual*1000:7.2f} "
                    f"total={loop_dt_total*1000:7.2f} "
                    f"freq_total={freq_total:5.1f}Hz"
                )

                print("-" * 140)

                timestamp_utc = datetime.now(timezone.utc).isoformat()

                csv_writer.writerow(
                    {
                        "timestamp_utc": timestamp_utc,
                        "iteration": iteration,
                        "segment": segment_index,
                        "closest_index": closest_index,
                        "stop_index": stop_index,

                        "position_x_m": float(state.position_utm[0]),
                        "position_y_m": float(state.position_utm[1]),
                        "cross_track_error_m": cross_track_error,
                        "cross_track_error_abs_m": abs(cross_track_error),
                        "nu1_deg": float(np.rad2deg(nu1_rad)),
                        "xtrack_i_deg": float(np.rad2deg(xtrack_i_rad)),
                        "integral_active": int(integral_active),
                        "integral_lateral_offset_m": float(integral_lateral_offset_m),

                        "heading_deg": np.rad2deg(state.heading_rad),
                        "ground_track_deg": np.rad2deg(ground_track),
                        "ground_speed_mps": ground_speed,
                        "airspeed_mps": self.aircraft.airspeed_mps,

                        "l1_distance_m": l1_distance,
                        "xtrack_ms": t_xtrack * 1000.0,
                        "l1_time_s": l1_time,
                        "desired_roll_deg": np.rad2deg(desired_roll),
                        "actual_roll_deg": np.rad2deg(state.roll_rad),
                        "roll_pwm": roll_pwm,
                        "throttle_pwm": throttle_pwm,

                        "solver_status": solver_status,

                        # Runtime before logging and sleeping.
                        "runtime_ms": loop_dt_before_print * 1000.0,
                        "target_period_ms": target_dt * 1000.0,
                        "loop_utilization_pct": loop_utilization_pct,
                        "frequency_hz": 1.0 / loop_dt_before_print if loop_dt_before_print > 1e-6 else float("inf"),

                        "max_check_ms": t_max_check * 1000.0,
                        "state_ms": t_state * 1000.0,
                        "velocity_ms": t_velocity * 1000.0,
                        "closest_ms": t_closest * 1000.0,
                        "l1_distance_ms": t_l1_distance * 1000.0,
                        "l1_point_ms": t_l1_point * 1000.0,
                        "bank_ms": t_desired_bank * 1000.0,
                        "clip_ms": t_clip * 1000.0,
                        "pwm_ms": t_pwm * 1000.0,
                        "send_ms": t_send * 1000.0,
                        "print_ms": t_print * 1000.0,
                        "sleep_commanded_ms": sleep_s * 1000.0,
                        "sleep_actual_ms": t_sleep_actual * 1000.0,
                        "total_loop_ms": loop_dt_total * 1000.0,
                    }
                )

                log_file.flush()



        print("Path follower complete.")
        log_file.close()
        return

    def _update_cross_track_integrator(
        self,
        cross_track_error_m: float,
        l1_distance_m: float,
        nominal_loop_period_s: float,
    ) -> tuple[float, float]:
        """
        Update the ArduPlane-style cross-track integrator.

        Returns:
            nu1_rad:
                Instantaneous L1 cross-track angle.
            xtrack_i_rad:
                Integrated cross-track angle.
        """
        l1_distance_m = max(float(l1_distance_m), 0.1)

        # ArduPlane limits the instantaneous position component to
        # approximately +/-45 degrees.
        nu1_rad = float(
            np.arcsin(
                np.clip(
                    cross_track_error_m / l1_distance_m,
                    -0.7071,
                    0.7071,
                )
            )
        )

        now = time.perf_counter()

        if self._xtrack_i_time is None:
            dt_s = float(nominal_loop_period_s)
        else:
            dt_s = now - self._xtrack_i_time

        self._xtrack_i_time = now

        # Reset after a genuine update gap greater than one second.
        if dt_s > 1.0:
            self.xtrack_i = 0.0

        # Match the ArduPlane-style maximum integration timestep.
        dt_s = float(np.clip(dt_s, 0.0, 0.1))

        # Reset if disabled or if the gain changes.
        if (
            self.xtrack_i_gain <= 0.0
            or self.xtrack_i_gain != self._last_xtrack_i_gain
        ):
            self.xtrack_i = 0.0
            self._last_xtrack_i_gain = self.xtrack_i_gain

        elif abs(nu1_rad) < self.xtrack_i_enable_angle:
            self.xtrack_i += (
                nu1_rad
                * self.xtrack_i_gain
                * dt_s
            )

            self.xtrack_i = float(
                np.clip(
                    self.xtrack_i,
                    -self.xtrack_i_limit,
                    self.xtrack_i_limit,
                )
            )

        return nu1_rad, self.xtrack_i

    def _compute_desired_bank(
        self,
        state,
        l1_point: np.ndarray,
        ground_speed: float,
        ground_track: float,
        l1_distance: float,
        t_max: float,
    ) -> tuple[float, str]:
        """
        Try wind-aware bank solver; fall back to standard L1 bank.
        
        Returns:
        desired_bank_rad, solver_status
        """
        if self.use_wind_aware_bank_solver:
            try:
                bank, info = desired_bank_to_point_with_wind_roll(
                    x0=state.position_utm[0],
                    y0=state.position_utm[1],
                    psi0=state.heading_rad,
                    phi0=state.roll_rad,
                    xt=l1_point[0],
                    yt=l1_point[1],
                    g=9.81,
                    V_TAS=self.aircraft.airspeed_mps,
                    V_w=self.l1.wind_speed_mps,
                    theta_wa=self.l1.wind_from_direction_rad,
                    p_max_roll=self.aircraft.roll_rate_rad_s,
                    t_max=t_max,
                    dt=self.l1.dt_solver,
                    phi_max=float(self.aircraft.max_bank_rad),
                    tol_pos=self.l1.tol_pos,
                    max_iter=int(self.l1.max_iter),
                )

                if info.get("converged", False):
                    return float(bank), "converged"

                print("Wind-aware bank solver did not converge; using L1 fallback.")

            except Exception as exc:
                print(f"Wind-aware bank solver error: {exc}; using L1 fallback.")

        return l1_bank_command(
            aircraft_position=state.position_utm,
            ground_track_rad=ground_track,
            l1_point=l1_point,
            ground_speed_mps=ground_speed,
            damping=self.l1.damping,
            l1_distance_m=l1_distance,
        ), "fallback_nonconverged"

def signed_cross_track_error_and_normal(
    aircraft_position_xy: np.ndarray,
    x_path: np.ndarray,
    y_path: np.ndarray,
    closest_index: int,
) -> tuple[float, np.ndarray]:
    """
    Return signed cross-track error and left-hand unit normal.

    Coordinates:
        x = East
        y = North

    Positive cross-track error means the aircraft is left of the
    selected path direction.
    """
    p_aircraft = np.asarray(
        aircraft_position_xy,
        dtype=float,
    )

    x_path = np.asarray(x_path, dtype=float)
    y_path = np.asarray(y_path, dtype=float)

    n = min(len(x_path), len(y_path))

    if n < 2:
        return (
            float("nan"),
            np.array([np.nan, np.nan], dtype=float),
        )

    i = int(np.clip(closest_index, 0, n - 1))

    # Search for a valid nonzero local segment. This protects against
    # duplicate points at segment boundaries.
    candidate_indices = []

    for offset in range(0, n):
        candidate_indices.append(i + offset)
        candidate_indices.append(i - offset)

    selected_segment = None

    for j in candidate_indices:
        if j < 0 or j >= n - 1:
            continue

        p0 = np.array(
            [x_path[j], y_path[j]],
            dtype=float,
        )

        p1 = np.array(
            [x_path[j + 1], y_path[j + 1]],
            dtype=float,
        )

        segment = p1 - p0
        segment_length = float(np.linalg.norm(segment))

        if np.isfinite(segment_length) and segment_length > 1.0e-6:
            selected_segment = (p0, p1, segment, segment_length)
            break

    # If no forward segment is valid, search backward.
    if selected_segment is None:
        for j in range(n - 2, -1, -1):
            p0 = np.array(
                [x_path[j], y_path[j]],
                dtype=float,
            )

            p1 = np.array(
                [x_path[j + 1], y_path[j + 1]],
                dtype=float,
            )

            segment = p1 - p0
            segment_length = float(np.linalg.norm(segment))

            if np.isfinite(segment_length) and segment_length > 1.0e-6:
                selected_segment = (
                    p0,
                    p1,
                    segment,
                    segment_length,
                )
                break

    if selected_segment is None:
        return (
            float("nan"),
            np.array([np.nan, np.nan], dtype=float),
        )

    p0, p1, segment, segment_length = selected_segment

    tangent = segment / segment_length

    normal_left = np.array(
        [-tangent[1], tangent[0]],
        dtype=float,
    )

    error_vector = p_aircraft - p0

    cross_track_error = float(
        np.dot(error_vector, normal_left)
    )

    if not np.isfinite(cross_track_error):
        return (
            float("nan"),
            np.array([np.nan, np.nan], dtype=float),
        )

    return cross_track_error, normal_left
