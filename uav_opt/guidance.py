import csv
import os
import time
from datetime import datetime, timezone
import matplotlib.pyplot as plt

import numpy as np
from pymavlink import mavutil

from uav_opt.config import AircraftConfig, L1Config
from uav_opt.maneuvers.bank_angle_solver import (
    desired_bank_to_point_with_wind_roll,
)
from uav_opt.mavlink_client import (
    MavlinkClient,
    airspeed_to_throttle_pwm,
    roll_to_pwm,
)
from uav_opt.path_utils import (
    extrapolate_end,
    fill_sparse_points,
    find_closest_point,
    find_l1_point_by_straight_distance,
    stack_with_next_segment,
)
from uav_opt.simulator import (
    compute_l1_circular_distance,
    l1_bank_command,
)
from uav_opt.simulator_helper import ardupilot_l1_distance
from uav_opt.wind import air_and_ground_velocity_xy


class SITLPathFollower:
    """
    Online path follower for ArduPilot SITL/vehicle.

    Sends RC roll and throttle overrides in FBWB mode.

    The target path must already be in the same UTM coordinate system
    returned by aircraft_state().
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
        self.use_wind_aware_bank_solver = bool(
            use_wind_aware_bank_solver
        )

        # Cross-track integral state. The stored value is an angular
        # guidance correction in radians.
        self.xtrack_i = 0.0
        self._xtrack_i_time: float | None = None
        
    
        self.bank_reversal_threshold_rad = np.deg2rad(
            float(getattr(self.l1, "bank_reversal_threshold_deg", 2.0))
            )

        self.xtrack_i_gain = float(
            getattr(self.l1, "xtrack_i_gain", 0.2)
        )
        self.xtrack_i_limit = abs(
            float(getattr(self.l1, "xtrack_i_limit", 0.1))
        )
        self.xtrack_i_enable_angle = abs(
            float(
                getattr(
                    self.l1,
                    "xtrack_i_enable_angle",
                    np.deg2rad(5.0),
                )
            )
        )

        if not np.isfinite(self.xtrack_i_gain):
            raise ValueError("xtrack_i_gain must be finite.")

        if not np.isfinite(self.xtrack_i_limit):
            raise ValueError("xtrack_i_limit must be finite.")

        if not np.isfinite(self.xtrack_i_enable_angle):
            raise ValueError(
                "xtrack_i_enable_angle must be finite."
            )

        self._last_xtrack_i_gain = self.xtrack_i_gain

    def follow(
        self,
        subarrays: list[tuple[np.ndarray, np.ndarray]],
        max_time_s: float = 500.0,
        message_rate_hz: float = 5.0,
        log_path: str = "path_follower_log.csv",
    ) -> None:
        """
        Follow the supplied path segments.

        The controller loop nominally runs at message_rate_hz. The
        cross-track integrator uses measured elapsed time, so its
        behavior is independent of whether the loop runs at 4, 5, 10,
        or another valid rate.
        """
        message_rate_hz = float(message_rate_hz)
        max_time_s = float(max_time_s)

        if (
            not np.isfinite(message_rate_hz)
            or message_rate_hz <= 0.0
        ):
            raise ValueError(
                "message_rate_hz must be finite and greater than zero."
            )

        if not np.isfinite(max_time_s) or max_time_s <= 0.0:
            raise ValueError(
                "max_time_s must be finite and greater than zero."
            )

        if not subarrays:
            print("No path segments were provided.")
            return

        target_dt = 1.0 / message_rate_hz

        log_directory = os.path.dirname(
            os.path.abspath(log_path)
        )
        os.makedirs(log_directory, exist_ok=True)

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

        print("Starting online SITL path follower.")
        print(
            f"Requested controller/message rate: "
            f"{message_rate_hz:.3f} Hz"
        )
        print(
            f"Requested controller period: "
            f"{target_dt * 1000.0:.3f} ms"
        )

        self._configure_mavlink_message_rates(
            message_rate_hz=message_rate_hz
        )

        self.mav.set_mode("FBWB")
        self.mav.wait_mode("FBWB", timeout=5.0)

        rc_limits = self.mav.get_rc_channel_limits()
        rc_map = rc_limits["map"]
        roll_limits = rc_limits["roll"]
        throttle_limits = rc_limits["throttle"]

        airspeed_min = float(
            self.mav.get_param("AIRSPEED_MIN")
        )
        airspeed_max = float(
            self.mav.get_param("AIRSPEED_MAX")
        )
        e2t = float(self.mav.get_e2t(timeout=2.0))

        if not np.isfinite(e2t) or abs(e2t) < 1.0e-9:
            raise ValueError(
                f"Invalid E2T conversion value: {e2t!r}"
            )

        target_throttle_airspeed = (
            float(self.aircraft.airspeed_mps) / e2t
        )

        throttle_pwm = airspeed_to_throttle_pwm(
            target_airspeed_mps=target_throttle_airspeed,
            min_airspeed_mps=airspeed_min,
            max_airspeed_mps=airspeed_max,
            pwm_min=throttle_limits["min"],
            pwm_max=throttle_limits["max"],
        )

        print(f"E2T conversion value: {e2t}")
        print(
            "Computed throttle PWM for target airspeed: "
            f"{throttle_pwm}"
        )

        start_time = time.perf_counter()
        next_loop_deadline = start_time + target_dt

        iteration = 0
        timed_out = False

        with open(
            log_path,
            "w",
            newline="",
            buffering=1,
        ) as log_file:
            csv_writer = csv.DictWriter(
                log_file,
                fieldnames=log_fields,
            )
            csv_writer.writeheader()

            for segment_index, segment_data in enumerate(subarrays):
                succ_count = 0
                fail_count = 0
                subopt_count = 0
                count = 0
                comp_cost = 0
                
                if self._max_time_reached(
                    start_time=start_time,
                    max_time_s=max_time_s,
                ):
                    timed_out = True
                    break

                x_seg_raw, y_seg_raw = segment_data

                print(
                    f"Following segment "
                    f"{segment_index + 1}/{len(subarrays)}"
                )

                # Each subarray is treated as a separate path leg.
                self._reset_cross_track_integrator()

                x_seg, y_seg, x_track, y_track = (
                    self._prepare_segment(
                        subarrays=subarrays,
                        segment_index=segment_index,
                        x_seg_raw=x_seg_raw,
                        y_seg_raw=y_seg_raw,
                    )
                )

                closest_index = 0
                stop_index = max(len(x_seg) - 8, 1)

                while closest_index < stop_index:
                    loop_t0 = time.perf_counter()

                    # ----------------------------------------
                    # 1. Maximum-runtime check
                    # ----------------------------------------
                    t0 = time.perf_counter()

                    if self._max_time_reached(
                        start_time=start_time,
                        max_time_s=max_time_s,
                    ):
                        timed_out = True
                        break

                    t_max_check = time.perf_counter() - t0

                    # ----------------------------------------
                    # 2. Receive aircraft state
                    # ----------------------------------------
                    t0 = time.perf_counter()

                    state = self.mav.aircraft_state(
                        timeout=2.0
                    )

                    t_state = time.perf_counter() - t0

                    aircraft_position = np.asarray(
                        state.position_utm,
                        dtype=float,
                    )

                    if (
                        aircraft_position.shape != (2,)
                        or not np.all(
                            np.isfinite(aircraft_position)
                        )
                    ):
                        raise ValueError(
                            "aircraft_state().position_utm must "
                            "contain two finite coordinates."
                        )

                    # ----------------------------------------
                    # 3. Air and ground velocity
                    # ----------------------------------------
                    t0 = time.perf_counter()

                    (
                        _air_velocity,
                        _ground_velocity,
                        ground_speed,
                        ground_track,
                    ) = air_and_ground_velocity_xy(
                        airspeed_mps=self.aircraft.airspeed_mps,
                        heading_rad=state.heading_rad,
                        wind_speed_mps=self.l1.wind_speed_mps,
                        wind_from_direction_rad=(
                            self.l1.wind_from_direction_rad
                        ),
                    )

                    ground_speed = float(ground_speed)
                    ground_track = float(ground_track)

                    if not np.isfinite(ground_speed):
                        ground_speed = 0.1

                    ground_speed = max(ground_speed, 0.1)

                    if not np.isfinite(ground_track):
                        ground_track = float(
                            state.heading_rad
                        )

                    t_velocity = time.perf_counter() - t0

                    # ----------------------------------------
                    # 4. Closest point and cross-track error
                    # ----------------------------------------
                    t0 = time.perf_counter()

                    _, closest_index = find_closest_point(
                        aircraft_position_xy=aircraft_position,
                        x_path=x_track,
                        y_path=y_track,
                        previous_index=closest_index,
                    )

                    closest_index = int(
                        np.clip(
                            closest_index,
                            0,
                            len(x_track) - 1,
                        )
                    )

                    (
                        cross_track_error,
                        _normal_left,
                    ) = signed_cross_track_error_and_normal(
                        aircraft_position_xy=aircraft_position,
                        x_path=x_track,
                        y_path=y_track,
                        closest_index=closest_index,
                    )

                    t_closest = time.perf_counter() - t0

                    # ----------------------------------------
                    # 5. Compute L1 distance
                    # ----------------------------------------
                    t0 = time.perf_counter()

                    l1_distance = self._compute_l1_distance(
                        ground_speed=ground_speed
                    )

                    l1_time = l1_distance / ground_speed

                    t_l1_distance = (
                        time.perf_counter() - t0
                    )

                    # ----------------------------------------
                    # 6. Update the cross-track integrator
                    # ----------------------------------------
                    t0 = time.perf_counter()

                    (
                        nu1_rad,
                        xtrack_i_rad,
                        integral_active,
                    ) = self._update_cross_track_integrator(
                        cross_track_error_m=cross_track_error,
                        l1_distance_m=l1_distance,
                        expected_loop_period_s=target_dt,
                        allow_update=True,
                    )

                    t_xtrack = time.perf_counter() - t0

                    # ----------------------------------------
                    # 7. Find the base L1 lookahead point
                    # ----------------------------------------
                    t0 = time.perf_counter()

                    l1_point, _ = (
                        find_l1_point_by_straight_distance(
                            closest_index=closest_index,
                            x_path=x_track,
                            y_path=y_track,
                            l1_distance_m=l1_distance,
                        )
                    )

                    l1_point = np.asarray(
                        l1_point,
                        dtype=float,
                    )

                    if (
                        l1_point.shape != (2,)
                        or not np.all(np.isfinite(l1_point))
                    ):
                        raise ValueError(
                            "The computed L1 point is invalid."
                        )

                    t_l1_point = (
                        time.perf_counter() - t0
                    )

                    # ----------------------------------------
                    # 8. Apply cross-track integral
                    # ----------------------------------------
                    #
                    # Positive cross-track error means the aircraft
                    # is left of the path. A positive integral
                    # therefore rotates the aircraft-to-target vector
                    # clockwise, producing a stronger rightward
                    # correction.
                    l1_point_corrected = (
                        apply_cross_track_integral_to_l1_point(
                            aircraft_position_xy=(
                                aircraft_position
                            ),
                            l1_point_xy=l1_point,
                            xtrack_i_rad=xtrack_i_rad,
                        )
                    )

                    aircraft_to_l1_distance = float(
                        np.linalg.norm(
                            l1_point - aircraft_position
                        )
                    )

                    # Positive means an equivalent rightward
                    # displacement.
                    integral_lateral_offset_m = float(
                        aircraft_to_l1_distance
                        * np.sin(xtrack_i_rad)
                    )

                    # ----------------------------------------
                    # 9. Compute desired bank
                    # ----------------------------------------
                    t0 = time.perf_counter()

                    if segment_index == 0:
                        desired_roll_unclipped, solver_status = (
                            self._compute_desired_bank(
                                state=state,
                                l1_point=l1_point_corrected,
                                ground_speed=ground_speed,
                                ground_track=ground_track,
                                l1_distance=l1_distance,
                                t_max=max(l1_time * 50.0, target_dt),
                            )
                        )
                    else:
                         desired_roll_unclipped, solver_status = (
                            self._compute_desired_bank(
                                state=state,
                                l1_point=l1_point_corrected,
                                ground_speed=ground_speed,
                                ground_track=ground_track,
                                l1_distance=l1_distance,
                                t_max=max(l1_time * 5.0, target_dt),
                            )
                        )                       

                    if not np.isfinite(desired_roll_unclipped):
                        desired_roll_unclipped = 0.0
                        solver_status += "_invalid_bank_zeroed"

                    # If the commanded bank genuinely reverses direction, the old
                    # integral belongs to the previous turn direction. Clear it and
                    # recompute the command using the original, uncorrected L1 point.
                    bank_direction_reversed = (
                        self._check_bank_direction_reversal(
                            desired_roll_unclipped
                        )
                    )

                    if bank_direction_reversed and abs(self.xtrack_i) > 0.0:
                        self._clear_cross_track_integral()

                        xtrack_i_rad = 0.0
                        integral_active = False
                        integral_lateral_offset_m = 0.0
                        l1_point_corrected = l1_point.copy()

                        desired_roll_unclipped, recompute_status = (
                            self._compute_desired_bank(
                                state=state,
                                l1_point=l1_point_corrected,
                                ground_speed=ground_speed,
                                ground_track=ground_track,
                                l1_distance=l1_distance,
                                t_max=max(l1_time * 5.0, target_dt),
                            )
                        )

                        solver_status = "converged"

                        if not np.isfinite(desired_roll_unclipped):
                            desired_roll_unclipped = 0.0
                            solver_status += "_invalid_bank_zeroed"
        
                    t_desired_bank = (
                        time.perf_counter() - t0
                    )

                    # ----------------------------------------
                    # 10. Clip desired roll
                    # ----------------------------------------
                    t0 = time.perf_counter()

                    if not np.isfinite(
                        desired_roll_unclipped
                    ):
                        desired_roll_unclipped = 0.0
                        solver_status += "_invalid_bank_zeroed"

                    desired_roll = float(
                        np.clip(
                            desired_roll_unclipped,
                            -self.aircraft.max_bank_rad,
                            self.aircraft.max_bank_rad,
                        )
                    )

                    t_clip = time.perf_counter() - t0

                    # ----------------------------------------
                    # 11. Convert roll to PWM
                    # ----------------------------------------
                    t0 = time.perf_counter()

                    roll_pwm = roll_to_pwm(
                        desired_roll_rad=desired_roll,
                        max_roll_rad=(
                            self.aircraft.max_bank_rad
                        ),
                        pwm_min=roll_limits["min"],
                        pwm_max=roll_limits["max"],
                    )

                    t_pwm = time.perf_counter() - t0

                    # ----------------------------------------
                    # 12. Send RC override
                    # ----------------------------------------
                    t0 = time.perf_counter()

                    self.mav.send_rc_override(
                        roll=roll_pwm,
                        throttle=throttle_pwm,
                        rc_map=rc_map,
                    )

                    t_send = time.perf_counter() - t0

                    runtime_before_print = (
                        time.perf_counter() - loop_t0
                    )

                    frequency_before_print = (
                        1.0 / runtime_before_print
                        if runtime_before_print > 1.0e-9
                        else float("inf")
                    )

                    # ----------------------------------------
                    # 13. Print controller state and timing
                    # ----------------------------------------
                    t0 = time.perf_counter()

                    if solver_status=="converged":
                        succ_count += 1
                    elif solver_status=="suboptimal_converged":
                        subopt_count += 1
                    else:
                        fail_count += 1
                        print(solver_status)
                    
                    comp_cost += t_desired_bank
                            
                    # ~ print(
                        # ~ f"seg={segment_index:02d} "
                        # ~ f"idx={closest_index:04d}/"
                        # ~ f"{stop_index:04d} "
                        # ~ f"xtrack={cross_track_error:7.2f}m "
                        # ~ f"nu1={np.rad2deg(nu1_rad):6.2f}deg "
                        # ~ f"xtrack_i="
                        # ~ f"{np.rad2deg(xtrack_i_rad):6.2f}deg "
                        # ~ f"i_active={int(integral_active)} "
                        # ~ f"solver={solver_status} "
                        # ~ f"cmd_roll="
                        # ~ f"{np.rad2deg(desired_roll):6.1f}deg "
                        # ~ f"actual_roll="
                        # ~ f"{np.rad2deg(state.roll_rad):6.1f}deg "
                        # ~ f"freq_no_sleep="
                        # ~ f"{frequency_before_print:5.1f}Hz "
                        # ~ f"dt_no_sleep="
                        # ~ f"{runtime_before_print * 1000.0:7.1f}ms"
                    # ~ )

                    # ~ print(
                        # ~ "TIMING ms | "
                        # ~ f"max_check="
                        # ~ f"{t_max_check * 1000.0:7.2f} "
                        # ~ f"state={t_state * 1000.0:7.2f} "
                        # ~ f"vel={t_velocity * 1000.0:7.2f} "
                        # ~ f"closest="
                        # ~ f"{t_closest * 1000.0:7.2f} "
                        # ~ f"xtrack="
                        # ~ f"{t_xtrack * 1000.0:7.2f} "
                        # ~ f"l1_dist="
                        # ~ f"{t_l1_distance * 1000.0:7.2f} "
                        # ~ f"l1_point="
                        # ~ f"{t_l1_point * 1000.0:7.2f} "
                        # ~ f"bank="
                        # ~ f"{t_desired_bank * 1000.0:7.2f} "
                        # ~ f"clip={t_clip * 1000.0:7.2f} "
                        # ~ f"pwm={t_pwm * 1000.0:7.2f} "
                        # ~ f"send={t_send * 1000.0:7.2f}"
                    # ~ )

                    t_print = time.perf_counter() - t0

                    iteration += 1

                    # ----------------------------------------
                    # 14. Maintain the requested loop rate
                    # ----------------------------------------
                    t0 = time.perf_counter()

                    now = time.perf_counter()
                    sleep_s = max(
                        0.0,
                        next_loop_deadline - now,
                    )

                    if sleep_s > 0.0:
                        time.sleep(sleep_s)

                    t_sleep_actual = (
                        time.perf_counter() - t0
                    )
                    loop_dt_total = (
                        time.perf_counter() - loop_t0
                    )

                    runtime_before_sleep = (
                        loop_dt_total - t_sleep_actual
                    )

                    loop_utilization_pct = (
                        100.0
                        * runtime_before_sleep
                        / target_dt
                    )

                    frequency_total = (
                        1.0 / loop_dt_total
                        if loop_dt_total > 1.0e-9
                        else float("inf")
                    )

                    # ~ print(
                        # ~ "LOOP ms   | "
                        # ~ f"print={t_print * 1000.0:7.2f} "
                        # ~ f"sleep_cmd="
                        # ~ f"{sleep_s * 1000.0:7.2f} "
                        # ~ f"sleep_actual="
                        # ~ f"{t_sleep_actual * 1000.0:7.2f} "
                        # ~ f"total="
                        # ~ f"{loop_dt_total * 1000.0:7.2f} "
                        # ~ f"freq_total={frequency_total:5.1f}Hz"
                    # ~ )
                    # ~ print("-" * 140)

                    timestamp_utc = datetime.now(
                        timezone.utc
                    ).isoformat()

                    csv_writer.writerow(
                        {
                            "timestamp_utc": timestamp_utc,
                            "iteration": iteration,
                            "segment": segment_index,
                            "closest_index": closest_index,
                            "stop_index": stop_index,
                            "position_x_m": float(
                                aircraft_position[0]
                            ),
                            "position_y_m": float(
                                aircraft_position[1]
                            ),
                            "cross_track_error_m": float(
                                cross_track_error
                            ),
                            "cross_track_error_abs_m": float(
                                abs(cross_track_error)
                            ),
                            "nu1_deg": float(
                                np.rad2deg(nu1_rad)
                            ),
                            "xtrack_i_deg": float(
                                np.rad2deg(xtrack_i_rad)
                            ),
                            "integral_active": int(
                                integral_active
                            ),
                            "integral_lateral_offset_m": (
                                integral_lateral_offset_m
                            ),
                            "heading_deg": float(
                                np.rad2deg(
                                    state.heading_rad
                                )
                            ),
                            "ground_track_deg": float(
                                np.rad2deg(ground_track)
                            ),
                            "ground_speed_mps": ground_speed,
                            "airspeed_mps": float(
                                self.aircraft.airspeed_mps
                            ),
                            "l1_distance_m": l1_distance,
                            "l1_time_s": l1_time,
                            "desired_roll_deg": float(
                                np.rad2deg(desired_roll)
                            ),
                            "actual_roll_deg": float(
                                np.rad2deg(state.roll_rad)
                            ),
                            "roll_pwm": roll_pwm,
                            "throttle_pwm": throttle_pwm,
                            "solver_status": solver_status,
                            "runtime_ms": (
                                runtime_before_sleep * 1000.0
                            ),
                            "target_period_ms": (
                                target_dt * 1000.0
                            ),
                            "loop_utilization_pct": (
                                loop_utilization_pct
                            ),
                            "frequency_hz": frequency_total,
                            "max_check_ms": (
                                t_max_check * 1000.0
                            ),
                            "state_ms": t_state * 1000.0,
                            "velocity_ms": (
                                t_velocity * 1000.0
                            ),
                            "closest_ms": (
                                t_closest * 1000.0
                            ),
                            "l1_distance_ms": (
                                t_l1_distance * 1000.0
                            ),
                            "xtrack_ms": (
                                t_xtrack * 1000.0
                            ),
                            "l1_point_ms": (
                                t_l1_point * 1000.0
                            ),
                            "bank_ms": (
                                t_desired_bank * 1000.0
                            ),
                            "clip_ms": t_clip * 1000.0,
                            "pwm_ms": t_pwm * 1000.0,
                            "send_ms": t_send * 1000.0,
                            "print_ms": t_print * 1000.0,
                            "sleep_commanded_ms": (
                                sleep_s * 1000.0
                            ),
                            "sleep_actual_ms": (
                                t_sleep_actual * 1000.0
                            ),
                            "total_loop_ms": (
                                loop_dt_total * 1000.0
                            ),
                        }
                    )

                    log_file.flush()

                    # Advance the absolute loop deadline. This allows
                    # minor printing/logging overhead to be recovered
                    # during the next cycle instead of accumulating
                    # permanently.
                    next_loop_deadline += target_dt

                    current_time = time.perf_counter()

                    # If the loop is more than one full period behind,
                    # abandon catch-up and resynchronize. This prevents
                    # repeated zero-sleep iterations.
                    if (
                        current_time - next_loop_deadline
                        > target_dt
                    ):
                        next_loop_deadline = (
                            current_time + target_dt
                        )
                
                count = succ_count + fail_count + subopt_count
                print (succ_count,"/",subopt_count,"/",fail_count,"/",count,"  computational cost: ",comp_cost*1000/count)
                if timed_out:
                    break

        if timed_out:
            print("Follower max_time_s reached.")
        else:
            print("Path follower complete.")

    def _configure_mavlink_message_rates(
        self,
        message_rate_hz: float,
    ) -> None:
        """Configure the MAVLink messages used by the controller."""
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

    @staticmethod
    def _max_time_reached(
        start_time: float,
        max_time_s: float,
    ) -> bool:
        return (
            time.perf_counter() - start_time
            >= max_time_s
        )

    def _reset_cross_track_integrator(self) -> None:
        self.xtrack_i = 0.0
        self._xtrack_i_time = None
        self._last_xtrack_i_gain = self.xtrack_i_gain
        self._last_bank_direction = 0

    def _get_bank_direction(self, bank_rad: float) -> int:
        """
        Convert a bank command into a stable direction indicator.

        Returns:
            -1: meaningful negative bank
             0: inside the zero-bank deadband
            +1: meaningful positive bank
        """
        bank_rad = float(bank_rad)

        if not np.isfinite(bank_rad):
            return 0

        if bank_rad > self.bank_reversal_threshold_rad:
            return 1

        if bank_rad < -self.bank_reversal_threshold_rad:
            return -1

        return 0

    def _check_bank_direction_reversal(
        self,
        desired_bank_rad: float,
    ) -> bool:
        """
        Detect a meaningful left/right bank-command reversal.

        Commands inside the deadband do not replace the last meaningful
        direction. This prevents noise near zero bank from repeatedly
        resetting the integrator.
        """
        current_direction = self._get_bank_direction(
            desired_bank_rad
        )

        if current_direction == 0:
            return False

        reversed_direction = (
            self._last_bank_direction != 0
            and current_direction != self._last_bank_direction
        )

        self._last_bank_direction = current_direction
        return reversed_direction


    def _clear_cross_track_integral(self) -> None:
        """
        Clear the accumulated correction without resetting the integrator
        clock. Keeping the clock avoids an unnecessary skipped update on
        the next control iteration.
        """
        self.xtrack_i = 0.0
        
    def _prepare_segment(
        self,
        subarrays: list[tuple[np.ndarray, np.ndarray]],
        segment_index: int,
        x_seg_raw: np.ndarray,
        y_seg_raw: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Validate and prepare one path segment."""
        x_raw = np.asarray(x_seg_raw, dtype=float)
        y_raw = np.asarray(y_seg_raw, dtype=float)

        if x_raw.ndim != 1 or y_raw.ndim != 1:
            raise ValueError(
                f"Segment {segment_index} must use 1-D arrays."
            )

        if len(x_raw) != len(y_raw):
            raise ValueError(
                f"Segment {segment_index} has different x/y "
                f"lengths: {len(x_raw)} and {len(y_raw)}."
            )

        if len(x_raw) < 2:
            raise ValueError(
                f"Segment {segment_index} must contain at "
                "least two points."
            )

        if (
            not np.all(np.isfinite(x_raw))
            or not np.all(np.isfinite(y_raw))
        ):
            raise ValueError(
                f"Segment {segment_index} contains non-finite "
                "coordinates."
            )

        if len(x_raw) == 2:
            x_seg, y_seg = fill_sparse_points(
                (x_raw, y_raw)
            )
            x_track, y_track = extrapolate_end(
                (x_seg, y_seg)
            )
        else:
            x_seg = x_raw.copy()
            y_seg = y_raw.copy()
            x_track, y_track = stack_with_next_segment(
                subarrays,
                segment_index,
                x_seg,
                y_seg,
            )

        x_seg = np.asarray(x_seg, dtype=float)
        y_seg = np.asarray(y_seg, dtype=float)
        x_track = np.asarray(x_track, dtype=float)
        y_track = np.asarray(y_track, dtype=float)

        if len(x_seg) != len(y_seg):
            raise ValueError(
                "Prepared segment has different x/y lengths."
            )

        if len(x_track) != len(y_track):
            raise ValueError(
                "Prepared tracking path has different x/y "
                "lengths."
            )

        if len(x_track) < 2:
            raise ValueError(
                "Prepared tracking path must contain at least "
                "two points."
            )

        if (
            not np.all(np.isfinite(x_track))
            or not np.all(np.isfinite(y_track))
        ):
            raise ValueError(
                "Prepared tracking path contains non-finite "
                "coordinates."
            )

        return x_seg, y_seg, x_track, y_track

    def _compute_l1_distance(
        self,
        ground_speed: float,
    ) -> float:
        """Compute and validate the current L1 distance."""
        if self.use_wind_aware_bank_solver:
            l1_circular = compute_l1_circular_distance(
                ground_speed_mps=ground_speed,
                period_s=self.l1.period_s,
                max_bank_rad=self.aircraft.max_bank_rad,
            )

            l1_reference = compute_l1_circular_distance(
                ground_speed_mps=self.aircraft.airspeed_mps,
                period_s=self.l1.period_s,
                max_bank_rad=self.aircraft.max_bank_rad,
            )

            l1_distance = min(
                float(l1_circular),
                float(l1_reference),
            )
        else:
            l1_distance = float(
                ardupilot_l1_distance(
                    ground_speed_mps=ground_speed,
                    damping=self.l1.damping,
                    period_s=self.l1.period_s,
                )
            )

        if (
            not np.isfinite(l1_distance)
            or l1_distance <= 0.0
        ):
            raise ValueError(
                f"Invalid L1 distance: {l1_distance!r}"
            )

        return max(l1_distance, self.aircraft.airspeed_mps)

    def _update_cross_track_integrator(
        self,
        cross_track_error_m: float,
        l1_distance_m: float,
        expected_loop_period_s: float,
        allow_update: bool = True,
    ) -> tuple[float, float, bool]:
        """
        Update the cross-track guidance-angle integrator.

        The integration uses measured elapsed time rather than an
        assumed fixed update period. Consequently, the integral gain
        has the same per-second meaning at 4, 5, 10, or another loop
        frequency.

        Positive cross-track error means the aircraft is left of the
        path. Positive xtrack_i produces a clockwise/rightward target
        correction.

        Returns:
            nu1_rad
            xtrack_i_rad
            integrator_active
        """
        now = time.perf_counter()

        if self._xtrack_i_time is None:
            dt_s = 0.0
        else:
            dt_s = now - self._xtrack_i_time

        self._xtrack_i_time = now

        expected_loop_period_s = float(
            expected_loop_period_s
        )

        if (
            not np.isfinite(expected_loop_period_s)
            or expected_loop_period_s <= 0.0
        ):
            expected_loop_period_s = 0.2

        # The timeout is rate-aware. Five missed nominal updates or
        # one second, whichever is greater, is treated as a loss of
        # controller continuity.
        continuity_timeout_s = max(
            1.0,
            5.0 * expected_loop_period_s,
        )

        if (
            not np.isfinite(dt_s)
            or dt_s < 0.0
            or dt_s > continuity_timeout_s
        ):
            self.xtrack_i = 0.0
            dt_s = 0.0

        cross_track_error_m = float(
            cross_track_error_m
        )
        l1_distance_m = float(l1_distance_m)

        if (
            not np.isfinite(cross_track_error_m)
            or not np.isfinite(l1_distance_m)
            or l1_distance_m <= 0.0
        ):
            self.xtrack_i = 0.0
            return 0.0, self.xtrack_i, False

        l1_distance_m = max(l1_distance_m, 0.1)

        nu1_rad = float(
            np.arcsin(
                np.clip(
                    cross_track_error_m / l1_distance_m,
                    -0.7071,
                    0.7071,
                )
            )
        )

        current_gain = float(self.xtrack_i_gain)

        gain_changed = (
            not np.isfinite(current_gain)
            or not np.isclose(
                current_gain,
                self._last_xtrack_i_gain,
                rtol=0.0,
                atol=1.0e-12,
            )
        )

        if current_gain <= 0.0 or gain_changed:
            self.xtrack_i = 0.0

            if np.isfinite(current_gain):
                self._last_xtrack_i_gain = current_gain

            return nu1_rad, self.xtrack_i, False

        integrator_active = (
            bool(allow_update)
            and dt_s > 0.0
            and abs(nu1_rad)
            < self.xtrack_i_enable_angle
        )

        if integrator_active:
            # Do not clamp dt to a fixed 0.1 or 0.25 seconds. Using
            # the measured elapsed time makes the integrator
            # independent of the requested controller frequency.
            self.xtrack_i += (
                nu1_rad
                * current_gain
                * dt_s
            )

            self.xtrack_i = float(
                np.clip(
                    self.xtrack_i,
                    -self.xtrack_i_limit,
                    self.xtrack_i_limit,
                )
            )

        return (
            nu1_rad,
            self.xtrack_i,
            integrator_active,
        )

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
        Try the wind-aware bank solver and fall back to standard L1.
        """
        fallback_status = "fallback_disabled"
        if self.use_wind_aware_bank_solver:
            try:
                bank, info = (
                    desired_bank_to_point_with_wind_roll(
                        x0=float(state.position_utm[0]),
                        y0=float(state.position_utm[1]),
                        psi0=float(state.heading_rad),
                        phi0=float(state.roll_rad),
                        xt=float(l1_point[0]),
                        yt=float(l1_point[1]),
                        g=9.81,
                        V_TAS=float(
                            self.aircraft.airspeed_mps
                        ),
                        V_w=float(
                            self.l1.wind_speed_mps
                        ),
                        theta_wa=float(
                            self.l1.wind_from_direction_rad
                        ),
                        p_max_roll=float(
                            self.aircraft.roll_rate_rad_s
                        ),
                        t_max=float(t_max),
                        dt=float(self.l1.dt_solver),
                        phi_max=float(
                            self.aircraft.max_bank_rad
                        ),
                        tol_pos=float(self.l1.tol_pos),
                        max_iter=int(self.l1.max_iter),
                    )
                )

                converged = bool(info.get("converged", False))
                reason = str(
                    info.get(
                        "reason",
                        info.get("status", ""),
                    )
                ).lower()

                # A suboptimal solution may be valid but constrained by phi_max.
                # Re-run with a nearly unconstrained bank angle to verify direction.
                if not converged and reason == "suboptimal":
                    verification_bank, verification_info = (
                        desired_bank_to_point_with_wind_roll(
                            x0=float(state.position_utm[0]),
                            y0=float(state.position_utm[1]),
                            psi0=float(state.heading_rad),
                            phi0=float(state.roll_rad),
                            xt=float(l1_point[0]),
                            yt=float(l1_point[1]),
                            g=9.81,
                            V_TAS=float(
                                self.aircraft.airspeed_mps
                            ),
                            V_w=float(
                                self.l1.wind_speed_mps
                            ),
                            theta_wa=float(
                                self.l1.wind_from_direction_rad
                            ),
                            p_max_roll=float(
                                self.aircraft.roll_rate_rad_s
                            ),
                            t_max=float(t_max),
                            dt=float(self.l1.dt_solver),
                            phi_max=float(
                                np.deg2rad(89)
                            ),
                            tol_pos=float(self.l1.tol_pos),
                            max_iter=int(self.l1.max_iter),
                        )
                    )

                    bank_is_finite = np.isfinite(bank)
                    verification_is_finite = np.isfinite(
                        verification_bank
                    )

                    # Use a small tolerance so numerical noise around zero is not
                    # interpreted as a meaningful bank direction.
                    direction_tolerance = np.deg2rad(0.1)

                    original_direction = (
                        0
                        if not bank_is_finite
                        or abs(bank) <= direction_tolerance
                        else int(np.sign(bank))
                    )

                    verification_direction = (
                        0
                        if not verification_is_finite
                        or abs(verification_bank) <= direction_tolerance
                        else int(np.sign(verification_bank))
                    )

                    same_direction = (
                        original_direction != 0
                        and verification_direction != 0
                        and original_direction == verification_direction
                    )

                    if same_direction:
                        # Accept the ORIGINAL bank, not the 89-degree verification
                        # solution. The second run is only checking its direction.
                        info = dict(info)
                        info["converged"] = True
                        info["reason"] = "suboptimal_converged"
                        info["original_reason"] = reason
                        info["verification_bank"] = float(
                            verification_bank
                        )
                        info["verification_info"] = verification_info

                        converged = True

                if converged and np.isfinite(bank):
                    return bank, info.get("reason", "converged")

                fallback_reason = info.get(
                    "reason",
                    info.get(
                        "message",
                        info.get("status", "unknown reason"),
                    ),
                )

                print(
                    "Wind-aware bank solver did not converge; "
                    "using L1 fallback. "
                    f"Reason: {fallback_reason}. Info: {info!r}"
                )

            except Exception as exc:
                fallback_status = "fallback_solver_error"
                print(
                    "Wind-aware bank solver error: "
                    f"{exc}; using L1 fallback."
                )

        fallback_bank = float(
            l1_bank_command(
                aircraft_position=state.position_utm,
                ground_track_rad=ground_track,
                l1_point=l1_point,
                ground_speed_mps=ground_speed,
                damping=self.l1.damping,
                l1_distance_m=l1_distance,
            )
        )

        if not np.isfinite(fallback_bank):
            return 0.0, f"{fallback_status}_invalid_zeroed"

        return fallback_bank, fallback_status


def apply_cross_track_integral_to_l1_point(
    aircraft_position_xy: np.ndarray,
    l1_point_xy: np.ndarray,
    xtrack_i_rad: float,
) -> np.ndarray:
    """
    Apply an angular cross-track-integral correction to an L1 point.

    Coordinate convention:
        x = East
        y = North

    Positive xtrack_i means the aircraft has persistently been left
    of the path. The aircraft-to-target vector is therefore rotated
    clockwise, creating a stronger rightward correction.

    Rotation preserves the aircraft-to-target distance.
    """
    aircraft_position_xy = np.asarray(
        aircraft_position_xy,
        dtype=float,
    )
    l1_point_xy = np.asarray(
        l1_point_xy,
        dtype=float,
    )
    xtrack_i_rad = float(xtrack_i_rad)

    if (
        aircraft_position_xy.shape != (2,)
        or l1_point_xy.shape != (2,)
        or not np.all(np.isfinite(aircraft_position_xy))
        or not np.all(np.isfinite(l1_point_xy))
        or not np.isfinite(xtrack_i_rad)
    ):
        return l1_point_xy.copy()

    target_vector = (
        l1_point_xy - aircraft_position_xy
    )
    target_distance = float(
        np.linalg.norm(target_vector)
    )

    if (
        not np.isfinite(target_distance)
        or target_distance < 1.0e-6
    ):
        return l1_point_xy.copy()

    # Clockwise rotation by xtrack_i_rad:
    #
    # R(-angle) = [[ cos(angle),  sin(angle)],
    #              [-sin(angle),  cos(angle)]]
    c = float(np.cos(xtrack_i_rad))
    s = float(np.sin(xtrack_i_rad))

    corrected_vector = np.array(
        [
            c * target_vector[0]
            + s * target_vector[1],
            -s * target_vector[0]
            + c * target_vector[1],
        ],
        dtype=float,
    )

    return aircraft_position_xy + corrected_vector


def signed_cross_track_error_and_normal(
    aircraft_position_xy: np.ndarray,
    x_path: np.ndarray,
    y_path: np.ndarray,
    closest_index: int,
) -> tuple[float, np.ndarray]:
    """
    Return signed cross-track error and the left-hand unit normal.

    Coordinates:
        x = East
        y = North

    Positive cross-track error means the aircraft is left of the
    selected path direction.
    """
    aircraft_position = np.asarray(
        aircraft_position_xy,
        dtype=float,
    )
    x_path = np.asarray(x_path, dtype=float)
    y_path = np.asarray(y_path, dtype=float)

    if (
        aircraft_position.shape != (2,)
        or not np.all(np.isfinite(aircraft_position))
    ):
        return (
            float("nan"),
            np.array([np.nan, np.nan], dtype=float),
        )

    n = min(len(x_path), len(y_path))

    if n < 2:
        return (
            float("nan"),
            np.array([np.nan, np.nan], dtype=float),
        )

    closest_index = int(
        np.clip(closest_index, 0, n - 1)
    )

    candidate_indices: list[int] = []
    seen_indices: set[int] = set()

    # Prefer the forward segment beginning at closest_index, then the
    # previous segment, followed by increasingly distant segments.
    for offset in range(n):
        for candidate in (
            closest_index + offset,
            closest_index - 1 - offset,
        ):
            if (
                0 <= candidate < n - 1
                and candidate not in seen_indices
            ):
                seen_indices.add(candidate)
                candidate_indices.append(candidate)

    selected_segment = None

    for segment_index in candidate_indices:
        p0 = np.array(
            [
                x_path[segment_index],
                y_path[segment_index],
            ],
            dtype=float,
        )
        p1 = np.array(
            [
                x_path[segment_index + 1],
                y_path[segment_index + 1],
            ],
            dtype=float,
        )

        if (
            not np.all(np.isfinite(p0))
            or not np.all(np.isfinite(p1))
        ):
            continue

        segment_vector = p1 - p0
        segment_length = float(
            np.linalg.norm(segment_vector)
        )

        if (
            np.isfinite(segment_length)
            and segment_length > 1.0e-6
        ):
            selected_segment = (
                p0,
                segment_vector,
                segment_length,
            )
            break

    if selected_segment is None:
        return (
            float("nan"),
            np.array([np.nan, np.nan], dtype=float),
        )

    p0, segment_vector, segment_length = (
        selected_segment
    )

    tangent = segment_vector / segment_length

    normal_left = np.array(
        [-tangent[1], tangent[0]],
        dtype=float,
    )

    error_vector = aircraft_position - p0

    cross_track_error = float(
        np.dot(error_vector, normal_left)
    )

    if not np.isfinite(cross_track_error):
        return (
            float("nan"),
            np.array([np.nan, np.nan], dtype=float),
        )

    return cross_track_error, normal_left
