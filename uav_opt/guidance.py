import time
import numpy as np
from pymavlink import mavutil

from uav_opt.config import AircraftConfig, L1Config
from uav_opt.mavlink_client import (
    MavlinkClient,
    roll_to_pwm,
    airspeed_to_throttle_pwm,
)
from uav_opt.simulator import compute_l1_circular_distance, l1_bank_command
from uav_opt.wind import wind_correction,  wind_to_xy_velocity, air_and_ground_velocity_xy
from uav_opt.path_utils import (
    fill_sparse_points,
    extrapolate_end,
    find_closest_point,
    find_l1_point_by_straight_distance,
    find_l1_point_by_path_distance,
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

    def follow(
        self,
        subarrays: list[tuple[np.ndarray, np.ndarray]],
        max_time_s: float = 500.0,
        message_rate_hz: float = 10.0,
    ) -> None:
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

        throttle_pwm = airspeed_to_throttle_pwm(
            target_airspeed_mps=self.aircraft.airspeed_mps,
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

            succ_count = 0
            fail_count = 0
            segment_stats = {
                "iterations": 0,

                "state": 0.0,
                "velocity": 0.0,
                "closest": 0.0,
                "l1_distance": 0.0,
                "l1_point": 0.0,
                "desired_bank": 0.0,
                "clip": 0.0,
                "pwm": 0.0,
                "send": 0.0,

                "execution": 0.0,
                "sleep_actual": 0.0,
                "total": 0.0,

                "freq_no_sleep": 0.0,
                "freq_total": 0.0,

                "cmd_roll_deg": 0.0,
                "actual_roll_deg": 0.0,
            }

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
            stop_index = max(len(x_seg) - 8, 1)

            while closest_index < stop_index:
                loop_t0 = time.perf_counter()

                t0 = time.perf_counter()
                if time.time() - start_time > max_time_s:
                    print("Follower max_time_s reached inside segment.")
                    return
                t_max_check = time.perf_counter() - t0

                t0 = time.perf_counter()
                state = self.mav.aircraft_state(timeout=2.0)
                t_state = time.perf_counter() - t0

                t0 = time.perf_counter()
                air_velocity, ground_velocity, ground_speed, ground_track = air_and_ground_velocity_xy(
                    airspeed_mps=self.aircraft.airspeed_mps,
                    heading_rad=state.heading_rad,
                    wind_speed_mps=self.l1.wind_speed_mps,
                    wind_from_direction_rad=self.l1.wind_from_direction_rad,
                )
                ground_speed = max(float(ground_speed), 0.1)
                t_velocity = time.perf_counter() - t0

                t0 = time.perf_counter()
                closest_point, closest_index = find_closest_point(
                    aircraft_position_xy=state.position_utm,
                    x_path=x_track,
                    y_path=y_track,
                    previous_index=closest_index,
                )
                t_closest = time.perf_counter() - t0

                t0 = time.perf_counter()
                if self.l1.use_wind_aware_bank_solver:
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

                    l1_time = l1_distance / ground_speed
                else:
                    l1_distance = ardupilot_l1_distance(
                        ground_speed_mps=ground_speed,
                        damping=self.l1.damping,
                        period_s=self.l1.period_s,
                    )
                    l1_time = l1_distance / ground_speed

                t_l1_distance = time.perf_counter() - t0

                t0 = time.perf_counter()
                l1_point, l1_index = find_l1_point_by_path_distance(
                    closest_index=closest_index,
                    x_path=x_track,
                    y_path=y_track,
                    l1_distance_m=l1_distance,
                )
                t_l1_point = time.perf_counter() - t0

                t0 = time.perf_counter()
                desired_roll, wind_aware = self._compute_desired_bank(
                    state=state,
                    l1_point=l1_point,
                    ground_speed=ground_speed,
                    ground_track=ground_track,
                    l1_distance=l1_distance,
                    t_max=l1_time * 20, #Ensuring enough time
                )
                t_desired_bank = time.perf_counter() - t0

                if wind_aware:
                    succ_count += 1
                else:
                    fail_count += 1
                    
                t0 = time.perf_counter()
                desired_roll = float(
                    np.clip(
                        desired_roll,
                        -self.aircraft.max_bank_rad,
                        self.aircraft.max_bank_rad,
                    )
                )
                t_clip = time.perf_counter() - t0

                t0 = time.perf_counter()
                roll_pwm = roll_to_pwm(
                    desired_roll_rad=desired_roll,
                    max_roll_rad=self.aircraft.max_bank_rad,
                    pwm_min=roll_limits["min"],
                    pwm_max=roll_limits["max"],
                )
                t_pwm = time.perf_counter() - t0

                t0 = time.perf_counter()
                self.mav.send_rc_override(
                    roll=roll_pwm,
                    throttle=throttle_pwm,
                    rc_map=rc_map,
                )
                t_send = time.perf_counter() - t0

                loop_dt_before_sleep = time.perf_counter() - loop_t0
                freq_before_sleep = (
                    1.0 / loop_dt_before_sleep
                    if loop_dt_before_sleep > 1e-6
                    else float("inf")
                )

                t_exec = (
                    t_velocity
                    + t_closest
                    + t_l1_distance
                    + t_l1_point
                    + t_desired_bank
                    + t_clip
                    + t_pwm
                    + t_send
                )

                iteration += 1

                target_dt = 1.0 / message_rate_hz
                elapsed_before_sleep = time.perf_counter() - loop_t0
                sleep_s = max(0.0, target_dt - elapsed_before_sleep)

                t0 = time.perf_counter()
                time.sleep(sleep_s)
                t_sleep_actual = time.perf_counter() - t0

                loop_dt_total = time.perf_counter() - loop_t0
                freq_total = (
                    1.0 / loop_dt_total
                    if loop_dt_total > 1e-6
                    else float("inf")
                )

                segment_stats["iterations"] += 1

                segment_stats["state"] += t_state
                segment_stats["velocity"] += t_velocity
                segment_stats["closest"] += t_closest
                segment_stats["l1_distance"] += t_l1_distance
                segment_stats["l1_point"] += t_l1_point
                segment_stats["desired_bank"] += t_desired_bank
                segment_stats["clip"] += t_clip
                segment_stats["pwm"] += t_pwm
                segment_stats["send"] += t_send

                segment_stats["execution"] += t_exec
                segment_stats["sleep_actual"] += t_sleep_actual
                segment_stats["total"] += loop_dt_total

                segment_stats["freq_no_sleep"] += freq_before_sleep
                segment_stats["freq_total"] += freq_total

                segment_stats["cmd_roll_deg"] += np.rad2deg(desired_roll)
                segment_stats["actual_roll_deg"] += np.rad2deg(state.roll_rad)

            n = segment_stats["iterations"]

            if n > 0:
                print("-" * 100)
                print(f"SEGMENT {segment_index + 1}/{len(subarrays)} COMPLETE")
                print(
                    f"iterations={n} "
                    f"final_idx={closest_index:04d}/{stop_index:04d}"
                    f"Wind-aware counter (success/fail): {succ_count}, {fail_count}"
                )

                print(
                    f"ROLL avg | "
                    f"cmd_roll={segment_stats['cmd_roll_deg'] / n:6.1f}deg "
                    f"actual_roll={segment_stats['actual_roll_deg'] / n:6.1f}deg"
                )

                print(
                    "TIMING avg ms | "
                    f"state={segment_stats['state'] / n * 1000:7.2f} "
                    f"velocity={segment_stats['velocity'] / n * 1000:7.2f} "
                    f"closest={segment_stats['closest'] / n * 1000:7.2f} "
                    f"l1_dist={segment_stats['l1_distance'] / n * 1000:7.2f} "
                    f"l1_point={segment_stats['l1_point'] / n * 1000:7.2f} "
                    f"solver={segment_stats['desired_bank'] / n * 1000:7.2f} "
                    f"clip={segment_stats['clip'] / n * 1000:7.2f} "
                    f"pwm={segment_stats['pwm'] / n * 1000:7.2f} "
                    f"send={segment_stats['send'] / n * 1000:7.2f}"
                )

                print(
                    "LOOP avg | "
                    f"retrieving_state={segment_stats['state'] / n * 1000:7.2f} "
                    f"execution={segment_stats['execution'] / n * 1000:7.2f}ms "
                    f"sleep_actual={segment_stats['sleep_actual'] / n * 1000:7.2f}ms "
                    f"total={segment_stats['total'] / n * 1000:7.2f}ms "
                    f"freq_no_sleep={segment_stats['freq_no_sleep'] / n:5.1f}Hz "
                    f"freq_total={segment_stats['freq_total'] / n:5.1f}Hz"
                )
                print("-" * 100)

        print("Path follower complete.")

    def _compute_desired_bank(
        self,
        state,
        l1_point: np.ndarray,
        ground_speed: float,
        ground_track: float,
        l1_distance: float,
        t_max: float,
    ) -> tuple[float, bool]:
        """
        Try wind-aware bank solver; fall back to standard L1 bank.
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
                    return float(bank), True

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
        ), False
