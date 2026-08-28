#!/usr/bin/env python3

"""
Main SITL entry point.

Flow:
1. Connect to ArduPilot.
2. Download mission.
3. Save mission to temp.waypoints.
4. Parse mission waypoints.
5. Convert waypoints into local/north-aligned frame.
6. Compute optimal maneuver path.
7. Simulate internally and show plot.
8. Ask user confirmation.
9. Arm/takeoff/transition/climb or resume AUTO.
10. Execute online guidance in FBWB using RC overrides.
11. Resume landing sequence or QRTL.
"""

from pathlib import Path
from dataclasses import replace
from pymavlink import mavutil
import time

from uav_opt.angles import wrap_2pi
from uav_opt.config import AppConfig
from uav_opt.mavlink_client import MavlinkClient
from uav_opt.mission_io import save_mavlink_mission, load_navigation_plan_from_wpl
from uav_opt.path_utils import export_path_to_csv
from uav_opt.planner import OptimalPathPlanner
from uav_opt.simulator import (
    simulate_l1_path,
    simulate_conventional_path,
    plot_plan_and_simulation,
)
from uav_opt.guidance import SITLPathFollower


def main() -> None:
    cfg = AppConfig.default()

    temp_mission_path = Path(cfg.execution.temp_mission_file)

    print("========== UAV Optimal Maneuver SITL ==========")

    # ---------------------------------------------------------------------
    # 1. Connect
    # ---------------------------------------------------------------------
    mav = MavlinkClient.connect(cfg.execution.connection_string)

    # ---------------------------------------------------------------------
    # 1.5. Apply airspeed correction
    # ---------------------------------------------------------------------
    # Read EAS-to-TAS ratio from barometric pressure and temperature.
    e2t = mav.get_e2t(timeout=2.0)
 
    original_airspeed_mps = cfg.aircraft.airspeed_mps
    adjusted_airspeed_mps = original_airspeed_mps * e2t

    print(f"EAS2TAS ratio: {e2t:.6f}")
    print(f"Configured airspeed: {original_airspeed_mps:.3f} m/s")
    print(f"Adjusted airspeed: {adjusted_airspeed_mps:.3f} m/s")

    # AircraftConfig is frozen, so create a modified copy.
    adjusted_aircraft = replace(
        cfg.aircraft,
        airspeed_mps=adjusted_airspeed_mps,
    )

    # Replace the aircraft configuration inside the application config.
    cfg = replace(
        cfg,
        aircraft=adjusted_aircraft,
    )

    # ---------------------------------------------------------------------
    # 2. Download mission from autopilot and save it
    # ---------------------------------------------------------------------
    mission_items = mav.download_mission()
    if not mission_items:
        raise RuntimeError("Autopilot mission is empty or could not be downloaded.")

    mav.ensure_home_matches_mission(mission_items)

    save_mavlink_mission(mission_items, temp_mission_path)
    print(f"Mission saved to: {temp_mission_path}")

    # ---------------------------------------------------------------------
    # 3. Parse navigation waypoints
    # ---------------------------------------------------------------------
    plan = load_navigation_plan_from_wpl(temp_mission_path)
    if len(plan.waypoints_utm) < 2:
        raise RuntimeError("Need at least two NAV_WAYPOINT items for path planning.")

    print(f"Loaded {len(plan.waypoints_utm)} navigation waypoints.")
    print(f"Start mission seq: {plan.start_seq}")
    print(f"Landing sequence available: {plan.do_land}, land_seq={plan.land_seq}")

    print("Planner inputs:")
    print(f"  airspeed_mps = {cfg.aircraft.airspeed_mps}")
    print(f"  wind speed = {cfg.l1.wind_speed_mps}")
    print(f"  wind direction = {cfg.l1.wind_from_direction_rad}")
    print(
        "  wind-aware solver =",
        cfg.l1.use_wind_aware_bank_solver,
    )

    # ---------------------------------------------------------------------
    # 4. Plan optimal path
    # ---------------------------------------------------------------------
    planner = OptimalPathPlanner(cfg.aircraft, cfg.l1)
    planned_path = planner.plan(plan.waypoints_utm)
    optimal_path_csv = Path("optimal_path.csv")

    export_path_to_csv(
        path_xy=planned_path.optimal_path_utm,
        output_file=optimal_path_csv,
        path_name="optimal",
    )

    print(f"Optimal path points: {len(planned_path.optimal_path_utm)}")
    print(f"Path split into {len(planned_path.subarrays)} segments.")

    # ---------------------------------------------------------------------
    # 5. Internal simulation and preview
    # ---------------------------------------------------------------------
    if cfg.execution.evaluate_before_flight:
        # -------------------------------------------------------------
        # Conventional waypoint-following simulation.
        #
        # This is now a MAIN simulator function, not a legacy wrapper.
        # It runs in the north-aligned planning frame and is converted
        # back to UTM only for plotting.
        # -------------------------------------------------------------
        l1_north_frame = replace(
            cfg.l1,
            wind_from_direction_rad=wrap_2pi(
                cfg.l1.wind_from_direction_rad
                - planned_path.local_frame.flight_path_angle_rad
            ),
        )
        
        conventional_dyn_north = simulate_conventional_path(
            waypoints=planned_path.original_waypoints_north,
            initial_heading_rad=planned_path.initial_heading_rad
            - planned_path.local_frame.flight_path_angle_rad,
            aircraft=cfg.aircraft,
            l1=l1_north_frame,
            sim_time_s=cfg.execution.internal_sim_time_s,
            overshoot_m=70,
            leadin_m=70,
        )

        conventional_dyn_utm = conventional_dyn_north.copy()

        if len(conventional_dyn_utm) > 0:
            conventional_xy_utm = planned_path.local_frame.from_north_frame(
                conventional_dyn_north[:, 1:3]
            )
            conventional_dyn_utm[:, 1:3] = conventional_xy_utm

        # -------------------------------------------------------------
        # Optimal-path L1 simulation.
        # This simulates the optimized path in UTM coordinates.
        # -------------------------------------------------------------
        optimal_sim_dyn = simulate_l1_path(
            subarrays=planned_path.subarrays,
            initial_position=planned_path.initial_position_utm,
            initial_heading=planned_path.initial_heading_rad,
            aircraft=cfg.aircraft,
            l1=cfg.l1,
            sim_time_s=cfg.execution.internal_sim_time_s,
        )

        plot_plan_and_simulation(
            original_waypoints_utm=plan.waypoints_utm,
            optimal_path_utm=planned_path.optimal_path_utm,
            optimal_sim_dyn=optimal_sim_dyn,
            conventional_sim_dyn_utm=conventional_dyn_utm,
            title="Internal Simulation Preview",
        )

        answer = input("Accept simulation and execute guidance? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("Mission aborted by user after simulation preview.")
            return

    # ---------------------------------------------------------------------
    # 6. Switch mode / arm / takeoff sequence
    # ---------------------------------------------------------------------
    mav.set_message_rate(
        mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
        20.0,
    )

    mav.set_message_rate(
        mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
        20.0,
    )

    mav.set_message_rate(
        mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD,
        20.0,
    )
    mode = mav.get_mode(blocking=True, timeout=2.0)
    print(f"Current mode: {mode}")

    if mode != "GUIDED":
        mav.set_mode("GUIDED")
        if not mav.wait_mode("GUIDED", timeout=5.0):
            print("WARNING: Failed to confirm GUIDED mode. Continuing cautiously.")

    if not mav.arm(timeout=cfg.execution.arm_timeout_s):
        raise RuntimeError("Vehicle could not be armed.")

    mav.reset_home_position()

    if plan.start_seq < 2:
        print("Starting from takeoff sequence.")
        mav.send_takeoff(cfg.execution.takeoff_alt_m)
        mav.perform_forward_transition(
            throttle_pwm=cfg.execution.transition_throttle_pwm,
            min_speed_mps=cfg.execution.transition_min_speed_mps,
            hold_time_s=cfg.execution.transition_hold_s,
        )
        mav.climb_to_altitude(cfg.execution.cruise_alt_m)
    else:
        print(f"Mission already has earlier sequence. Switching AUTO until seq {plan.start_seq}.")
        mav.set_mode("AUTO")
        mav.wait_until_mission_seq(plan.start_seq)
        mav.set_mode("GUIDED")
        mav.wait_mode("GUIDED", timeout=5.0)

    # ---------------------------------------------------------------------
    # 7. Execute online path-following guidance
    # ---------------------------------------------------------------------
    follower = SITLPathFollower(
        mav=mav,
        aircraft=cfg.aircraft,
        l1=cfg.l1,
        use_wind_aware_bank_solver=cfg.l1.use_wind_aware_bank_solver,
    )

    follower.follow(
        subarrays=planned_path.subarrays,
        max_time_s=cfg.execution.guidance_max_time_s,
        message_rate_hz=5.0, #Frequency
    )

    # ---------------------------------------------------------------------
    # 8. Landing / RTL
    # ---------------------------------------------------------------------
    print("Guidance complete. Starting landing/return procedure.")

    if plan.do_land:
        mav.set_current_mission_item(plan.land_seq)
        time.sleep(0.5)
        mav.set_mode("AUTO")
        print(f"Returned to AUTO landing sequence at mission item {plan.land_seq}.")
    else:
        mav.return_to_land()

    print("Mission Complete!")


if __name__ == "__main__":
    main()
