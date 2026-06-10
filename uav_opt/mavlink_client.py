from dataclasses import dataclass
import time
import math
import numpy as np
import utm
from pymavlink import mavutil


@dataclass(frozen=True)
class AircraftState:
    position_utm: np.ndarray
    heading_rad: float
    roll_rad: float
    relative_alt_m: float | None = None
    airspeed_mps: float | None = None


class MavlinkClient:
    def __init__(self, master):
        self.master = master
        self._last_roll_rad = 0.0
        self._last_airspeed_mps = None
        self._last_relative_alt_m = None

    @classmethod
    def connect(cls, connection_string: str) -> "MavlinkClient":
        master = mavutil.mavlink_connection(connection_string)
        master.wait_heartbeat()
        print(
            f"Connected to autopilot: system={master.target_system}, "
            f"component={master.target_component}"
        )
        return cls(master)

    # ------------------------------------------------------------------
    # Mission download
    # ------------------------------------------------------------------
    def download_mission(
        self,
        overall_timeout_s: float = 30.0,
        per_request_timeout_s: float = 3.0,
        max_retries: int = 3,
    ):
        print("Requesting mission from autopilot...")

        start_time = time.time()
        expected_count = None
        mission_items = {}

        retries = 0
        while expected_count is None and retries < max_retries:
            self.master.waypoint_request_list_send()
            t0 = time.time()

            while time.time() - t0 < per_request_timeout_s:
                if time.time() - start_time > overall_timeout_s:
                    print("Overall timeout while waiting for MISSION_COUNT.")
                    return []

                msg = self.master.recv_match(blocking=False)
                if msg is None:
                    time.sleep(0.05)
                    continue

                if msg.get_type() == "MISSION_COUNT":
                    expected_count = msg.count
                    print(f"Autopilot reports {expected_count} mission items.")
                    break

            if expected_count is None:
                retries += 1
                print(f"No MISSION_COUNT received. Retry {retries}/{max_retries}")

        if expected_count is None:
            return []

        for seq in range(expected_count):
            got_item = False
            retries = 0

            while not got_item and retries < max_retries:
                self.master.waypoint_request_send(seq)
                t0 = time.time()

                while time.time() - t0 < per_request_timeout_s:
                    if time.time() - start_time > overall_timeout_s:
                        print("Overall timeout while downloading mission.")
                        return [mission_items[i] for i in sorted(mission_items)]

                    msg = self.master.recv_match(blocking=False)
                    if msg is None:
                        time.sleep(0.05)
                        continue

                    if msg.get_type() in ("MISSION_ITEM", "MISSION_ITEM_INT"):
                        if msg.seq == seq:
                            mission_items[seq] = msg
                            print(f"Received mission item {seq + 1}/{expected_count}")
                            got_item = True
                            break

                if not got_item:
                    retries += 1
                    print(f"Retry seq {seq}: {retries}/{max_retries}")

        ordered = [mission_items[i] for i in sorted(mission_items)]
        print(f"Downloaded {len(ordered)}/{expected_count} mission items.")
        return ordered

    # ------------------------------------------------------------------
    # Mode / arming
    # ------------------------------------------------------------------
    def get_mode(self, blocking: bool = False, timeout: float = 1.0) -> str | None:
        msg = self.master.recv_match(type="HEARTBEAT", blocking=blocking, timeout=timeout)
        if msg is None:
            return None
        return mavutil.mode_string_v10(msg)

    def set_mode(self, mode: str) -> None:
        mapping = self.master.mode_mapping()
        if mode not in mapping:
            raise ValueError(f"Mode {mode!r} not available. Available: {list(mapping)}")

        mode_id = mapping[mode]

        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        print(f"Requested mode: {mode}")

    def wait_mode(self, mode: str, timeout: float = 10.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            current = self.get_mode(blocking=True, timeout=1.0)
            if current == mode:
                print(f"Confirmed mode: {mode}")
                return True
        return False

    def is_armed(self) -> bool:
        msg = self.master.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if msg is None:
            return False

        return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def arm(self, timeout: float = 20.0, retry_delay_s: float = 1.0) -> bool:
        if self.is_armed():
            print("Vehicle already armed.")
            return True

        print("Attempting to arm vehicle...")
        t0 = time.time()

        while time.time() - t0 < timeout:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            )

            time.sleep(retry_delay_s)

            if self.is_armed():
                print("Vehicle armed.")
                return True

            print("Arming retry...")

        print("ERROR: arm timeout.")
        return False

    # ------------------------------------------------------------------
    # Home / takeoff / climb
    # ------------------------------------------------------------------
    def reset_home_position(self) -> None:
        print("Resetting home position to current location...")

        msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=10.0)
        if msg is None:
            raise RuntimeError("No GLOBAL_POSITION_INT for reset_home_position.")

        self.master.mav.set_home_position_send(
            self.master.target_system,
            int(msg.lat),
            int(msg.lon),
            int(msg.alt),
            0,
            0,
            0,
            [1, 0, 0, 0],
            0,
            0,
            0,
        )

        print("Home position reset command sent.")

    def send_takeoff(self, target_alt_m: float, tolerance_m: float = 1.0) -> None:
        msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=10.0)
        if msg is None:
            raise RuntimeError("No GLOBAL_POSITION_INT before takeoff.")

        initial_rel_alt = msg.relative_alt / 1000.0
        target_rel_alt = initial_rel_alt + target_alt_m

        print(f"Takeoff target relative altitude: {target_rel_alt:.1f} m")

        self.set_mode("GUIDED")

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            target_rel_alt,
        )

        while True:
            msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2.0)
            if msg is None:
                continue

            current_alt = msg.relative_alt / 1000.0
            print(f"Takeoff altitude: {current_alt:.1f} m")

            if current_alt >= target_rel_alt - tolerance_m:
                print("Takeoff altitude reached.")
                break

            time.sleep(0.2)

    def perform_forward_transition(
        self,
        throttle_pwm: int = 1500,
        min_speed_mps: float = 15.0,
        hold_time_s: float = 3.0,
    ) -> None:
        print("Performing forward transition in FBWB.")
        self.set_mode("FBWB")
        self.wait_mode("FBWB", timeout=5.0)

        print(f"Throttle override: {throttle_pwm}")
        self.send_rc_override(throttle=throttle_pwm)

        while True:
            msg = self.master.recv_match(type="VFR_HUD", blocking=True, timeout=2.0)
            if msg is None:
                continue

            airspeed = float(msg.airspeed)
            print(f"Airspeed: {airspeed:.1f} m/s")
            self.send_rc_override(throttle=throttle_pwm)

            if airspeed >= min_speed_mps:
                break

            time.sleep(0.1)

        print(f"Holding transition for {hold_time_s:.1f} s.")
        t0 = time.time()
        while time.time() - t0 < hold_time_s:
            self.send_rc_override(throttle=throttle_pwm)
            time.sleep(0.1)

        self.send_rc_override(throttle=1500)
        print("Forward transition complete.")

    def climb_to_altitude(self, target_alt_m: float, tolerance_m: float = 1.0) -> None:
        print(f"Climbing to {target_alt_m:.1f} m relative altitude.")
        self.set_mode("GUIDED")

        msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=10.0)
        if msg is None:
            raise RuntimeError("No GLOBAL_POSITION_INT before climb.")

        current_lat = msg.lat / 1e7
        current_lon = msg.lon / 1e7

        self.master.mav.mission_item_send(
            self.master.target_system,
            self.master.target_component,
            0,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            2,
            0,
            0,
            0,
            0,
            0,
            current_lat,
            current_lon,
            target_alt_m,
        )

        while True:
            msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2.0)
            if msg is None:
                continue

            alt = msg.relative_alt / 1000.0
            print(f"Climb altitude: {alt:.1f} m")

            if abs(alt - target_alt_m) <= tolerance_m:
                print("Cruise altitude reached.")
                break

            time.sleep(0.2)

    # ------------------------------------------------------------------
    # Mission control
    # ------------------------------------------------------------------
    def wait_until_mission_seq(self, seq: int) -> None:
        print(f"Waiting until MISSION_CURRENT seq == {seq}")

        while True:
            msg = self.master.recv_match(type="MISSION_CURRENT", blocking=True, timeout=1.0)
            if msg is None:
                continue

            print(f"Current mission seq: {msg.seq}")

            if msg.seq == seq:
                print(f"Reached mission seq {seq}.")
                return

    def set_current_mission_item(self, seq: int) -> None:
        print(f"Setting current mission item: {seq}")
        self.master.mav.mission_set_current_send(
            self.master.target_system,
            self.master.target_component,
            seq,
        )

    def return_to_land(self) -> None:
        print("Switching to QRTL.")
        self.set_mode("QRTL")

    # ------------------------------------------------------------------
    # Parameters and message rates
    # ------------------------------------------------------------------
    def set_message_rate(self, message_id: int, rate_hz: float) -> None:
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive.")

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            message_id,
            int(1e6 / rate_hz),
            0,
            0,
            0,
            0,
            0,
        )

    def get_param(self, name: str, timeout: float = 10.0) -> float:
        self.master.mav.param_request_read_send(
            self.master.target_system,
            self.master.target_component,
            name.encode("utf-8"),
            -1,
        )

        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self.master.recv_match(type="PARAM_VALUE", blocking=True, timeout=1.0)
            if msg is None:
                continue

            pid = msg.param_id
            if isinstance(pid, bytes):
                pid = pid.decode("utf-8", errors="ignore")
            pid = pid.strip("\x00")

            if pid == name:
                return float(msg.param_value)

        raise RuntimeError(f"Timeout reading parameter {name}")

    def get_rc_channel_limits(self) -> dict:
        rcmap_roll = int(self.get_param("RCMAP_ROLL"))
        rcmap_pitch = int(self.get_param("RCMAP_PITCH"))
        rcmap_throttle = int(self.get_param("RCMAP_THROTTLE"))
        rcmap_yaw = int(self.get_param("RCMAP_YAW"))

        def params(ch: int) -> dict:
            return {
                "min": int(self.get_param(f"RC{ch}_MIN")),
                "max": int(self.get_param(f"RC{ch}_MAX")),
                "trim": int(self.get_param(f"RC{ch}_TRIM")),
            }

        return {
            "roll": params(rcmap_roll),
            "pitch": params(rcmap_pitch),
            "throttle": params(rcmap_throttle),
            "yaw": params(rcmap_yaw),
            "map": {
                "roll": rcmap_roll,
                "pitch": rcmap_pitch,
                "throttle": rcmap_throttle,
                "yaw": rcmap_yaw,
            },
        }

    # ------------------------------------------------------------------
    # RC override
    # ------------------------------------------------------------------
    def send_rc_override(
        self,
        roll: int | None = None,
        pitch: int | None = None,
        throttle: int | None = None,
        yaw: int | None = None,
        rc_map: dict | None = None,
    ) -> None:
        """
        Send RC override.

        If rc_map is provided, it should map logical controls to physical channels:
        {"roll": 1, "pitch": 2, "throttle": 3, "yaw": 4}

        Unspecified channels are sent as 0, meaning release/no override in ArduPilot.
        """
        channels = [0] * 8

        if rc_map is None:
            rc_map = {"roll": 1, "pitch": 2, "throttle": 3, "yaw": 4}

        def set_channel(name: str, value: int | None):
            if value is None:
                return
            ch = rc_map[name]
            if not (1 <= ch <= 8):
                return
            channels[ch - 1] = int(value)

        set_channel("roll", roll)
        set_channel("pitch", pitch)
        set_channel("throttle", throttle)
        set_channel("yaw", yaw)

        self.master.mav.rc_channels_override_send(
            self.master.target_system,
            self.master.target_component,
            *channels,
        )

    # ------------------------------------------------------------------
    # Aircraft state
    # ------------------------------------------------------------------
    def aircraft_state(self, timeout: float = 2.0) -> AircraftState:
        """
        Read aircraft UTM position, heading, latest roll, relative altitude, airspeed.
        """
        t0 = time.time()

        while time.time() - t0 < timeout:
            msg = self.master.recv_match(blocking=True, timeout=timeout)
            if msg is None:
                continue

            mtype = msg.get_type()

            if mtype == "ATTITUDE":
                self._last_roll_rad = float(msg.roll)

            elif mtype == "VFR_HUD":
                self._last_airspeed_mps = float(msg.airspeed)

            elif mtype == "GLOBAL_POSITION_INT":
                lat = msg.lat / 1e7
                lon = msg.lon / 1e7
                heading_rad = math.radians(msg.hdg * 1e-2)
                rel_alt = msg.relative_alt / 1000.0

                x, y, _, _ = utm.from_latlon(lat, lon)

                self._last_relative_alt_m = rel_alt

                return AircraftState(
                    position_utm=np.array([x, y], dtype=float),
                    heading_rad=heading_rad,
                    roll_rad=self._last_roll_rad,
                    relative_alt_m=rel_alt,
                    airspeed_mps=self._last_airspeed_mps,
                )

        raise TimeoutError("Timed out waiting for aircraft state.")


def airspeed_to_throttle_pwm(
    target_airspeed_mps: float,
    min_airspeed_mps: float,
    max_airspeed_mps: float,
    pwm_min: int,
    pwm_max: int,
) -> int:
    if max_airspeed_mps <= min_airspeed_mps:
        raise ValueError("max_airspeed_mps must be greater than min_airspeed_mps.")

    v = max(min_airspeed_mps, min(max_airspeed_mps, target_airspeed_mps))
    frac = (v - min_airspeed_mps) / (max_airspeed_mps - min_airspeed_mps)

    return int(round(pwm_min + frac * (pwm_max - pwm_min)))


def roll_to_pwm(
    desired_roll_rad: float,
    max_roll_rad: float,
    pwm_min: int,
    pwm_max: int,
) -> int:
    if max_roll_rad <= 0:
        raise ValueError("max_roll_rad must be positive.")

    desired_roll_rad = float(np.clip(desired_roll_rad, -max_roll_rad, max_roll_rad))

    center = 0.5 * (pwm_min + pwm_max)
    half_range = 0.5 * (pwm_max - pwm_min)

    pwm = center + (desired_roll_rad / max_roll_rad) * half_range

    return int(round(np.clip(pwm, pwm_min, pwm_max)))
