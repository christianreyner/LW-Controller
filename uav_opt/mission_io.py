from dataclasses import dataclass
from pathlib import Path
import numpy as np
import utm


NAV_WAYPOINT = 16

# Your old code used 189 as the land-start marker.
# I keep that, but also include common MAVLink landing commands.
LAND_RELATED_COMMANDS = {
    21,   # MAV_CMD_NAV_LAND
    85,   # MAV_CMD_NAV_VTOL_LAND
    189,  # MAV_CMD_DO_LAND_START
}


@dataclass(frozen=True)
class NavigationPlan:
    waypoints_utm: np.ndarray
    do_land: bool
    land_seq: int
    start_seq: int


def save_mavlink_mission(mission_items, filename: str | Path) -> None:
    """
    Save MAVLink mission items in Mission Planner WPL-like format.

    Format:
    index current frame command param1 param2 param3 param4 lat lon alt autocontinue
    """
    filename = Path(filename)

    if not mission_items:
        raise ValueError("No mission items to save.")

    with filename.open("w", encoding="utf-8") as f:
        f.write("Companion WPL\n")

        for wp in mission_items:
            seq = getattr(wp, "seq", 0)
            frame = getattr(wp, "frame", 0)
            command = getattr(wp, "command", 0)
            current = getattr(wp, "current", 0)
            autocontinue = getattr(wp, "autocontinue", 1)

            p1 = getattr(wp, "param1", 0.0)
            p2 = getattr(wp, "param2", 0.0)
            p3 = getattr(wp, "param3", 0.0)
            p4 = getattr(wp, "param4", 0.0)

            if wp.get_type() == "MISSION_ITEM_INT":
                lat = wp.x / 1e7
                lon = wp.y / 1e7
            else:
                lat = wp.x
                lon = wp.y

            alt = wp.z

            f.write(
                f"{seq}\t{current}\t{frame}\t{command}\t"
                f"{p1:.8f}\t{p2:.8f}\t{p3:.8f}\t{p4:.8f}\t"
                f"{lat:.7f}\t{lon:.7f}\t{alt:.6f}\t{autocontinue}\n"
            )


def load_navigation_plan_from_wpl(filename: str | Path) -> NavigationPlan:
    """
    Load NAV_WAYPOINT entries from a WPL file and convert to UTM x/y.

    Skips seq 0 home item.
    """
    filename = Path(filename)

    waypoints = []
    do_land = False
    land_seq = 0
    start_seq = None

    with filename.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if not line[0].isdigit():
                continue

            parts = line.split()
            if len(parts) < 11:
                continue

            seq = int(parts[0])
            command = int(parts[3])
            lat = float(parts[8])
            lon = float(parts[9])

            if seq == 0:
                continue

            if command in LAND_RELATED_COMMANDS:
                do_land = True
                land_seq = seq

            if command != NAV_WAYPOINT:
                continue

            if start_seq is None:
                start_seq = seq

            x, y, _, _ = utm.from_latlon(lat, lon)
            waypoints.append([x, y])

    if start_seq is None:
        raise RuntimeError("No NAV_WAYPOINT found in mission.")

    return NavigationPlan(
        waypoints_utm=np.asarray(waypoints, dtype=float),
        do_land=do_land,
        land_seq=land_seq,
        start_seq=start_seq,
    )
