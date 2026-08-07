from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import math
import utm


@dataclass(frozen=True)
class WplItem:
    seq: int
    current: int
    frame: int
    command: int
    param1: float
    param2: float
    param3: float
    param4: float
    lat: float
    lon: float
    alt: float
    autocontinue: int = 1


def _utm_to_latlon(
    e0: float,
    n0: float,
    zone_number: int,
    zone_letter: str,
    east_m: float,
    north_m: float,
) -> tuple[float, float]:
    return utm.to_latlon(e0 + east_m, n0 + north_m, zone_number, zone_letter)


def _bearing_vectors(lane_orientation_deg: float) -> tuple[float, float, float, float]:
    """
    lane_orientation_deg is the heading of the FIRST lane, clockwise from North.

    Returns:
        along_e, along_n  -> lane direction unit vector
        left_e, left_n    -> left-of-heading unit vector
    """
    theta = math.radians(lane_orientation_deg % 360.0)

    along_e = math.sin(theta)
    along_n = math.cos(theta)

    left_e = -math.cos(theta)
    left_n = math.sin(theta)

    return along_e, along_n, left_e, left_n


def generate_qgc_wpl_110_mission(
    lane_length_m: float,
    lane_orientation_deg: float,
    lane_distances_m: Sequence[float],
    *,
    survey_start_lat: float = -35.3562262,
    survey_start_lon: float = 149.1677713,
    survey_alt_m: float = 100.0,
    home_lat: float = -35.3632571,
    home_lon: float = 149.1652396,
    home_alt_m: float = 584.1,
    takeoff_lat: float = -35.36326030,
    takeoff_lon: float = 149.16523720,
    takeoff_alt_m: float = 25.0,
    loiter1_lat: float = -35.3567162,
    loiter1_lon: float = 149.1709471,
    loiter1_alt_m: float = 100.0,
    final_loiter_lat: float = -35.36345560,
    final_loiter_lon: float = 149.16658180,
    final_loiter_alt_m: float = 30.0,
    final_cmd_lat: float = -35.36326130,
    final_cmd_lon: float = 149.16523750,
    turn_side: str = "left",
) -> list[WplItem]:
    """
    Generate a QGC WPL 110 mission matching your example.

    Survey pattern:
      - starts at the top-right
      - first lane goes according to lane_orientation_deg
      - next lane shifts downward
      - lanes alternate direction (snake / lawnmower)

    For your example, use:
      lane_orientation_deg = 270.0
    because the first lane goes from right to left.
    """
    if lane_length_m <= 0:
        raise ValueError("lane_length_m must be positive.")
    if len(lane_distances_m) == 0:
        raise ValueError("lane_distances_m must contain at least one spacing.")
    if turn_side.lower() not in ("left", "right"):
        raise ValueError("turn_side must be 'left' or 'right'.")

    # Survey start point defines the local UTM origin for the generated lanes.
    e0, n0, zone_number, zone_letter = utm.from_latlon(survey_start_lat, survey_start_lon)
    along_e, along_n, left_e, left_n = _bearing_vectors(lane_orientation_deg)

    if turn_side.lower() == "right":
        left_e, left_n = -left_e, -left_n

    mission: list[WplItem] = []
    seq = 0

    def add(
        current: int,
        frame: int,
        command: int,
        p1: float,
        p2: float,
        p3: float,
        p4: float,
        lat: float,
        lon: float,
        alt: float,
        autocontinue: int = 1,
    ) -> None:
        nonlocal seq
        mission.append(
            WplItem(
                seq=seq,
                current=current,
                frame=frame,
                command=command,
                param1=p1,
                param2=p2,
                param3=p3,
                param4=p4,
                lat=lat,
                lon=lon,
                alt=alt,
                autocontinue=autocontinue,
            )
        )
        seq += 1

    # 0) Home / current waypoint
    add(
        current=1,
        frame=0,
        command=16,
        p1=0.0,
        p2=0.0,
        p3=0.0,
        p4=0.0,
        lat=home_lat,
        lon=home_lon,
        alt=home_alt_m,
    )

    # 1) VTOL takeoff
    add(
        current=0,
        frame=3,
        command=84,
        p1=0.0,
        p2=0.0,
        p3=0.0,
        p4=0.0,
        lat=takeoff_lat,
        lon=takeoff_lon,
        alt=takeoff_alt_m,
    )

    # 2) LOITER_TO_ALT
    add(
        current=0,
        frame=3,
        command=31,
        p1=0.0,
        p2=-50.0,
        p3=0.0,
        p4=1.0,
        lat=loiter1_lat,
        lon=loiter1_lon,
        alt=loiter1_alt_m,
    )

    # Survey lanes:
    # lane_distances_m = [d1, d2, d3, ...]
    # means:
    #   lane 1 -> lane 2 = d1
    #   lane 2 -> lane 3 = d2
    #   ...
    # so number of lanes = len(lane_distances_m) + 1
    num_lanes = len(lane_distances_m) + 1

    cur_e = 0.0
    cur_n = 0.0

    for lane_idx in range(num_lanes):
        entry_e = cur_e
        entry_n = cur_n

        # Alternate direction each lane
        forward = (lane_idx % 2 == 0)

        if forward:
            exit_e = entry_e + along_e * lane_length_m
            exit_n = entry_n + along_n * lane_length_m
        else:
            exit_e = entry_e - along_e * lane_length_m
            exit_n = entry_n - along_n * lane_length_m

        # Lane entry waypoint
        lat, lon = _utm_to_latlon(e0, n0, zone_number, zone_letter, entry_e, entry_n)
        add(
            current=0,
            frame=3,
            command=16,
            p1=0.0,
            p2=0.0,
            p3=0.0,
            p4=0.0,
            lat=lat,
            lon=lon,
            alt=survey_alt_m,
        )

        # Lane exit waypoint
        lat, lon = _utm_to_latlon(e0, n0, zone_number, zone_letter, exit_e, exit_n)
        add(
            current=0,
            frame=3,
            command=16,
            p1=0.0,
            p2=0.0,
            p3=0.0,
            p4=0.0,
            lat=lat,
            lon=lon,
            alt=survey_alt_m,
        )

        # Move to the next lane downward
        if lane_idx < num_lanes - 1:
            spacing = float(lane_distances_m[lane_idx])
            cur_e = exit_e + left_e * spacing
            cur_n = exit_n + left_n * spacing

    # Landing sequence exactly like your file
    add(
        current=0,
        frame=0,
        command=189,
        p1=0.0,
        p2=0.0,
        p3=0.0,
        p4=0.0,
        lat=0.0,
        lon=0.0,
        alt=0.0,
    )

    add(
        current=0,
        frame=3,
        command=31,
        p1=0.0,
        p2=75.0,
        p3=0.0,
        p4=1.0,
        lat=final_loiter_lat,
        lon=final_loiter_lon,
        alt=final_loiter_alt_m,
    )

    add(
        current=0,
        frame=3,
        command=85,
        p1=0.0,
        p2=0.0,
        p3=0.0,
        p4=0.0,
        lat=final_cmd_lat,
        lon=final_cmd_lon,
        alt=0.0,
    )

    return mission


def write_qgc_wpl_110(path: str | Path, mission: list[WplItem]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        f.write("QGC WPL 110\n")
        for item in mission:
            f.write(
                f"{item.seq}\t{item.current}\t{item.frame}\t{item.command}\t"
                f"{item.param1:.8f}\t{item.param2:.8f}\t{item.param3:.8f}\t{item.param4:.8f}\t"
                f"{item.lat:.8f}\t{item.lon:.8f}\t{item.alt:.6f}\t{item.autocontinue}\n"
            )
