import io
import math
from typing import Dict, Tuple

import folium
import streamlit as st
from streamlit_folium import st_folium

import wp_gen


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def gsd_to_lane_spacing_m(
    gsd_cm_per_px: float,
    sidelap_percent: float,
    image_width_px: int = 6000,
) -> float:
    """
    Convert GSD to lane-to-lane spacing.

    Example:
        10 cm/px, 60% sidelap, 6000 px image width
        => 240 m lane spacing
    """
    return (
        (gsd_cm_per_px / 100.0)
        * image_width_px
        * (1.0 - sidelap_percent / 100.0)
    )


def mission_to_text(mission) -> str:
    """
    Convert the generated mission to QGC WPL 110 text
    without writing to disk first.
    """
    output = io.StringIO()
    output.write("QGC WPL 110\n")

    for item in mission:
        output.write(
            f"{item.seq}\t"
            f"{item.current}\t"
            f"{item.frame}\t"
            f"{item.command}\t"
            f"{item.param1:.8f}\t"
            f"{item.param2:.8f}\t"
            f"{item.param3:.8f}\t"
            f"{item.param4:.8f}\t"
            f"{item.lat:.8f}\t"
            f"{item.lon:.8f}\t"
            f"{item.alt:.6f}\t"
            f"{item.autocontinue}\n"
        )

    return output.getvalue()


def calculate_lane_preview(
    survey_start_lat: float,
    survey_start_lon: float,
    lane_length_m: float,
    lane_orientation_deg: float,
    lane_distances_m: list[float],
    turn_side: str,
) -> list[Tuple[float, float]]:
    """
    Calculate only the survey-lane points for map visualization.
    """

    e0, n0, zone_number, zone_letter = wp_gen.utm.from_latlon(
        survey_start_lat,
        survey_start_lon,
    )

    along_e, along_n, left_e, left_n = wp_gen._bearing_vectors(
        lane_orientation_deg
    )

    if turn_side.lower() == "right":
        left_e = -left_e
        left_n = -left_n

    num_lanes = len(lane_distances_m) + 1

    cur_e = 0.0
    cur_n = 0.0

    points = []

    for lane_idx in range(num_lanes):
        entry_e = cur_e
        entry_n = cur_n

        forward = lane_idx % 2 == 0

        if forward:
            exit_e = entry_e + along_e * lane_length_m
            exit_n = entry_n + along_n * lane_length_m
        else:
            exit_e = entry_e - along_e * lane_length_m
            exit_n = entry_n - along_n * lane_length_m

        entry_lat, entry_lon = wp_gen._utm_to_latlon(
            e0,
            n0,
            zone_number,
            zone_letter,
            entry_e,
            entry_n,
        )

        exit_lat, exit_lon = wp_gen._utm_to_latlon(
            e0,
            n0,
            zone_number,
            zone_letter,
            exit_e,
            exit_n,
        )

        points.append((entry_lat, entry_lon))
        points.append((exit_lat, exit_lon))

        if lane_idx < num_lanes - 1:
            spacing = float(lane_distances_m[lane_idx])

            cur_e = exit_e + left_e * spacing
            cur_n = exit_n + left_n * spacing

    return points


def add_point_marker(
    fmap: folium.Map,
    name: str,
    point: Tuple[float, float],
    color: str,
) -> None:
    """
    Add a labeled marker to the map.
    """
    lat, lon = point

    folium.Marker(
        location=[lat, lon],
        tooltip=name,
        popup=f"{name}<br>{lat:.8f}, {lon:.8f}",
        icon=folium.Icon(color=color, icon="map-marker"),
    ).add_to(fmap)


# ---------------------------------------------------------------------
# Default points
# ---------------------------------------------------------------------

DEFAULT_POINTS: Dict[str, Tuple[float, float]] = {
    "Survey start": (-35.3562262, 149.1677713),
    "Home": (-35.3632571, 149.1652396),
    "Takeoff": (-35.36326030, 149.16523720),
    "First loiter": (-35.3567162, 149.1709471),
    "Final loiter": (-35.36345560, 149.16658180),
    "Final command": (-35.36326130, 149.16523750),
}

POINT_COLORS = {
    "Survey start": "green",
    "Home": "red",
    "Takeoff": "blue",
    "First loiter": "orange",
    "Final loiter": "purple",
    "Final command": "black",
}


# ---------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Waypoint Mission Generator",
    layout="wide",
)

st.title("Interactive Waypoint Mission Generator")

st.write(
    """
    Select a point type in the sidebar, then click on the map to move that point.
    Change the survey parameters and download the generated QGroundControl
    waypoint file.
    """
)


# ---------------------------------------------------------------------
# Initialize session state
# ---------------------------------------------------------------------

if "points" not in st.session_state:
    st.session_state.points = DEFAULT_POINTS.copy()

if "last_click" not in st.session_state:
    st.session_state.last_click = None


# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------

st.sidebar.header("Map point editing")

selected_point = st.sidebar.selectbox(
    "Point to move",
    list(st.session_state.points.keys()),
)

st.sidebar.info(
    f"Click the map to move: {selected_point}"
)

st.sidebar.header("Survey settings")

lane_length_m = st.sidebar.number_input(
    "Lane length (m)",
    min_value=1.0,
    value=500.0,
    step=10.0,
)

lane_orientation_deg = st.sidebar.number_input(
    "First lane orientation (degrees)",
    min_value=0.0,
    max_value=360.0,
    value=270.0,
    step=1.0,
    help="0 = North, 90 = East, 180 = South, 270 = West",
)

turn_side = st.sidebar.selectbox(
    "Lane turn side",
    ["left", "right"],
)

survey_alt_m = st.sidebar.number_input(
    "Survey altitude (m)",
    min_value=0.0,
    value=100.0,
    step=1.0,
)

st.sidebar.header("Camera settings")

gsd_text = st.sidebar.text_input(
    "GSD values (cm/px)",
    value="10, 7.5, 5, 2.5, 1",
    help="Comma-separated values",
)

sidelap_percent = st.sidebar.number_input(
    "Sidelap (%)",
    min_value=0.0,
    max_value=99.0,
    value=60.0,
    step=1.0,
)

image_width_px = st.sidebar.number_input(
    "Image width (pixels)",
    min_value=1,
    value=6000,
    step=100,
)


# ---------------------------------------------------------------------
# Parse GSD values
# ---------------------------------------------------------------------

try:
    gsd_list = [
        float(value.strip())
        for value in gsd_text.split(",")
        if value.strip()
    ]

    if not gsd_list:
        raise ValueError

    if any(value <= 0 for value in gsd_list):
        raise ValueError

except ValueError:
    st.error(
        "Invalid GSD list. Use a format such as: 10, 7.5, 5, 2.5, 1"
    )
    st.stop()


lane_distances_m = [
    gsd_to_lane_spacing_m(
        gsd_cm_per_px=gsd,
        sidelap_percent=sidelap_percent,
        image_width_px=image_width_px,
    )
    for gsd in gsd_list
]


# ---------------------------------------------------------------------
# Build map
# ---------------------------------------------------------------------

survey_start = st.session_state.points["Survey start"]

fmap = folium.Map(
    location=survey_start,
    zoom_start=15,
    control_scale=True,
    tiles="OpenStreetMap",
)


# Add point markers
for name, point in st.session_state.points.items():
    add_point_marker(
        fmap=fmap,
        name=name,
        point=point,
        color=POINT_COLORS[name],
    )


# Add survey-lane preview
try:
    lane_points = calculate_lane_preview(
        survey_start_lat=survey_start[0],
        survey_start_lon=survey_start[1],
        lane_length_m=lane_length_m,
        lane_orientation_deg=lane_orientation_deg,
        lane_distances_m=lane_distances_m,
        turn_side=turn_side,
    )

    folium.PolyLine(
        locations=lane_points,
        color="green",
        weight=4,
        opacity=0.8,
        tooltip="Survey lanes",
    ).add_to(fmap)

    # Add small circles at lane endpoints
    for index, point in enumerate(lane_points):
        folium.CircleMarker(
            location=point,
            radius=4,
            color="green",
            fill=True,
            fill_opacity=1.0,
            tooltip=f"Survey waypoint {index + 1}",
        ).add_to(fmap)

except Exception as error:
    st.error(f"Could not calculate survey preview: {error}")


# Display map
map_result = st_folium(
    fmap,
    width=1100,
    height=700,
)


# ---------------------------------------------------------------------
# Process map click
# ---------------------------------------------------------------------

if map_result and map_result.get("last_clicked"):
    clicked = map_result["last_clicked"]

    new_click = (
        round(clicked["lat"], 8),
        round(clicked["lng"], 8),
    )

    if new_click != st.session_state.last_click:
        st.session_state.points[selected_point] = new_click
        st.session_state.last_click = new_click
        st.rerun()


# ---------------------------------------------------------------------
# Show current coordinates
# ---------------------------------------------------------------------

st.subheader("Current coordinates")

coordinate_rows = []

for name, point in st.session_state.points.items():
    coordinate_rows.append(
        {
            "Point": name,
            "Latitude": point[0],
            "Longitude": point[1],
        }
    )

st.dataframe(
    coordinate_rows,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------------------
# Show calculated lane distances
# ---------------------------------------------------------------------

st.subheader("Calculated lane spacing")

spacing_rows = []

for gsd, spacing in zip(gsd_list, lane_distances_m):
    spacing_rows.append(
        {
            "GSD (cm/px)": gsd,
            "Lane spacing (m)": round(spacing, 3),
        }
    )

st.dataframe(
    spacing_rows,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------------------
# Generate mission
# ---------------------------------------------------------------------

try:
    mission = wp_gen.generate_qgc_wpl_110_mission(
        lane_length_m=lane_length_m,
        lane_orientation_deg=lane_orientation_deg,
        lane_distances_m=lane_distances_m,
        survey_start_lat=st.session_state.points["Survey start"][0],
        survey_start_lon=st.session_state.points["Survey start"][1],
        survey_alt_m=survey_alt_m,

        home_lat=st.session_state.points["Home"][0],
        home_lon=st.session_state.points["Home"][1],

        takeoff_lat=st.session_state.points["Takeoff"][0],
        takeoff_lon=st.session_state.points["Takeoff"][1],

        loiter1_lat=st.session_state.points["First loiter"][0],
        loiter1_lon=st.session_state.points["First loiter"][1],

        final_loiter_lat=st.session_state.points["Final loiter"][0],
        final_loiter_lon=st.session_state.points["Final loiter"][1],

        final_cmd_lat=st.session_state.points["Final command"][0],
        final_cmd_lon=st.session_state.points["Final command"][1],

        turn_side=turn_side,
    )

    waypoint_text = mission_to_text(mission)

    st.subheader("Mission summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Number of waypoints", len(mission))

    with col2:
        st.metric("Number of survey lanes", len(lane_distances_m) + 1)

    with col3:
        st.metric("Lane length", f"{lane_length_m:.1f} m")

    st.download_button(
        label="Download generated.waypoints",
        data=waypoint_text,
        file_name="generated.waypoints",
        mime="text/plain",
    )

    with st.expander("Show generated waypoint file"):
        st.code(waypoint_text, language="text")

except Exception as error:
    st.error(f"Mission generation failed: {error}")