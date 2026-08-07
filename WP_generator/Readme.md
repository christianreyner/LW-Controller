# Waypoint Generator

The waypoint generator creates survey missions in Mission Planner.

Generated waypoint files can be imported into Mission Planner and uploaded to an ArduPilot vehicle or SITL instance.

## Usage

Navigate to the waypoint-generator directory:

    cd WP_generator

Run the generator:

    python generate.py

The generated mission is saved as:

    generated.waypoints

## Configuring the Generator

Open `generate.py` and update the mission parameters before running the script.

Typical parameters include:

    gsd_list = [10, 7.5, 5, 2.5, 1]
    sidelap_percent = 60.0
    image_width_px = 6000
    lane_length_m = 200.0
    lane_orientation_deg = 270.0

The parameters control the survey pattern:

- `gsd_list`: Ground sample distance values in centimeters per pixel.
- `sidelap_percent`: Desired image sidelap between adjacent survey lanes.
- `image_width_px`: Camera image width in pixels.
- `lane_length_m`: Length of each survey lane in meters.
- `lane_orientation_deg`: Direction of the survey lanes as a compass bearing.

## Lane Orientation

The lane orientation is measured clockwise from North:

- `0` degrees: North
- `90` degrees: East
- `180` degrees: South
- `270` degrees: West

For example:

    lane_orientation_deg = 270.0

creates a survey lane oriented from East to West.

The generator alternates the direction of consecutive lanes to create a lawn-mower survey pattern.

## Lane Spacing

The lane spacing is calculated using the ground sample distance, image width, and sidelap:

    lane spacing =
        ground sample distance
        × image width
        × (1 − sidelap)

If the ground sample distance is specified in centimeters per pixel, convert the calculated spacing to meters before using it for geographic waypoint generation.

For example:

    gsd_cm_per_pixel = 10
    image_width_px = 6000
    sidelap_percent = 60

The generator uses these values to determine the distance between adjacent survey lanes.

## Generated File

After running the generator, inspect:

    generated.waypoints

The file uses the QGroundControl WPL 110 format and contains the mission waypoints required for the survey pattern.

Before importing the file, verify:

- The file was generated successfully.
- The waypoint coordinates are correct.
- The waypoint order is correct.
- The altitude values are correct.
- The survey lanes cover the intended area.
- The lane spacing is appropriate.
- The lane orientation is correct.

## Importing into QGroundControl

To use the generated mission:

1. Open QGroundControl.
2. Open the Plan view.
3. Import `generated.waypoints`.
4. Inspect the mission on the map.
5. Verify all waypoint locations and altitudes.
6. Save or upload the mission to the vehicle or SITL instance.

Example file path:

    WP_generator/generated.waypoints

## Running with SITL

Start ArduPilot SITL and upload the generated mission to the simulated vehicle.

From the repository root, run:

    python run_sitl.py

The main controller can download and process the mission through MAVLink.


## Safety

Always test generated missions in simulation before using them on a real vehicle.

Before flight, verify:

- Mission boundaries.
- Waypoint coordinates.
- Altitudes.
- Lane spacing.
- Aircraft turn performance.
- Wind conditions.
- Geofence limits.
- Takeoff and landing behavior.
- Return-to-launch behavior.
- Manual takeover capability.
- Autopilot failsafes.

> [!WARNING]
> Never upload a generated waypoint mission to a real aircraft without inspecting it in any mission planner software and testing it in SITL.
