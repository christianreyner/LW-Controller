# Interactive Waypoint Mission Generator

Generates Mission Planner `.waypoints` files for survey missions.

Features:

- Interactive map-based waypoint editing
- Survey-lane visualization
- GSD-based lane-spacing calculation

## Project Structure

~~~text
.
├── app.py
├── wp_gen.py
├── generate.py
~~~

## Run the Interactive Application

Start the Streamlit application:

~~~bash
streamlit run app.py
~~~

# Run the Original Code (optional)

The original workflow using `generate.py` and `wp_gen.py` remains available.

Run the original generator:

~~~bash
python generate.py
~~~

This creates:

~~~text
generated.waypoints
~~~

The generated file can be imported into Mission Planner.

## Modify the Original Mission

Edit the parameters in `generate.py`:

~~~python
mission = generate_qgc_wpl_110_mission(
    lane_length_m=500.0,
    lane_orientation_deg=270.0,
    lane_distances_m=lane_distances_m,
)
~~~

To use manually specified lane spacing:

~~~python
lane_distances_m = [168.0, 111.0, 86.0]
~~~

Replace the GSD-based calculation with the manual lane-spacing list.

## Orientation Convention

The first-lane orientation is measured clockwise from North:

~~~text
0°   = North
90°  = East
180° = South
270° = West
~~~

For a first lane traveling from right to left:

~~~python
lane_orientation_deg=270.0
~~~
