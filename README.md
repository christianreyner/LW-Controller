# LW-Controller

## Wind-Integrated Navigation Dynamics

LW-Controller is a research and development framework for wind-aware navigation, optimal path planning, and path following for unmanned aerial vehicles.

The project combines:

- Wind-aware aircraft dynamics
- L1 path-following simulation
- Optimal maneuver-path generation
- MAVLink mission download and parsing
- ArduPilot SITL integration
- Online guidance using MAVLink and RC overrides
- Conventional-versus-optimized trajectory comparison
- Waypoint-file generation
- Boustrophedon survey mission generation

> **Warning:** This controller will communicate with and control an aircraft through MAVLink. Use it only in a controlled simulation or test environment. Always verify vehicle configuration, geofencing, failsafes, flight modes, and recovery procedures before attempting flight.

---

## Project Workflow

The main SITL workflow is:

1. Connect to an ArduPilot vehicle through MAVLink.
2. Download the current mission.
3. Save the mission locally as a QGroundControl WPL 110 file.
4. Parse navigation waypoints.
5. Convert the mission into a local, north-aligned planning frame.
6. Generate an optimized maneuver path.
7. Simulate both:
   - Conventional waypoint following
   - Optimized L1 path following
8. Display a trajectory comparison plot.
9. Ask the operator to approve the simulation.
10. Arm, take off, transition, and climb when required.
11. Execute the optimized path online.
12. Resume the landing sequence or initiate return-to-land.

The primary entry point is [`run_sitl.py`](run_sitl.py).

---

## Running the Controller and Configuring It

The main entry point for the controller is `run_sitl.py`.

The controller requests confirmation before starting online guidance:

    Accept simulation and execute guidance? [y/N]:

Enter `y` or `yes` to approve execution. Any other response stops the controller without starting the mission.

> **Safety warning:** Do not approve the simulation unless the plotted trajectory, waypoints, altitude, wind conditions, and expected vehicle behavior have been checked. Always test the complete workflow in ArduPilot SITL before using a real aircraft.

---

## Configuration

The controller configuration is defined in:

    uav_opt/config.py

---
    
## Repository Structure

```text
.
├── WP_generator/
│   ├── generate.py              # Example waypoint-generation script
│   ├── wp_gen.py                # QGroundControl WPL 110 mission generator
│   ├── generated.waypoints      # Example generated mission
│   └── waypoints/               # Waypoint-related files
│
├── legacy/
│   ├── bank_solver_adapter.py   # Legacy bank-angle solver adapter
│   └── optimizer_adapter.py     # Legacy optimizer adapter
│
├── uav_opt/
│   ├── aero.py                  # Aircraft and aerodynamic models
│   ├── angles.py                # Angle conversion and wrapping utilities
│   ├── config.py                # Application and vehicle configuration
│   ├── guidance.py              # Online path-following guidance
│   ├── mavlink_client.py        # MAVLink communication utilities
│   ├── mission_io.py            # Mission parsing and serialization
│   ├── path_utils.py            # Path-processing helpers
│   ├── planner.py               # Optimal-path planning
│   ├── simulator.py             # Dynamic path-following simulation
│   ├── simulator_helper.py      # Simulation helpers
│   ├── transforms.py            # Coordinate-frame transformations
│   ├── wind.py                  # Wind models and calculations
│   ├── maneuvers/               # Maneuver-generation utilities
│   └── wind_optimizer/          # Wind-aware optimization routines
│
└── run_sitl.py                  # Main ArduPilot SITL entry point
