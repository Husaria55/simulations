# AGH Space Systems - Rocket Simulation Files

## Core Files

### [config.py](config.py)
Centralized configuration for all simulation parameters including environment settings, tank geometry, motor specifications, rocket dimensions, aerodynamics, and parachute properties.

### [excel_sheet_functions.py](excel_sheet_functions.py)
Utility functions for extracting key flight metrics and generating report data:
- Rail departure velocity, TWR, static margins
- Accelerations, speeds, Mach numbers, dynamic pressures
- Pitch/yaw moments, landing coordinates
- Advanced damping ratio analysis using Hilbert transform

### [final_version.ipynb](final_version.ipynb)
Primary simulation notebook implementing the complete rocket model with physically accurate tank calculations for Far-out launch site. Defines fluids, tanks, motor, rocket geometry, and executes single flight simulation.

### [far_out_final_terminator.ipynb](far_out_final_terminator.ipynb)
Monte Carlo-style simulation runner for parametric studies across:
- Multiple dates (2016-2025, various times)
- Thrust curve multipliers (0.9-1.1)
- Oxidizer mass and piston positions
- Generates comparison data for sensitivity analysis

### [notebook_for_excel_sheet.ipynb](notebook_for_excel_sheet.ipynb)
Testing and development notebook for experimenting with functions from `excel_sheet_functions.py` on sample flights.

### [setup.py](setup.py)
Factory module for creating simulation objects (environment, fluids, tanks, motor, rocket, flight). Encapsulates RocketPy object initialization using parameters from `config.py` for clean, reusable code.

### [or_version.ipynb](or_version.ipynb)
Validation notebook comparing RocketPy simulation results with OpenRocket predictions. Uses custom atmospheric conditions matching OpenRocket environment to verify model accuracy.

### [thrusts_sim.ipynb](thrusts_sim.ipynb)
Parametric study analyzing the impact of launch rod inclination (45-90°) on flight performance across three thrust profiles (mean, 95% low, 95% high). Generates plots for max speed, acceleration, altitude, and landing positions.
