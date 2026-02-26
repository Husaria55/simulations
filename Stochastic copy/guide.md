# Hot-Fire Thrust Data Parser: How It Works

This script processes raw load cell (tensometer) data from solid or hybrid rocket motor hot-fire tests. It takes raw mass measurements, accounts for the weight of the test stand/rocket, applies noise filtering, aligns multiple test runs, and outputs clean, statistical thrust curves ready for use in flight simulators like RocketPy.

## 1. Setup & Prerequisites
* **Data Format:** The script expects space- or tab-separated text files inside a specific folder (by default, `Stochastic copy\mass_data`). Each file must have two columns: `time` (in seconds) and `measured_mass` (in kg). The files should not have headers.
* **Dependencies:** You will need standard scientific Python libraries: `numpy`, `pandas`, `scipy`, and `matplotlib`.

## 2. The Configuration Knobs (User Settings)
At the top of the script, there is a `USER SETTINGS` block where you control how the data is processed:
* **Filtering & Smoothing:** `FILTER_CUTOFF` determines the harshness of the low-pass filter to remove sensor ringing/vibration. `SMOOTH_WINDOW` and `SMOOTH_POLY` control the final visual smoothing of the thrust curve using a Savitzky-Golay filter.
* **Mass Handling:** `INITIAL_MASS` lets you explicitly state the starting weight of the rocket. `MASS_LOSS_COMPENSATION` dictates whether the parser assumes the rocket loses mass linearly during the burn (`"variable"`) or keeps a steady weight (`"constant"`).
* **Time Manipulation:** `TIME_SHIFT`, `TIME_TRIM_START`, and `TIME_TRIM_END` allow you to manually crop out useless data (like long cooldown periods) and shift the burn to start exactly at $t=0$ or any other desired time.

## 3. The Processing Pipeline (Step-by-Step)

Here is exactly what the code does to your data, in chronological order:

### Step A: Data Loading & Noise Filtering
The script reads every file in your target directory. For each file, it runs the raw mass measurements through a 4th-order Butterworth low-pass filter. This removes high-frequency "noise" caused by the physical vibrations of the test stand, leaving a cleaner underlying force trend.

### Step B: Ignition & Burn Detection

The script looks for the exact moment the load cell reading drops rapidly (indicating the motor is pushing against gravity/the stand). It flags this as the `ignition time`. It then scans forward to find when the force reading returns to a baseline state and flags that as the `burn end`. It uses these markers to isolate only the active burn window.

### Step C: Mass Compensation & Thrust Calculation
A load cell doesn't just measure thrust; it measures *Thrust + Weight of the Stand/Rocket*. The script calculates true thrust by subtracting the weight. 
* If set to **Variable**, it calculates a linear trend line from the pre-ignition weight to the post-burn weight and subtracts this sliding value over the course of the burn. 
* It then multiplies the result by the gravitational constant (9.80665) to convert the mass reading (kg) into true force (Newtons).

### Step D: Time Grid Alignment (Interpolation)
Because different test fires might have slightly different sample rates or durations, you cannot just average them together line-by-line. The script creates a "Common Time Grid" (defined by `TIME_STEP`). It interpolates every test run onto this standardized grid. Now, a timestamp like $t=1.002$ means exactly the same thing for every single run.

### Step E: Statistical Averaging
* **Multi-File Mode:** If you have multiple runs, the script calculates the `mean_thrust` across all files for every microsecond on the grid. It also calculates the standard deviation, factors in your baseline sensor uncertainty (`SENSOR_STD_FORCE`), and generates a 95% Confidence Interval band using a T-distribution.
* **Single-File Mode:** If you only have one file, it safely bypasses the standard deviation math to prevent division-by-zero errors, outputting confidence bands of exactly 0.

### Step F: Curve Fitting
The script takes your smoothed, averaged data and fits it to a 3rd-degree polynomial equation ($ax^3 + bx^2 + cx + d$). This provides a highly idealized, purely mathematical representation of your motor's thrust profile.

### Step G: Final Trim & Export
Finally, the script applies any user-defined Time Shifts or Trims. It slices the arrays to your exact desired time window, then packages the data into four clean `.csv` files:
1.  **`expected_thrust.csv`:** A comprehensive file with headers, containing time, mean thrust, fitted thrust, standard deviation, and upper/lower confidence bounds.
2.  **`mean_thrust.csv`:** A clean, header-less 2-column file (Time, Mean Thrust) ideal for importing into simulation software.
3.  **`fitted_thrust.csv`:** A header-less file containing the mathematical curve fit.
4.  **`mean_thrust_uncertainty.csv`:** A header-less file containing just the confidence band width.

It concludes by plotting a graph of all raw runs, the mean expected thrust, the fitted curve, and the confidence interval shadow so you can visually verify the math.