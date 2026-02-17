import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.stats import t
import glob
import os

"""
Script to calculate expected thrust with uncertainty, 
based on data measured from hot fire by tensometer (in mass)
Just keep data files in 'mass_data' folder
"""

# ============================================================
# USER SETTINGS
# ============================================================
DATA_DIR = "./mass_data/"          # folder with raw files (no headers)
OUTPUT_CSV = "expected_thrust.csv"

TIME_STEP = 0.002                  # common time grid [s]
GRAVITY = 9.80665                  # m/s^2

FILTER_CUTOFF = 30.0               # Hz (None to disable)
SENSOR_STD_FORCE = 5.0             # N (tensometer uncertainty)

PRE_IGN_AVG_TIME = 5             # seconds before ignition for weight estimate
POST_BURN_AVG_TIME = 5           # seconds after burn for weight estimate

# ============================================================
# HELPER FUNCTIONS
# ============================================================
"""
def lowpass_filter(signal, fs, cutoff):
    b, a = butter(4, cutoff / (0.5 * fs), btype='low', output='ba')
    return filtfilt(b, a, signal)
"""
def detect_ignition_and_burn_end(time, force):
    sign = np.sign(force)

    ign_idx = None
    for i in range(1, len(sign)):
        if sign[i - 1] >= 0 and sign[i] < 0:
            ign_idx = i
            break
    if ign_idx is None:
        raise RuntimeError("Ignition not detected")

    end_idx = None
    for i in range(ign_idx + 1, len(sign)):
        # mass of Turbulance is I believe 58kg
        if sign[i] > 0 and force[i]>= 50:
            end_idx = i
            break
    if end_idx is None:
        raise RuntimeError("Burn end not detected")

    return ign_idx, end_idx

# ============================================================
# LOAD, PROCESS EACH RUN
# ============================================================
runs = []

files = sorted(glob.glob(os.path.join(DATA_DIR, "*")))

if len(files) < 2:
    raise RuntimeError("At least two hot-fire files required")

time_end_ign_longest = 0
time_end_all = []
for file in files:
    data = np.loadtxt(file)
    time = data[:, 0]
    force_meas = data[:, 1]     # tensometer mass measurement

    # filtering
    """
    if FILTER_CUTOFF is not None:
        fs = 1.0 / np.mean(np.diff(time))
        force_meas = lowpass_filter(force_meas, fs, FILTER_CUTOFF)
    """
    # detect ignition and burn end
    ign_idx, end_idx = detect_ignition_and_burn_end(time, force_meas)

    t_ign = time[ign_idx]
    t_end = time[end_idx]
    time_end_all.append(t_end - t_ign)
    if t_end > time_end_ign_longest:
        time_end_ign_longest = t_end

    burn_time = t_end - t_ign
    print(f"Ignition time:{t_ign} end time:{t_end}")
    # estimate start weight (before ignition)
    pre_mask = (time >= t_ign - PRE_IGN_AVG_TIME) & (time < t_ign)
    W_start = np.mean(force_meas[pre_mask])

    # estimate end weight (after burn)
    post_mask = (time > t_end) & (time <= t_end + POST_BURN_AVG_TIME)
    W_end = np.mean(force_meas[post_mask])

    # shift time so ignition = 0
    time = time - t_ign

    # isolate burn window
    burn_mask = (time >= 0.0) & (time <= burn_time)
    time = time[burn_mask]
    force_meas = force_meas[burn_mask]

    # build linear weight model
    W_t = W_start + (W_end - W_start) * (time / burn_time)
    print(W_t.mean())
    # compute thrust (CORRECT FORMULA)
    thrust = -(force_meas - W_t)*9.81

    runs.append((time, thrust))

# ============================================================
# COMMON TIME GRID (longest run, fill shorter with NaN)
# ============================================================
t_end_common = max(run[0][-1] for run in runs)

time_grid = np.arange(0.0, t_end_common, TIME_STEP)

# ============================================================
# INTERPOLATE
# ============================================================
thrust_matrix = []

for i, (time, thrust) in enumerate(runs):
    time_mask = time <= time_end_all[i]

    interp = interp1d(
        time[time_mask],
        thrust[time_mask],
        kind="linear",
        fill_value=np.nan,
        bounds_error=False
    )

    thrust_interp = interp(time_grid)

    # Set values outside allowed time range to NaN
    thrust_interp[time_grid > time_end_all[i]] = np.nan

    thrust_matrix.append(thrust_interp)

thrust_matrix = np.array(thrust_matrix)


# ============================================================
# STATISTICS
# ============================================================
"""
mean_thrust = np.mean(thrust_matrix, axis=0)
std_thrust = np.std(thrust_matrix, axis=0, ddof=1)
"""

mean_thrust = np.nanmean(thrust_matrix, axis=0)
std_thrust = np.nanstd(thrust_matrix, axis=0, ddof=1)

total_std = np.sqrt(std_thrust**2 + SENSOR_STD_FORCE**2)

N = np.sum(~np.isnan(thrust_matrix), axis=0)
df = np.maximum(N - 1, 1)
t_factor = t.ppf(0.975, df=df)

#confidence_band = np.minimum(t_factor * total_std / np.sqrt(N), 200)
confidence_band = np.full_like(mean_thrust, 200.0)

valid = N > 1
ci = t_factor[valid] * total_std[valid] / np.sqrt(N[valid])
confidence_band[valid] = np.minimum(ci, 200.0)

mean_thrust = np.maximum(mean_thrust, 0.0)
ci_lower = np.maximum(mean_thrust - confidence_band, 0.0)

# ============================================================
# EXPORT TO CSV
# ============================================================
df_out = pd.DataFrame({
    "time_s": time_grid,
    "mean_thrust_N": mean_thrust,
    "std_thrust_N": total_std,
    "ci95_lower_N": mean_thrust - confidence_band,
    "ci95_upper_N": mean_thrust + confidence_band
})

df_out.to_csv(OUTPUT_CSV, index=False)

# ============================================================
# PLOT
# ============================================================
plt.figure(figsize=(10, 6))

for run in thrust_matrix:
    plt.plot(time_grid, run, color="gray", alpha=0.3)

plt.plot(time_grid, mean_thrust, "k", linewidth=2, label="Expected Thrust")


plt.fill_between(
    time_grid,
    mean_thrust - confidence_band,
    mean_thrust + confidence_band,
    color="blue",
    alpha=0.3,
    label="95% Confidence Interval"
)

plt.xlabel("Time [s]")
plt.ylabel("Thrust [N]")
plt.title("Corrected Expected Thrust (Weight-Compensated)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


print(f"Common burn time: {t_end_common:.3f} s")
