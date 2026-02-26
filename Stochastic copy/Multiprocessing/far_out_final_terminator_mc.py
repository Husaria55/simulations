import os
import multiprocessing
import numpy as np
from numpy.random import default_rng
import pandas as pd
import tempfile
import setup
import config as cfg
import simplekml
from tqdm import tqdm
from scipy.stats import chi2
from functools import partial


# --- Constants ---
LATITUDE = 35.35
LONGITUDE = -117.81
ELEVATION = 621
MAX_EXPECTED_HEIGHT = 6000

# add global constants to avoid loading all data for each flight
mean_data = []
unc_data = []

# ------------------------------------------------------------------
# Thrust Sampling (Point-by-Point)
# ------------------------------------------------------------------

def create_random_thrust_curve(mean_data, unc_data):

    time = mean_data[:, 0]
    mean_thrust = mean_data[:, 1]
    uncertainty = unc_data[:, 1]

    # Safety: avoid zero/negative std
    uncertainty = np.maximum(uncertainty, 1e-6)

    if cfg.THRUST_DISTRIBUTION == "normal":
        sampled_thrust = rng.normal(mean_thrust, uncertainty)

    elif cfg.THRUST_DISTRIBUTION == "uniform":
        sampled_thrust = np.random.uniform(
            mean_thrust - uncertainty,
            mean_thrust + uncertainty
        )
    else:
        raise ValueError("THRUST_DISTRIBUTION must be 'normal' or 'uniform'")

    # Remove NaN / inf
    sampled_thrust = np.nan_to_num(sampled_thrust, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip negative thrust
    sampled_thrust = np.clip(sampled_thrust, 0.0, None)

    # Force physically correct boundaries
    sampled_thrust[0] = 0.0
    sampled_thrust[-1] = 0.0

    # Light smoothing to avoid solver instability
    window = 5
    kernel = np.ones(window) / window
    sampled_thrust = np.convolve(sampled_thrust, kernel, mode="same")

    sampled_curve = np.column_stack((time, sampled_thrust))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    np.savetxt(tmp.name, sampled_curve, delimiter=",")

    return tmp.name


# ------------------------------------------------------------------
# Single Simulation
# ------------------------------------------------------------------

def run_single_simulation(_):
    global mean_data, unc_data, env

    try:
        thrust_file = create_random_thrust_curve(mean_data, unc_data)

        # create NEW tanks per simulation
        ox_tank, fuel_tank = setup.create_tanks()

        liq_motor = setup.create_motor(
            tanks=(ox_tank, fuel_tank),
            thrust_curve_file=thrust_file,
            burn_time=cfg.BURN_TIME
        )
        if cfg.PARACHUTE_MC_ENABLED:
            parachute_works = rng.random() < cfg.PARACHUTE_SUCCESS_PROB
        else:
            parachute_works = True

        rocket = setup.create_rocket(motor=liq_motor, enable_parachutes=parachute_works)

        # uncertainty of inclination
        inclination = rng.uniform(
            cfg.INCLINATION_ANGLE - cfg.INCLINATION_ANGLE_UNC,
            cfg.INCLINATION_ANGLE + cfg.INCLINATION_ANGLE_UNC
        )

        flight = setup.create_flight(
            rocket=rocket,
            env=env,
            inclination=inclination,
            heading=cfg.HEADING_ANGLE,
            rail_length=cfg.ROD_LENGTH
        )

        final_lat = flight.latitude(flight.t_final)
        final_lon = flight.longitude(flight.t_final)

        os.remove(thrust_file)

        return {"lat": float(final_lat),
                "lon": float(final_lon),
                "error": None}

    except Exception as e:
        print("WORKER ERROR:", e)
        return {"error": str(e)}
# ------------------------------------------------------------------
# Sigma Ellipse Calculation
# ------------------------------------------------------------------

def get_ellipse_points(center_lat, center_lon, data, sigma_level):
    coords = data[['lon', 'lat']].values
    cov = np.cov(coords, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    scale = np.sqrt(
        chi2.ppf(
            1 - 2 * (
                1 - 0.6827 if sigma_level == 1
                else 0.9545 if sigma_level == 2
                else 0.9973
            ) / 2, 2
        )
    )

    theta = np.linspace(0, 2 * np.pi, 100)
    ellipsis = (np.sqrt(vals)[:, None] *
                np.array([np.cos(theta), np.sin(theta)]))
    ellipse_pts = np.dot(vecs, ellipsis).T * scale

    return [(center_lon + pt[0], center_lat + pt[1])
            for pt in ellipse_pts]


# ------------------------------------------------------------------
# KML Export
# ------------------------------------------------------------------

def export_to_kml_with_sigmas(df, filename="rocket_dispersion.kml"):
    kml = simplekml.Kml()

    # Landing points
    fol_pts = kml.newfolder(name="Landing Points")
    for _, row in df.iterrows():
        fol_pts.newpoint(coords=[(row['lon'], row['lat'])])

    mean_lat = df['lat'].mean()
    mean_lon = df['lon'].mean()

    colors = {
        1: simplekml.Color.green,
        2: simplekml.Color.yellow,
        3: simplekml.Color.red
    }

    fol_ell = kml.newfolder(name="Dispersion Ellipses")
    for s in [1, 2, 3]:
        pts = get_ellipse_points(mean_lat, mean_lon, df, s)
        pol = fol_ell.newpolygon(
            name=f"{s}-Sigma Boundary",
            outerboundaryis=pts
        )
        pol.style.polystyle.color = \
            simplekml.Color.changealphaint(50, colors[s])
        pol.style.linestyle.color = colors[s]
        pol.style.linestyle.width = 3

    kml.save(filename)
    print(f"KML exported to {filename}")

# --- For multiprocessing ----
def init_worker():
    global mean_data, unc_data, rng, env

    # Load thrust data once per worker
    mean_data = np.loadtxt("data/mean_thrust.csv", delimiter=",")
    unc_data = np.loadtxt("data/mean_thrust_uncertainty.csv", delimiter=",")

    # for random seed
    rng = default_rng()

    env = setup.create_environment()

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":



    with multiprocessing.Pool(
        processes=multiprocessing.cpu_count(),
        initializer=init_worker
    ) as pool:

        results = list(
            tqdm(
                pool.imap_unordered(run_single_simulation,
                                    range(cfg.N_FLIGHTS)),
                total=cfg.N_FLIGHTS
            )
        )

    # Below version without multiprocessing
    """
    results = []
    for _ in tqdm(range(cfg.N_FLIGHTS)):
        results.append(run_single_simulation( mean_data, unc_data, env))
    """
    print("Total runs:", len(results))
    print("Successful runs:", sum(r.get("error") is None for r in results))
    print("Failed runs:", sum(r.get("error") is not None for r in results))

    df_results = pd.DataFrame(
        [r for r in results if r.get("error") is None]
    )

    if not df_results.empty:
        export_to_kml_with_sigmas(df_results)