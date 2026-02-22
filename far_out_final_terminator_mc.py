import os
import multiprocessing
import numpy as np
import pandas as pd
import setup
import config as cfg
import excel_sheet_functions as ex
from rocketpy import Environment, Rocket, Flight

try:
    import simplekml
except ImportError:
    print("Please install simplekml: pip install simplekml")
from tqdm import tqdm
from scipy.stats import chi2

# --- Constants ---
LATITUDE = 35.35
LONGITUDE = -117.81
ELEVATION = 621
MAX_EXPECTED_HEIGHT = 6000


def get_env(date, latitude, longitude, elevation, max_expected_height):
    env = Environment(date=date, latitude=latitude, longitude=longitude,
                      elevation=elevation, max_expected_height=max_expected_height)
    env.set_atmospheric_model(type="Reanalysis", file="data_for_weather_reanalysis.nc", dictionary="ECMWF")
    return env


def run_single_simulation(params):
    date, thrust_multiplier, ox_mass, piston_pos = params
    try:
        env = get_env(date, LATITUDE, LONGITUDE, ELEVATION, MAX_EXPECTED_HEIGHT)
        ox_tank, fuel_tank = setup.create_tanks(total_ox_mass=ox_mass, piston_pos=piston_pos)
        liq_motor = setup.create_motor(tanks=(ox_tank, fuel_tank),
                                       thrust_curve_file=f"./data/thrust_curve_{int(thrust_multiplier * 100)}.csv",
                                       burn_time=(1, cfg.BURN_TIME))
        rocket = setup.create_rocket(motor=liq_motor)
        flight = setup.create_flight(rocket=rocket, env=env, inclination=87, heading=0, rail_length=15.24)

        # FIX: Extract the scalar value at the last recorded time (impact)
        # We access the last value in the data array [-1][1] or call the function at t_final
        final_lat = flight.latitude(flight.t_final)
        final_lon = flight.longitude(flight.t_final)

        return {
            "lat": float(final_lat),
            "lon": float(final_lon),
            "error": None
        }
    except Exception as e:
        # It's better to return the string of the error to avoid pickling the Exception object itself
        return {"error": str(e)}
    
def get_ellipse_points(center_lat, center_lon, data, sigma_level):
    """Calculates points for a sigma-level error ellipse."""
    coords = data[['lon', 'lat']].values
    cov = np.cov(coords, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # Chi-square value for 2 degrees of freedom
    # 1-sigma: ~1.52, 2-sigma: ~2.48, 3-sigma: ~3.44 (sqrt of chi2.ppf)
    scale = np.sqrt(
        chi2.ppf(1 - 2 * (1 - 0.6827 if sigma_level == 1 else 0.9545 if sigma_level == 2 else 0.9973) / 2, 2))

    theta = np.linspace(0, 2 * np.pi, 100)
    ellipsis = (np.sqrt(vals)[:, None] * np.array([np.cos(theta), np.sin(theta)]))
    ellipse_pts = np.dot(vecs, ellipsis).T * scale

    return [(center_lon + pt[0], center_lat + pt[1]) for pt in ellipse_pts]


def export_to_kml_with_sigmas(df, filename="rocket_dispersion.kml"):
    kml = simplekml.Kml()

    # 1. Add Landing Points
    fol_pts = kml.newfolder(name="Landing Points")
    for _, row in df.iterrows():
        fol_pts.newpoint(coords=[(row['lon'], row['lat'])])

    # 2. Calculate and Add Sigma Ellipses
    mean_lat, mean_lon = df['lat'].mean(), df['lon'].mean()
    colors = {1: simplekml.Color.green, 2: simplekml.Color.yellow, 3: simplekml.Color.red}

    fol_ell = kml.newfolder(name="Dispersion Ellipses")
    for s in [1, 2, 3]:
        pts = get_ellipse_points(mean_lat, mean_lon, df, s)
        pol = fol_ell.newpolygon(name=f"{s}-Sigma Boundary", outerboundaryis=pts)
        pol.style.polystyle.color = simplekml.Color.changealphaint(50, colors[s])
        pol.style.linestyle.color = colors[s]
        pol.style.linestyle.width = 3

    kml.save(filename)
    print(f"KML exported with 1, 2, 3 Sigma boundaries to {filename}")


if __name__ == "__main__":
    # Define tasks (truncated list for brevity)
    dates = [(2024, 5, 28, 8), (2024, 5, 28, 12)]
    thrust_multipliers = [0.9, 1.0, 1.1]
    mass_configs = [(18, 0.85), (15, 0.70)]
    tasks = [(d, tm, m, p) for d in dates for tm in thrust_multipliers for m, p in mass_configs]

    with multiprocessing.Pool(os.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(run_single_simulation, tasks), total=len(tasks)))

    df_results = pd.DataFrame([r for r in results if r.get("error") is None])
    if not df_results.empty:
        export_to_kml_with_sigmas(df_results)