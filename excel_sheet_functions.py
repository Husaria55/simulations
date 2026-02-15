from rocketpy import Flight, LiquidMotor, Rocket
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rail_departure_velocity_in_ft_per_sec(flight: Flight) -> float:
    """
    Calculate the rail departure velocity in feet per second.

    Parameters:
    flight (Flight): The Flight object containing the flight data.

    Returns:
    float: The rail departure velocity in feet per second.
    """
    v_m_s = flight.out_of_rail_velocity
    # 2. Convert to Imperial (ft/s)
    v_ft_s = v_m_s * 3.28084
    return v_ft_s


def average_thrust_during_rail_phase(flight: Flight, motor: LiquidMotor, rocket : Rocket) -> float:
    """
    Calculate the average thrust during the rail phase.

    Parameters:
    flight (Flight): The Flight object containing the flight data.
    motor (LiquidMotor): The motor object containing the thrust data.
    rocket (Rocket): The rocket object containing the mass data.
    Returns:
    float: The average thrust during the rail phase in Newtons.
    """
    t_exit = flight.out_of_rail_time 
    g = 9.80665  
    #  100 sample points between 1 and t_exit, engine starts working really at 1s
    rail_times = np.linspace(1, t_exit, 100)

    # 3. Calculate Average Thrust 
    thrust_values = [motor.thrust(t) for t in rail_times]
    avg_thrust = np.mean(thrust_values)

    # 4. Calculate Average Mass during rail phase
    propellant_mass_values = [motor.propellant_mass(t) for t in rail_times]
    total_mass_values = [rocket.mass + motor.propellant_mass(t) for t in rail_times]
    avg_propellant_mass = np.mean(propellant_mass_values)
    avg_total_mass = rocket.mass + avg_propellant_mass # rocket.mass is dry mass

    # 5. Calculate TWR
    twr_rail_avg = avg_thrust / (avg_total_mass * g)

    return twr_rail_avg


def max_static_margin(flight: Flight) -> float:
    """
    Calculate the maximum static margin during the flight.

    Parameters:
    flight (Flight): The Flight object containing the flight data.

    Returns:
    float: The maximum static margin during the flight.
    """
    flight.static_margin
    nd_static_margin = np.array(flight.static_margin)
    max_static_margin = np.max(nd_static_margin[:,1])
    return max_static_margin


def min_static_margin(flight: Flight) -> float:
    """
    Calculate the minimum static margin during the flight.

    Parameters:
    flight (Flight): The Flight object containing the flight data.

    Returns:
    float: The minimum static margin during the flight.
    """
    flight.static_margin
    nd_static_margin = np.array(flight.static_margin)
    min_static_margin = np.min(nd_static_margin[:,1])
    return min_static_margin


def max_acceleration(flight: Flight) -> tuple:
    """
    Calculate the maximum acceleration during the flight.

    Parameters:
    flight (Flight): The Flight object containing the flight data.

    Returns:
    tuple: The maximum acceleration during the flight in m/s^2 and the time at which it occurs in seconds.
    """
    max_acceleration = flight.max_acceleration
    max_acceleration_time = flight.max_acceleration_time
    return max_acceleration, max_acceleration_time


def max_speed(flight: Flight) -> tuple:
    """
    Calculate the maximum speed during the flight.

    Parameters:
    flight (Flight): The Flight object containing the flight data.

    Returns:
    tuple: The maximum speed during the flight in m/s and the time at which it occurs in seconds.
    """
    max_speed = flight.max_speed
    max_speed_time = flight.max_speed_time
    return max_speed, max_speed_time


def max_mach(flight: Flight) -> float:
    """
    Calculate the maximum Mach number during the flight.

    Parameters:
    flight (Flight): The Flight object containing the flight data.

    Returns:
    float: The maximum Mach number during the flight.
    """
    max_mach = flight.max_mach_number
    return max_mach


def max_dynamic_pressure(flight: Flight) -> tuple:
    """
    Calculate the maximum dynamic pressure during the flight.
    Parameters:
    flight (Flight): The Flight object containing the flight data.
    Returns:
    tuple: The maximum dynamic pressure during the flight in Pa and the time at which it occurs in seconds and the altitude at which it occurs in meters.
    """
    max_dynamic_pressure = flight.max_dynamic_pressure
    max_dynamic_pressure_time = flight.max_dynamic_pressure_time
    altitude_at_max_pressure = flight.z(max_dynamic_pressure_time)
    return max_dynamic_pressure, max_dynamic_pressure_time, altitude_at_max_pressure


def max_acceleration_in_g(flight: Flight) -> tuple:
    """
    Calculate the maximum acceleration during the flight in g's.

    Parameters:
    flight (Flight): The Flight object containing the flight data.

    Returns:
    tuple: The maximum acceleration during the flight in g's and the time at which it occurs in seconds.
    """
    max_acceleration = flight.max_acceleration
    max_acceleration_time = flight.max_acceleration_time
    max_acceleration_in_g = max_acceleration / 9.80665
    return max_acceleration_in_g, max_acceleration_time


def max_acceleration_power_on_in_g(flight: Flight) -> tuple:
    """
    Calculate the maximum acceleration during the powered flight in g's.

    Parameters:
    flight (Flight): The Flight object containing the flight data.

    Returns:
    tuple: The maximum acceleration during the powered flight in g's and the time at which it occurs in seconds.
    """
    max_acceleration_power_on = flight.max_acceleration_power_on
    max_acceleration_power_on_time = flight.max_acceleration_power_on_time
    max_acceleration_power_on_in_g = max_acceleration_power_on / 9.80665
    return max_acceleration_power_on_in_g, max_acceleration_power_on_time


def max_velocity_in_ft_per_sec(flight: Flight) -> tuple:
    """
    Calculate the maximum velocity during the flight in feet per second.

    Parameters:
    flight (Flight): The Flight object containing the flight data.

    Returns:
    tuple: The maximum velocity during the flight in feet per second and the time at which it occurs in seconds.
    """
    max_speed = flight.max_speed
    max_speed_time = flight.max_speed_time
    max_speed_ft_s = max_speed * 3.28084
    return max_speed_ft_s, max_speed_time   


def max_q_in_psf_and_altitude_in_ft(flight: Flight) -> tuple:
    """
    Calculate the maximum dynamic pressure during the flight in psi and the altitude at which it occurs in feet.

    Parameters:
    flight (Flight): The Flight object containing the flight data.

    Returns:
    tuple: The maximum dynamic pressure during the flight in psf and the time at which it occurs in seconds and the altitude at which it occurs in feet.
    """
    max_dynamic_pressure = flight.max_dynamic_pressure
    max_dynamic_pressure_time = flight.max_dynamic_pressure_time
    altitude_at_max_pressure = flight.z(max_dynamic_pressure_time) * 3.28084
    max_dynamic_pressure_psf = max_dynamic_pressure * 0.0208854
    return max_dynamic_pressure_psf, max_dynamic_pressure_time, altitude_at_max_pressure


def max_altitude_in_ft_and_time(flight: Flight) -> tuple:
    """
    Calculate the maximum altitude during the flight in feet and the time at which it occurs in seconds.

    Parameters:
    flight (Flight): The Flight object containing the flight data.

    Returns:
    tuple: The maximum altitude during the flight in feet and the time at which it occurs in seconds.
    """
    max_altitude = flight.apogee
    max_altitude_time = flight.apogee_time
    max_altitude_ft = max_altitude * 3.28084
    return max_altitude_ft, max_altitude_time


def distance_from_pad(flight: Flight) -> float:
    """
    Calculate the distance from the pad at impact.

    Parameters:
    flight (Flight): The Flight object containing the flight data.

    Returns:
    float: The distance from the pad at impact in meters.
    """
    x_impact = flight.x_impact
    y_impact = flight.y_impact
    distance_from_pad = np.sqrt(x_impact**2 + y_impact**2)
    return distance_from_pad


def max_yaw_moment(flight: Flight) -> tuple:
    """
    Calculate the maximum yaw moment during the flight.

    Parameters:
    flight (Flight): The Flight object containing the flight data.

    Returns:
    tuple: The maximum yaw moment during the flight in N⋅m and the time at which it occurs in seconds.
    """
    M3 = flight.M3
    M3_array = np.array(M3)
    M3_column = M3_array[:, 1]
    M3_max = M3_column.max()
    M3_max_time = M3_array[M3_column.argmax(), 0]
    return M3_max, M3_max_time


def max_pitch_moment(flight: Flight) -> tuple:
    """
    Calculate the maximum pitch moment during the flight.

    Parameters:
    flight (Flight): The Flight object containing the flight data.

    Returns:
    tuple: The maximum pitch moment during the flight in N⋅m and the time at which it occurs in seconds.
    """
    M2 = flight.M2
    M2_array = np.array(M2)
    M2_column = M2_array[:, 1]
    M2_max = M2_column.max()
    M2_max_time = M2_array[M2_column.argmax(), 0]
    return M2_max, M2_max_time


def get_df_for_mach_number(flight: Flight) -> pd.DataFrame:
    """
    Get the Mach number data as a DataFrame.

    Parameters:
    flight (Flight): The Flight object containing the flight data.

    Returns:
    dataframe: A DataFrame containing the time and Mach number data.
    """
    mach_number = flight.mach_number
    mach_number_array = np.array(mach_number)
    mach_df = pd.DataFrame(mach_number_array, columns=['Time', 'Mach Number'])
    return mach_df


def get_df_for_center_of_pressure(rocket: Rocket, flight: Flight) -> pd.DataFrame:
    """
    Get the center of pressure data as a DataFrame.

    Parameters:
    rocket (Rocket): The Rocket object containing the rocket data.
    flight (Flight): The Flight object containing the flight data.

    Returns:
    dataframe: A DataFrame containing the Mach number and center of pressure data.
    """
    max_mach_number = flight.max_mach_number
    cp = rocket.evaluate_center_of_pressure()
    cp.savetxt('center_of_pressure.csv', lower=0, upper=max_mach_number, samples=1000)
    cp_df = pd.read_csv('center_of_pressure.csv') # center of pressure dataframe
    return cp_df


def get_df_for_angle_of_attack(flight: Flight) -> pd.DataFrame:
    """
    Get the angle of attack data as a DataFrame.

    Parameters:
    flight (Flight): The Flight object containing the flight data.

    Returns:
    dataframe: A DataFrame containing the time and angle of attack data.
    """
    aoa = flight.angle_of_attack
    aoa.savetxt('angle_of_attack.csv', samples=1000, encoding='utf-8')
    aoa_df = pd.read_csv('angle_of_attack.csv') # angle of attack dataframe
    return aoa_df


def get_df_for_center_of_gravity(rocket: Rocket, flight: Flight) -> pd.DataFrame:
    """
    Get the center of gravity data as a DataFrame.

    Parameters:
    rocket (Rocket): The Rocket object containing the rocket data.
    flight (Flight): The Flight object containing the flight data.

    Returns:
    dataframe: A DataFrame containing the time and center of gravity data.
    """
    cog = rocket.evaluate_center_of_mass()
    cog.savetxt('center_of_gravity.csv', samples=1000)
    cog_df = pd.read_csv('center_of_gravity.csv') # center of gravity dataframe
    return cog_df
 

def get_merged_df_for_cp_and_mach_number(rocket: Rocket, flight: Flight) -> pd.DataFrame:
    """
    Get a merged DataFrame for center of pressure and Mach number.

    Parameters:
    rocket (Rocket): The Rocket object containing the rocket data.
    flight (Flight): The Flight object containing the flight data.

    Returns:
    dataframe: A merged DataFrame containing the Mach number and center of pressure data.
    """
    mach_df = get_df_for_mach_number(flight)
    cp_df = get_df_for_center_of_pressure(rocket, flight)
    max_mach_number_index = mach_df['Mach Number'].idxmax()
    mach_to_max_df = mach_df.loc[:max_mach_number_index]
    cp_df = cp_df.rename(columns={'x': 'Mach_Ref'}) 
    cp_df = cp_df.sort_values('Mach_Ref')
    mach_to_max_df = mach_to_max_df.sort_values('Mach Number')
    merged_df = pd.merge_asof(
        cp_df, 
        mach_to_max_df, 
        left_on='Mach_Ref', 
        right_on='Mach Number', 
        direction='nearest'
    )
    return merged_df


def plot_center_of_pressure_vs_mach_number(cop_df: pd.DataFrame):
    """
    Plot the center of pressure position vs Mach number.

    Parameters:
    cop_df (pd.DataFrame): A DataFrame containing the Mach number and center of pressure data.
    """
    x_mach = np.array(cop_df["x"])
    y_cop = np.array(cop_df["Scalar"])
    font = {'family': 'serif',
        'color':  'black',
        'size': 12}
    plt.plot(x_mach, y_cop)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 1)
    plt.xlabel("Mach number", fontdict=font)
    plt.ylabel("Center of pressure position (m)", fontdict=font)
    plt.title("Center of pressure position vs Mach number", fontdict=font)
    plt.show()


def plot_center_of_gravity_and_of_pressure_vs_time(cog_df: pd.DataFrame, merged_df: pd.DataFrame):
    """
    Plot the center of gravity and center of pressure position vs time.

    Parameters:
    cog_df (pd.DataFrame): A DataFrame containing the time and center of gravity data.
    merged_df (pd.DataFrame): A DataFrame containing the time and center of pressure data.
    """
    font = {'family': 'serif',
        'color':  'black',
        'size': 12}
    y2_cog = np.array(cog_df["Center of Mass Position (m)"])
    merged_cp_and_mach_df_reshaped = merged_df.groupby(merged_df.index // 10).mean()
    x_time = np.array(merged_cp_and_mach_df_reshaped["Time"])
    y_cop = np.array(merged_cp_and_mach_df_reshaped["Scalar"])
    plt.plot(x_time, y_cop)
    plt.plot(x_time, y2_cog)
    plt.xlabel("Time (s)", fontdict=font)
    plt.ylabel("Position (m)", fontdict=font)
    plt.title("Center of pressure and center of gravity position vs time", fontdict=font)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)


def plot_center_of_pressure_vs_angle_of_attack(merged_df: pd.DataFrame, aoa_df: pd.DataFrame):
    """
    Plot the center of pressure position vs angle of attack.

    Parameters:
    merged_df (pd.DataFrame): A DataFrame containing the time and center of pressure data.
    aoa_df (pd.DataFrame): A DataFrame containing the time and angle of attack data.
    """
    font = {'family': 'serif',
    'color':  'black',
    'size': 12}
    reshaped_aoa_df = aoa_df.iloc[:250, :]
    merged_cp_and_mach_df_reshaped_to_250 = merged_df.groupby(merged_df.index // 4).mean()
    merged_cp_and_mach_df_reshaped_to_250.head()
    merged_cop_and_aoa_df = pd.merge_asof(
        reshaped_aoa_df, 
        merged_cp_and_mach_df_reshaped_to_250, 
        left_on='Time (s)', 
        right_on='Time', 
        direction='nearest'
    )
    x_aoa = np.array(merged_cop_and_aoa_df["Angle of Attack (°)"])
    y_cop = np.array(merged_cop_and_aoa_df["Scalar"])
    plt.plot(x_aoa, y_cop)
    plt.xlabel("Angle of attack (deg)", fontdict=font)
    plt.ylabel("Center of pressure position (m)", fontdict=font)
    plt.title("Center of pressure position vs angle of attack", fontdict=font)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)