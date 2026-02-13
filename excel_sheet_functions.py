from rocketpy import Flight, LiquidMotor, Rocket
import numpy as np

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


def max_q_in_psi_and_altitude_in_ft(flight: Flight) -> tuple:
    """
    Calculate the maximum dynamic pressure during the flight in psi and the altitude at which it occurs in feet.

    Parameters:
    flight (Flight): The Flight object containing the flight data.

    Returns:
    tuple: The maximum dynamic pressure during the flight in psi and the time at which it occurs in seconds and the altitude at which it occurs in feet.
    """
    max_dynamic_pressure = flight.max_dynamic_pressure
    max_dynamic_pressure_time = flight.max_dynamic_pressure_time
    altitude_at_max_pressure = flight.z(max_dynamic_pressure_time) * 3.28084
    max_dynamic_pressure_psi = max_dynamic_pressure * 0.000145038
    return max_dynamic_pressure_psi, max_dynamic_pressure_time, altitude_at_max_pressure


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


