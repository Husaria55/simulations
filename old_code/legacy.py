"""
This file contains deprecated or unused functions kept strictly for 
reference or historical algorithmic approaches. Do not import these 
into active production code.
"""
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


def damping_ratio(flight: Flight) -> tuple[list[float], list[float]]:
    """Calculates the damping ratio of angle of attack oscillations.

    This function isolates the flight phase between rail exit and apogee. 
    It identifies peaks in the angle of attack and uses the logarithmic 
    decrement method to calculate the damping ratio (zeta) between 
    successive decaying peaks with an amplitude greater than 0.5.
    
    Note: 
        This function produces side effects. It prints validation results 
        to the console and displays two blocking matplotlib plots (AoA 
        oscillations and Damping Ratio over time).

    Args:
        flight (Flight): The object containing flight telemetry and state data.

    Returns:
        tuple[list[float], list[float]]: A tuple containing:
            - list[float]: The calculated damping ratios (zeta) for each valid peak pair.
            - list[float]: The corresponding times (in seconds) of the first peak in each pair.
    """
    # get aoa
    raw_alpha = np.array(flight.angle_of_attack)
    alpha_values = raw_alpha[:, 1] 
    time_values = raw_alpha[:, 0]


    # from leaving rail to apogee
    t_exit = flight.out_of_rail_time
    t_apogee = flight.apogee_time
    mask = (time_values > t_exit + 0.1) & (time_values < t_apogee - 0.1) # Buffer to avoid noise

    t_data = time_values[mask]
    alpha_data = alpha_values[mask]

    # finding peaks
    peaks, _ = find_peaks(alpha_data, distance=5) 
    peak_times = t_data[peaks]
    peak_values = alpha_data[peaks]

    # Calculate Damping Ratio (Zeta) using Logarithmic Decrement
    damping_ratios = []
    damping_times = []

    for i in range(len(peak_values) - 1):
        A1 = peak_values[i]
        A2 = peak_values[i+1]
        
        # Only calculate if amplitude is decaying and significant
        if A1 > A2 and A1 > 0.5: 
            delta = np.log(A1 / A2)
            zeta = 1 / np.sqrt(1 + (2 * np.pi / delta)**2)
            damping_ratios.append(zeta)
            damping_times.append(peak_times[i])

    # 5. Output Results for Table
    if len(damping_ratios) == 0:
        print("sth is not working")
    else:
        min_zeta = np.min(damping_ratios)
        max_zeta = np.max(damping_ratios)
        t_min = damping_times[np.argmin(damping_ratios)]
        t_max = damping_times[np.argmax(damping_ratios)]

        print(f"Lowest Damping Ratio:  {min_zeta:.4f} at t={t_min:.2f} s")
        print(f"Highest Damping Ratio: {max_zeta:.4f} at t={t_max:.2f} s")
        
        # Validation
        if min_zeta < 0.05:
            print(f"Underdamped (< 0.05)")
        else:
            print(f"Minimum damping > 0.05. Passed")

        # 6. Plot for Verification
        plt.figure(figsize=(10, 5))
        plt.plot(t_data, alpha_data, label='Angle of Attack (deg)')
        plt.plot(peak_times, peak_values, "x", color='red', label='Peaks')
        plt.xlabel("Time (s)")
        plt.ylabel("Alpha (deg)")
        plt.title("Angle of Attack Oscillations")
        plt.legend()
        plt.grid(True)
        plt.show()

    x = np.array(damping_times)
    y = np.array(damping_ratios)
    plt.plot(x, y, marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Damping Ratio (Zeta)")
    plt.title("Damping Ratio over Time")
    plt.grid(True)
    plt.show()    
    return damping_ratios, damping_times


def damping_ratio_verbessert(flight: Flight) -> tuple[list[float], list[float]]:
    """Function to calculate the damping ratio of the angle of attack oscillations during the flight using a more advanced method (Hilbert Transform and Sliding Window).
    Parameters:
    flight (Flight): The Flight object containing the flight data.
    Returns:
    tuple: A tuple containing a list of damping ratios and a list of corresponding times.
    Plots the angle of attack, its envelope, instantaneous frequency, and the damping ratio evolution over time."""
    # here i go with partial angle of attack, it's important to choose the right one, because angle of attack in opposite to partial angle of attack is always positive and this is problematic for this function, on the other hand angle of sideslip might be a good choice as well as (partial angle of attack) it depends on the wind direction in which direction will the rocket oscillate

    raw_alpha = np.array(flight.partial_angle_of_attack) 
    # alpha could be here: partial_angle_of_attack, angle_of_sideslip, also w1, w2 (angle velocities) could also maybe work

    alpha_values = raw_alpha[:, 1] 
    time_values = raw_alpha[:, 0]

    # Filter to flight region (Rail to Apogee)
    t_exit = flight.out_of_rail_time
    t_apogee = flight.apogee_time

    # it's best to set this so that it covers the period with oscillations and cuts when they die out
    t_end_analysis = t_exit + 10.0 

    mask = (time_values > t_exit + 0.1) & (time_values < t_end_analysis)
    t_data = time_values[mask]
    alpha_data = alpha_values[mask]

    # here we do some dark magic:
    alpha_detrended = detrend(alpha_data, type='constant') 
    analytic_signal = hilbert(alpha_detrended)
    amplitude_envelope = np.abs(analytic_signal)
    # we use some advanced math trick to get more data points to analyse which yields much better results than only logarythimc decrement on peaks
    # 2. Extract Instantaneous Frequency
    # Unwrap phase to avoid jumps from pi to -pi
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    # Calculate derivative of phase w.r.t time to get frequency (rad/s)
    dt = np.mean(np.diff(t_data))
    instantaneous_omega = np.gradient(instantaneous_phase, dt)

    # Optional: Smooth the envelope and frequency to reduce noise
    # (Window length must be odd, polyorder 2 or 3 is usually good)
    window_len = min(51, len(t_data) // 5) 
    if window_len % 2 == 0: window_len += 1
    amplitude_envelope = savgol_filter(amplitude_envelope, window_len, 3)
    instantaneous_omega = savgol_filter(instantaneous_omega, window_len, 3)


    # --- 3. SLIDING WINDOW DAMPING CALCULATION ---
    damping_ratios = []
    damping_times = []

    # Define a window size (e.g., 0.5 seconds or a set number of samples)
    # Adjust this based on your flight duration. 
    window_time_width = 2  # seconds, a bit higher usually works better
    samples_per_window = int(window_time_width / dt)

    step = 1 # Step size for sliding (lower = higher resolution, slower code)

    for i in range(0, len(t_data) - samples_per_window, step):
        # Get the slice of data
        t_chunk = t_data[i : i + samples_per_window]
        amp_chunk = amplitude_envelope[i : i + samples_per_window]
        omega_chunk = instantaneous_omega[i : i + samples_per_window]
        
        # Check for valid data (avoid log(0) or very low amplitudes that are just noise)
        if np.min(amp_chunk) < 0.1: # Threshold: Ignore if amplitude is < 0.1 deg
            continue

        # Linear Regression on Log(Amplitude)
        # The equation is: ln(A) = -zeta * omega * t + C
        # So the Slope = -zeta * omega
        
        log_amp = np.log(amp_chunk)
        
        # Polyfit returns [slope, intercept]
        # We fit log_amp against relative time (t_chunk - t_chunk[0]) to avoid large number errors
        slope, intercept = np.polyfit(t_chunk - t_chunk[0], log_amp, 1)
        
        # Calculate Zeta
        # Zeta = -Slope / Average_Frequency_in_Window
        avg_omega = np.mean(omega_chunk)
        
        # Protect against divide by zero
        if avg_omega > 0:
            zeta = -slope / avg_omega
            
            # Filter outliers: Zeta is usually between 0 and 0.5 for stable rockets
            if -0.1 < zeta < 1.0:
                damping_ratios.append(zeta)
                damping_times.append(np.mean(t_chunk)) # Store at center of window

    # --- 4. OUTPUT & VISUALIZATION ---
    damping_ratios = np.array(damping_ratios)
    damping_times = np.array(damping_times)

    if len(damping_ratios) > 0:
        print(f"Mean Damping Ratio: {np.mean(damping_ratios):.4f}")
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot 1: Raw Data vs Detrended + Envelope
        ax1.plot(t_data, alpha_data, label='Raw Alpha', alpha=0.5)
        ax1.plot(t_data, alpha_detrended, label='Detrended', color='blue')
        ax1.plot(t_data, amplitude_envelope, label='Hilbert Envelope', color='red', linestyle='--')
        ax1.set_ylabel("Angle (deg)")
        ax1.legend(loc='upper right')
        ax1.set_title("Signal Processing")
        ax1.grid(True)
        
        # Plot 2: Instantaneous Frequency
        ax2.plot(t_data, instantaneous_omega / (2*np.pi), color='green')
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_title("Instantaneous Frequency")
        ax2.grid(True)
        
        # Plot 3: Damping Ratio
        ax3.plot(damping_times, damping_ratios, 'o-', markersize=4, color='purple')
        ax3.set_ylabel("Damping Ratio ($\zeta$)")
        ax3.set_xlabel("Time (s)")
        ax3.set_title("Damping Ratio Evolution")
        #ax3.set_ylim(-0.05, 0.4) # Adjust limit to focus on relevant area
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

    else:
        print("Could not extract valid damping ratios (check thresholds or data quality).")
    return damping_ratios, damping_times


def cop_and_cog_and_aoa_plots_different_method(rocket: Rocket, flight: Flight):
    """Plot the center of pressure, center of gravity, and angle of attack over time using a different method.
    Parameters:
    rocket (Rocket): The Rocket object containing the rocket data.
    flight (Flight): The Flight object containing the flight data.
    Returns:
    None: This function does not return anything, it just plots the graphs.
    """
    time = flight.time  
    time = time[time <= flight.max_speed_time] 
    stability = flight.stability_margin(time)
    aoa = flight.angle_of_attack(time) 
    diameter = 2 * flight.rocket.radius
    cg_pos = rocket.center_of_mass(time) 
    cp_pos = cg_pos + (stability * diameter)
    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time, cp_pos, label='Center of Pressure ($X_{cp}$)', color='red')
    plt.plot(time, cg_pos, label='Center of Gravity ($X_{cg}$)', color='blue', linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Position from Origin (m)")
    plt.title("Center of Pressure vs Center of Gravity")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(aoa, cp_pos, label='Angle of Attack (deg)', color='green')
    plt.xlabel("Angle of Attack (deg)")
    plt.ylabel("Center of Pressure Position (m)")
    plt.title("Center of Pressure vs Angle of Attack")
    plt.grid(True)
    plt.show()