# Rocketpy simulation of flight dynamics
For now it's single run, target is to run on uncertainties to get a range of results
## Out of rail velocity
```python
test_flight.out_of_rail_velocity
Rail Departure Velocity (SI): 32.16 m/s
Rail Departure Velocity (Imp): 105.51 ft/s
Time of Rail Exit: 1.792 s
```

## Rail Departure Thrust-to-Weight Ratio (TWR)

### **Definition**
The **Rail Departure TWR** represents the average acceleration performance of the launch vehicle while physically constrained by the launch rail. This metric ensures the vehicle has sufficient acceleration to achieve a stable aerodynamic velocity before entering free flight.

Unlike a static "Liftoff TWR" (calculated at $t=0$), this value accounts for the dynamic thrust curve ramp-up and propellant mass depletion during the rail guidance phase.

### **Methodology**
The TWR is calculated by integrating the motor's thrust curve and the vehicle's wet mass depletion profile over the specific time interval $[t_{0}, t_{exit}]$, where $t_{exit}$ is the moment the second-to-last rail button clears the rail. $t_{0}$ is the time when engine starts working.

**Formula:**
$$TWR_{rail} = \frac{\overline{F}_{thrust}}{\overline{m}_{total} \cdot g_0}$$

Where:
* $t_{exit}$ = Time of rail departure (s)
* $\overline{F}_{thrust} = \frac{1}{t_{exit}} \int_{t_{0}}^{t_{exit}} F(t) \, dt$ (Average Thrust during rail contact)
* $\overline{m}_{total} = \frac{1}{t_{exit}} \int_{t_{0}}^{t_{exit}} m(t) \, dt$ (Average Wet Mass during rail departure)
* $g_0$ = Standard gravity ($9.80665 \text{ m/s}^2$ or $32.174 \text{ ft/s}^2$)

```python
Time on Rail: 1.8038 s
Average Thrust (Rail Phase): 3974.61 N
Average Total Mass (Rail Phase): 79.17 kg
---------------------------------------------------
TWR (Rail Phase Average): 5.1193
```


## Stability Margin
```python
test_flight.prints.stability_margin()
#Output:
Stability Margin

Initial Stability Margin: 2.871 c at 0.00 s
Out of Rail Stability Margin: 2.863 c at 1.80 s
Maximum Stability Margin: 3.818 c at 12.25 s
Minimum Stability Margin: 2.863 c at 1.77 s
```
In the excel sheet they want in % of body diameter:
max = 381.8%,
min = 286.3%

## Damping 
trudne się wylosowało

## Max Acceleration and speed and mach
```python
max_acceleration = test_flight.max_acceleration
max_speed = test_flight.max_speed
max_mach = test_flight.max_mach_number
max_dynamic_pressure = test_flight.max_dynamic_pressure
max_acceleration_time = test_flight.max_acceleration_time
max_speed_time = test_flight.max_speed_time
max_dynamic_pressure_time = test_flight.max_dynamic_pressure_time
altitude_at_max_pressure = test_flight.z(max_dynamic_pressure_time)
max_altitude = test_flight.apogee
max_altitude_time = test_flight.apogee_time

# converting to imperial
max_acceleration_ft_s2 = max_acceleration * 3.28084
max_speed_ft_s = max_speed * 3.28084
max_dynamic_pressure_psf = max_dynamic_pressure * 0.0208854342
altitude_at_max_pressure_ft = altitude_at_max_pressure * 3.28084
max_altitude_ft = max_altitude * 3.28084
```
For the mach we should also include if the cd was calculated using incompressible or compressible air model


## Max Distance From the Pad (Nominal)
```python
x_max = test_flight.x.x_array.max()
print(f"Maximum Horizontal Distance from Pad: {x_max:.2f} m")
y_max = test_flight.y.y_array.max()
print(f"Maximum Lateral Distance from Pad: {y_max:.2f} m")
x_array = test_flight.x.x_array
y_array = test_flight.y.y_array
max_distance = np.sqrt(x_array**2 + y_array**2).max()
print(f"Maximum Distance from Pad: {max_distance:.2f} m")
test_flight.plots.trajectory_3d()

Maximum Horizontal Distance from Pad: 240.82 m
Maximum Lateral Distance from Pad: 2200.09 m
Maximum Distance from Pad: 2200.54 m
```

## Max Distance from the Pad (balistic)
```python
# Max Distance from Pad (Ballistic)
# here we have to turn off the parachutes, just comment the lines where they are added to the rocket
x_max = test_flight.x.x_array.max()
print(f"Maximum Horizontal Distance from Pad: {x_max:.2f} m")
y_max = test_flight.y.y_array.max()
print(f"Maximum Lateral Distance from Pad: {y_max:.2f} m")
x_array = test_flight.x.x_array
y_array = test_flight.y.y_array
max_distance = np.sqrt(x_array**2 + y_array**2).max()
print(f"Maximum Distance from Pad: {max_distance:.2f} m")

Maximum Horizontal Distance from Pad: 69.14 m
Maximum Lateral Distance from Pad: 3672.60 m
Maximum Distance from Pad: 3673.25 m
```

## Max Distance from the Pad (Drogue)
```python
# Max Distance from Pad (Ballistic)
# here we have to turn off only main
x_max = test_flight.x.x_array.max()
print(f"Maximum Horizontal Distance from Pad: {x_max:.2f} m")
y_max = test_flight.y.y_array.max()
print(f"Maximum Lateral Distance from Pad: {y_max:.2f} m")
x_array = test_flight.x.x_array
y_array = test_flight.y.y_array
max_distance = np.sqrt(x_array**2 + y_array**2).max()
print(f"Maximum Distance from Pad: {max_distance:.2f} m")

Maximum Horizontal Distance from Pad: 181.58 m
Maximum Lateral Distance from Pad: 2200.09 m
Maximum Distance from Pad: 2200.54 m
```


## Max distance only Main at apogee
```python
# Max Distance from Pad (Ballistic)
# here we have to turn off the parachutes, just comment the lines where they are added to the rocket
x_max = test_flight.x.x_array.max()
print(f"Maximum Horizontal Distance from Pad: {x_max:.2f} m")
y_max = test_flight.y.y_array.max()
print(f"Maximum Lateral Distance from Pad: {y_max:.2f} m")
x_array = test_flight.x.x_array
y_array = test_flight.y.y_array
max_distance = np.sqrt(x_array**2 + y_array**2).max()
print(f"Maximum Distance from Pad: {max_distance:.2f} m")
test_flight.plots.trajectory_3d()

Maximum Horizontal Distance from Pad: 487.26 m
Maximum Lateral Distance from Pad: 2368.07 m
Maximum Distance from Pad: 2368.46 m
```


## Max pitch/yaw moments
```python
# pitch/yaw moments
# Aerodynamic moments in body frame (N⋅m)
M1 = test_flight.M1      # Roll moment
M2 = test_flight.M2      # Pitch moment
M3 = test_flight.M3      # Yaw moment
M1_array = np.array(M1)
M2_array = np.array(M2)
M3_array = np.array(M3)
print(f"Max roll moment: {M1_array.max()} N⋅m")
print(f"Max pitch moment: {M2_array.max()} N⋅m")
print(f"Max yaw moment: {M3_array.max()} N⋅m")    


Max roll moment: 487.25924119974275 N⋅m
Max pitch moment: 487.25924119974275 N⋅m
Max yaw moment: 487.25924119974275 N⋅m
```

```python
# pitch/yaw moments
# Aerodynamic moments in body frame (N⋅m)
M1 = test_flight.M1      # Roll moment
M2 = test_flight.M2      # Pitch moment
M3 = test_flight.M3      # Yaw moment
M1_array = np.array(M1)
M2_array = np.array(M2)
M3_array = np.array(M3)
M1_column = M1_array[:, 1] 
M2_column = M2_array[:, 1]
M3_column = M3_array[:, 1]
M1_max = M1_column.max()
M2_max = M2_column.max()
M3_max = M3_column.max()
M1_max_time = M1_array[M1_column.argmax(), 0]
M2_max_time = M2_array[M2_column.argmax(), 0]
M3_max_time = M3_array[M3_column.argmax(), 0]
print(f"Max roll moment: {M1_max} N⋅m")
print(f"Max pitch moment: {M2_max} N⋅m")
print(f"Max yaw moment: {M3_max} N⋅m") 
print(f"Time of max roll moment: {M1_max_time:.2f} s")
print(f"Time of max pitch moment: {M2_max_time:.2f} s")
print(f"Time of max yaw moment: {M3_max_time:.2f} s") 


Max roll moment: 16.677900153778737 N⋅m
Max pitch moment: 3.765006778704003 N⋅m
Max yaw moment: 2.5289656428054817e-13 N⋅m
Time of max roll moment: 3.31 s
Time of max pitch moment: 3.24 s
Time of max yaw moment: 17.33 s
```

