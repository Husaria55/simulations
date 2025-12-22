from rocketpy import Rocket, Environment, Flight, LiquidMotor, Fluid, CylindricalTank, MassFlowRateBasedTank
from math import exp

# Define the environment
env = Environment() # Using Standard Atmosphere by default, 

# Define LiquidMotor

# Define fluids
# When it comes to N20, there is a problem that it's properties change with temperature, so it should be considered if the density shouldn't be given as a function of temperature

oxidizer_liq = Fluid(name="N2O_l", density=1220)
oxidizer_gas = Fluid(name="N2O_g", density=1.9277)
fuel_liq = Fluid(name="ethanol_l", density=789) 
fuel_gas = Fluid(name="ethanol_g", density=1.59)


# Define tanks geometry
# Here is the main problem in TRB the PV is a one tank divided by a piston, which is impossible to match 1 to 1 in rocketpy, so the precise simulation of the center of mass spot and stability margin can't be done, nevertheless the piston should increase rather then decrease the stability margin as it moves up away from the center of pressure.

fuel_tank = CylindricalTank(radius = 0.186, height = 0.25, spherical_caps = False) #I assume cylindrical shape with no spherical caps, as only one side will have them
oxidizer_tank = CylindricalTank(radius = 0.186, height = 0.83, spherical_caps = False)

# The data for the sizes comes from rocketpy, the problem is that inside the tanks are pipes that change the volumes, but here I will neglect that fact


# Define tanks
oxidizer_tank = MassFlowRateBasedTank(
    name="oxidizer tank",
    geometry=oxidizer_tank,
    flux_time=10.5, #From openrocket
    initial_liquid_mass=19, #Guess for now
    initial_gas_mass=0.01,
    liquid_mass_flow_rate_in=0,
    liquid_mass_flow_rate_out=19/10.5,
    gas_mass_flow_rate_in=0,
    gas_mass_flow_rate_out=0,
    liquid=oxidizer_liq,
    gas=oxidizer_gas,
)

fuel_tank = MassFlowRateBasedTank(
    name="fuel tank",
    geometry=fuel_tank,
    flux_time=10.5,
    initial_liquid_mass=6,
    initial_gas_mass=0.01,
    liquid_mass_flow_rate_in=0,
    liquid_mass_flow_rate_out=6/10.5-0.01, #heuristics
    gas_mass_flow_rate_in=0,
    gas_mass_flow_rate_out=lambda t: 0.01 / 10.5 * exp(-0.25 * t),
    liquid=fuel_liq,
    gas=fuel_gas,
)
# To sum up the tanks, they are just here to more or less correctly change the rocket mass, probably it should be based on the temperature, pressure functions over time with changing mass flow etc.

# Motor
# Thrust source from openrocket .eng file, dry mass it the mass of a motor plus empty tanks: 2.7(motor mass) + 16.6(PV) + 1.8(upper dome) + 1.8(bottom dome) + 2.2(piston mass)=25.1
z4000 = LiquidMotor(
    thrust_source="C:\\Users\\krikb\\Desktop\\simulations\\data\\AGH-SS_Z4000-10sBurn-optimal.eng", #From tests
    dry_mass=25.1,
    dry_inertia=(8.46, 8.46, 0.2), #This should be calculated using CAD, here I use estimations, bad estimations
    nozzle_radius=0.0137, #From technical report
    center_of_dry_mass_position=1.33, #Estimated from openrocket
    nozzle_position=0,
    burn_time=14.4,
    coordinate_system_orientation="nozzle_to_combustion_chamber",
)
z4000.add_tank(tank=oxidizer_tank, position=1.285) #From nozzle to center of the tank
z4000.add_tank(tank=fuel_tank, position=2.01)

z4000.all_info()


# Rocket 
trb = Rocket(
    radius=0.2,
    mass=35.969, #All mass - engine mass and fuel
    inertia=(61.47, 61.47, 1.44), #Rough estimations, the correct values should be calculated from CAD
    power_off_drag="C:\\Users\\krikb\\Desktop\\simulations\\data\\powerOffDragCurve.csv", #This should be taken from ansys or some other cfd simulation, here I use just some random data
    power_on_drag='C:\\Users\\krikb\\Desktop\\simulations\\data\\powerOnDragCurve.csv',
    center_of_mass_without_motor=2.43, #from openrocket
    coordinate_system_orientation="nose_to_tail", #Same as in openrocket
)

trb.add_motor(z4000, position=4.2)

nose_cone = trb.add_nose(
    length=0.7, kind="lvhaack", position=0
)

fin_set = trb.add_trapezoidal_fins(
    n=4,
    root_chord=0.287,
    tip_chord=0.059,
    span=0.202,
    sweep_length=0.228,
    position=4.21,
    cant_angle=0,
)

tail = trb.add_tail(
    top_radius=0.2, bottom_radius=0.13, length=0.287, position=4.21
)
trb.draw()
