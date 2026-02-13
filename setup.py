# setup.py
import numpy as np
from CoolProp.CoolProp import PropsSI
from rocketpy import Rocket, Environment, Flight, LiquidMotor, Fluid, CylindricalTank, MassFlowRateBasedTank
import config as cfg

def create_environment(date=cfg.ENV_DATE, lat=cfg.ENV_LAT, lon=cfg.ENV_LON):
    """Creates the Environment object."""
    env = Environment(date=date, latitude=lat, longitude=lon)
    env.set_elevation(cfg.ENV_ELEVATION)
    env.set_atmospheric_model(type=cfg.ENV_ATM_MODEL_TYPE, file=cfg.ENV_ATM_MODEL_FILE)
    env.max_expected_height = cfg.ENV_MAX_HEIGHT
    return env

def create_fluids(p_0=cfg.P_0, ethanol_temp=cfg.ETHANOL_TEMPERATURE):
    """Creates Fluid objects and calculates densities using CoolProp."""
    # Calculate densities
    n2o_liq_rho = PropsSI("D", "P", p_0, "Q", 0, "NitrousOxide")
    n2o_gas_rho = PropsSI("D", "P", p_0, "Q", 1, "NitrousOxide")
    # PropSI shows 63e5 above critical for Ethanol, subtracting 1e5 as per notebook
    eth_liq_rho = PropsSI("D", "P", p_0 - 1e5, "T", ethanol_temp, "Ethanol")
    eth_gas_rho = PropsSI("D", "P", p_0 - 1e5, "Q", 1, "Ethanol")

    oxidizer_liq = Fluid(name="N2O_l", density=n2o_liq_rho)
    oxidizer_gas = Fluid(name="N2O_g", density=n2o_gas_rho)
    fuel_liq = Fluid(name="ethanol_l", density=eth_liq_rho)
    fuel_gas = Fluid(name="ethanol_g", density=eth_gas_rho)

    return oxidizer_liq, oxidizer_gas, fuel_liq, fuel_gas

def create_tanks(
    total_ox_mass=cfg.TOTAL_OXIDIZER_MASS,
    flux_time=cfg.FLUX_TIME,
    piston_pos=cfg.PISTON_POSITION,
    p_0=cfg.P_0
):
    """
    Creates Oxidizer and Fuel Tank objects.
    Recalculates initial mass splits and flow rates based on inputs.
    """
    # 1. Get Fluids
    ox_l, ox_g, fuel_l, fuel_g = create_fluids(p_0=p_0)

    # 2. Geometry
    tank_radius = (cfg.EXTERNAL_TANK_DIAMETER - 2 * cfg.THICKNESS_TANK) / 2
    vol_tank = 0.25 * np.pi * (cfg.EXTERNAL_TANK_DIAMETER - 2 * cfg.THICKNESS_TANK)**2 * cfg.TANK_HEIGHT
    vol_piston = 0.25 * np.pi * (cfg.EXTERNAL_TANK_DIAMETER - 2 * cfg.THICKNESS_TANK)**2 * cfg.THICKNESS_PISTON
    
    vol_ox = piston_pos * vol_tank
    vol_fuel = vol_tank - vol_ox - vol_piston

    # 3. Mass
    gas_init_mass_ox = (vol_ox - (total_ox_mass / ox_l.density)) / (1/ox_g.density - 1/ox_l.density)
    liq_init_mass_ox = total_ox_mass - gas_init_mass_ox
    liq_init_mass_fuel = vol_fuel * fuel_l.density
    gas_init_mass_fuel = cfg.GAS_INITIAL_MASS_FUEL

    # 4. Tank Geometries
    adj_height_ox = piston_pos * cfg.TANK_HEIGHT
    adj_height_fuel = cfg.TANK_HEIGHT - adj_height_ox - cfg.THICKNESS_PISTON

    ox_geom = CylindricalTank(radius=tank_radius, height=adj_height_ox)
    fuel_geom = CylindricalTank(radius=tank_radius, height=adj_height_fuel)

    # 5. Flow Rates
    mfr_liq_ox = liq_init_mass_ox / flux_time - 0.005
    mfr_gas_ox = gas_init_mass_ox / flux_time - 0.005
    mfr_fuel = liq_init_mass_fuel / flux_time - 0.01

    # 6. Create Tank Objects
    ox_tank = MassFlowRateBasedTank(
        name="oxidizer tank",
        geometry=ox_geom,
        flux_time=flux_time,
        initial_liquid_mass=liq_init_mass_ox,
        initial_gas_mass=gas_init_mass_ox,
        liquid_mass_flow_rate_in=0,
        liquid_mass_flow_rate_out=mfr_liq_ox,
        gas_mass_flow_rate_in=0,
        gas_mass_flow_rate_out=mfr_gas_ox,
        liquid=ox_l,
        gas=ox_g,
    )

    fuel_tank = MassFlowRateBasedTank(
        name="fuel tank",
        geometry=fuel_geom,
        flux_time=flux_time,
        initial_liquid_mass=liq_init_mass_fuel - 0.00001,
        initial_gas_mass=gas_init_mass_fuel,
        liquid_mass_flow_rate_in=0,
        liquid_mass_flow_rate_out=mfr_fuel,
        gas_mass_flow_rate_in=0,
        gas_mass_flow_rate_out=0,
        liquid=fuel_l,
        gas=fuel_g,
    )

    return ox_tank, fuel_tank

def create_motor(tanks=None):
    """
    Creates the LiquidMotor object.
    
    Args:
        tanks: A tuple (oxidizer_tank, fuel_tank). If None, creates default tanks.
    """
    if tanks is None:
        tanks = create_tanks()
    
    ox_tank, fuel_tank = tanks

    motor = LiquidMotor(
        thrust_source=cfg.ENGINE_FILE, 
        dry_mass=cfg.MOTOR_DRY_MASS,
        dry_inertia=cfg.MOTOR_DRY_INERTIA,
        nozzle_radius=cfg.NOZZLE_RADIUS,
        center_of_dry_mass_position=cfg.CENTER_OF_DRY_MASS_POS,
        nozzle_position=cfg.NOZZLE_POSITION,
        burn_time=cfg.BURN_TIME,
        coordinate_system_orientation=cfg.MOTOR_COORD_SYS,
    )
    
    motor.add_tank(tank=ox_tank, position=cfg.TANK_POSITION_OX)
    motor.add_tank(tank=fuel_tank, position=cfg.TANK_POSITION_FUEL)
    
    return motor

def create_rocket(motor=None):
    """
    Creates the Rocket object with aerodynamic surfaces and parachutes.
    
    Args:
        motor: LiquidMotor object. If None, creates default motor.
    """
    if motor is None:
        motor = create_motor()

    rocket = Rocket(
        radius=cfg.ROCKET_RADIUS,
        mass=cfg.ROCKET_MASS,
        inertia=cfg.ROCKET_INERTIA,
        power_off_drag=cfg.DRAG_FILE_OFF,
        power_on_drag=cfg.DRAG_FILE_ON,
        center_of_mass_without_motor=cfg.CENTER_OF_MASS_NO_MOTOR,
        coordinate_system_orientation=cfg.ROCKET_COORD_SYS,
    )
    
    rocket.add_motor(motor, position=cfg.MOTOR_POSITION) 

    # Aero Surfaces
    rocket.add_nose(length=cfg.NOSE_LENGTH, kind=cfg.NOSE_KIND, position=cfg.NOSE_POSITION)
    rocket.add_trapezoidal_fins(
        n=cfg.FIN_N,
        root_chord=cfg.FIN_ROOT_CHORD,
        tip_chord=cfg.FIN_TIP_CHORD,
        span=cfg.FIN_SPAN,
        sweep_length=cfg.FIN_SWEEP_LENGTH,
        position=cfg.FIN_POSITION,
        cant_angle=cfg.FIN_CANT_ANGLE
    )
    rocket.add_tail(
        top_radius=cfg.TAIL_TOP_RADIUS, 
        bottom_radius=cfg.TAIL_BOTTOM_RADIUS, 
        length=cfg.TAIL_LENGTH, 
        position=cfg.TAIL_POSITION
    )
    rocket.set_rail_buttons(
        upper_button_position=cfg.BUTTON_UPPER_POS,
        lower_button_position=cfg.BUTTON_LOWER_POS,
        angular_position=cfg.BUTTON_ANGULAR_POS
    )

    # Parachutes (Applying drag factor to Cd_s)
    rocket.add_parachute(
        name="main",
        cd_s=cfg.MAIN_CD_S,
        trigger=cfg.MAIN_TRIGGER,
        sampling_rate=cfg.MAIN_SAMPLING_RATE,
        lag=cfg.MAIN_LAG,
        noise=cfg.MAIN_NOISE,
        radius=cfg.MAIN_RADIUS, 
        height=cfg.MAIN_HEIGHT, 
        porosity=cfg.MAIN_POROSITY
    )

    rocket.add_parachute(
        name="drogue",
        cd_s=cfg.DROGUE_CD_S,
        trigger=cfg.DROGUE_TRIGGER,
        sampling_rate=cfg.DROGUE_SAMPLING_RATE,
        lag=cfg.DROGUE_LAG,
        noise=cfg.DROGUE_NOISE,
        radius=cfg.DROGUE_RADIUS, 
        height=cfg.DROGUE_HEIGHT, 
        porosity=cfg.DROGUE_POROSITY
    )

    return rocket

def create_flight(rocket=None, env=None, inclination=85, heading=0, rail_length=15.24):
    """Creates the Flight object."""
    if rocket is None:
        rocket = create_rocket()
    if env is None:
        env = create_environment()

    return Flight(
        rocket=rocket,
        environment=env,
        rail_length=rail_length,
        inclination=inclination,
        heading=heading
    )