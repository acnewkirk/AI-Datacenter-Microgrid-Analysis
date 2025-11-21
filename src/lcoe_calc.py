"""
LCOE Calculator for Datacenter Power Systems
Calculates and compares levelized cost of electricity for different power generation options.
Refactored to use config.py
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import logging
import sys
import reverse_geocoder as rg
from config import Config, load_config

# Import required functions from other modules
from microgrid_optimizer import MicrogridOptimizer, SystemCosts, OptimizationResult, PowerSystemOptimizer
from natgas_system_tool import NGPowerPlantCalculator, calculate_part_load_efficiency
from power_systems_estimator import PowerFlowAnalyzer
from datacenter_analyzer import DatacenterAnalyzer
from pvstoragesim import SimulationResult
from degradation_model import interpolate_annual_energy
import it_facil
from it_facil import FacilityLoad

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

HOURS_PER_YEAR = 8760
SQ_KM_PER_ACRE = 0.00404686   # Conversion factor

# Grid baseline lookup table by state/region using Industrial electricity prices Source from EIA. y 2022 dollars, 4 year rolling avg
# interconnection years, industrial electricity price in cents/kWh, and natural gas price in $/MMBtu
GRID_BASELINE_DATA = {
# New England 
'Connecticut':   ('ISO-NE', 2.5, 15.20, 7.87),
'Maine':         ('ISO-NE', 2.5, 12.01, 9.47),
'Massachusetts': ('ISO-NE', 2.5, 16.79, 12.04),
'New Hampshire': ('ISO-NE', 2.5, 15.39, 12.02),
'Rhode Island':  ('ISO-NE', 2.5, 16.42, 11.57),
'Vermont':       ('ISO-NE', 2.5, 11.02, 5.16),

# Middle Atlantic
'New Jersey':    ('PJM',   3.75, 11.03, 8.35),
'New York':      ('NYISO', 4.5,  7.82,  9.43),
'Pennsylvania':  ('PJM',   3.75, 7.26, 10.67),

# East North Central
'Illinois':      ('PJM/MISO', 3.75, 7.88, 6.79),
'Indiana':       ('MISO',     3.75, 7.49, 6.17),
'Michigan':      ('MISO',     3.75, 7.52, 7.70),
'Ohio':          ('PJM',      3.75, 6.39, 7.02),
'Wisconsin':     ('MISO',     3.75, 7.53, 6.22),

# West North Central
'Iowa':          ('MISO',       3.75, 5.63, 6.40),
'Kansas':        ('SPP',        4.0,  7.02, 5.04),
'Minnesota':     ('MISO',       3.75, 8.46, 5.60),
'Missouri':      ('SPP/MISO',   4.0,  7.10, 8.69),
'Nebraska':      ('SPP',        4.0,  6.67, 5.54),
'North Dakota':  ('MISO/SPP',   4.0,  6.98, 2.98),
'South Dakota':  ('SPP',        4.0,  7.63, 5.57),

# South Atlantic
'Delaware':          ('PJM',         3.75, 7.92, 12.43),
'District of Columbia': ('PJM',      3.75, 9.03, None),
'Florida':           ('Non-ISO',     3.0,  8.09, 6.50),
'Georgia':           ('Non-ISO',     3.0,  5.41, 4.86),
'Maryland':          ('PJM',         3.75, 8.75, 12.07),
'North Carolina':    ('PJM/Non-ISO', 3.5,  7.26, 6.08),
'South Carolina':    ('Non-ISO',     3.0,  6.01, 4.68),
'Virginia':          ('PJM',         3.75, 8.26, 4.58),
'West Virginia':     ('PJM',         3.75, 7.19, 3.85),

# East South Central
'Alabama':      ('Non-ISO',    3.0,  6.73, 4.11),
'Kentucky':     ('PJM/MISO',   3.75, 6.12, 4.07),
'Mississippi':  ('MISO',       3.75, 6.38, 5.04),
'Tennessee':    ('Non-ISO/MISO',3.5, 5.99, 4.99),

# West South Central
'Arkansas':     ('SPP/MISO', 4.0, 5.85, 8.57),
'Louisiana':    ('MISO/SPP', 4.0, 5.98, 3.49),
'Oklahoma':     ('SPP',      4.0, 4.68, 3.01),
'Texas':        ('ERCOT',    3.0, 5.48, 3.24),

# Mountain (largely non-ISO “West”)
'Arizona':      ('Non-ISO', 4.0, 6.84, 6.29),
'Colorado':     ('Non-ISO', 4.0, 7.75, 7.70),
'Idaho':        ('Non-ISO', 4.0, 6.15, 6.03),
'Montana':      ('Non-ISO', 4.0, 6.67, 7.14),
'Nevada':       ('CAISO/Non-ISO', 4.5, 7.45, 10.64),
'New Mexico':   ('SPP/Non-ISO', 4.0, 5.39, 4.58),
'Utah':         ('Non-ISO', 4.0, 6.11, 9.06),
'Wyoming':      ('Non-ISO', 4.0, 7.26, 5.92),

# Pacific Contiguous
'California':   ('CAISO',   4.75, 18.50, 12.75),
'Oregon':       ('Non-ISO', 4.0,  7.18, 7.23),
'Washington':   ('Non-ISO', 4.0,  5.87, 10.32),

# Pacific Noncontiguous
'Alaska':       ('Non-ISO', 4.0, 20.09, 5.92),
'Hawaii':       ('Non-ISO', 4.0, 33.71, 25.82),
}

# Default 

GRID_DEFAULT = ('Unknown', 4.0, 7.25, 4.11)  # US average commercial electricity price and industrial natural gas price from EIA, 2022$



# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LCOEResult:
   """Results from LCOE calculation for a single system"""
   system_type: str
   capex_npv: float  # NPV of capital costs ($)
   opex_npv: float   # NPV of operating costs ($)
   energy_npv: float  # NPV of energy production (MWh)
   lcoe: float       # Levelized cost ($/MWh)
   construction_years: float
   nameplate_capacity_mw: float
   
@dataclass
class SystemComparison:
   """Comparison results across all systems"""
   ac_solar: LCOEResult
   dc_solar: LCOEResult
   natural_gas: LCOEResult
   grid_baseline: LCOEResult
   total_gpus: int
   facility_load_mw: float
   annual_energy_gwh: float
   # Detailed capacity information
   ac_solar_mw: float
   ac_battery_mw: float
   ac_battery_mwh: float
   dc_solar_mw: float
   dc_battery_mw: float
   dc_battery_mwh: float
   ng_nameplate_mw: float
   ng_configuration: str

# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def calculate_facility_load(total_gpus: int, required_uptime_pct: float, config: Config) -> it_facil.FacilityLoad:
   """
   Calculates facility load using the it_facil module.
   """
   result = it_facil.calculate_facility_load(
       total_gpus=total_gpus,
       required_uptime_pct=required_uptime_pct,
       config=config
   )
   return result


def get_state_from_coords(latitude: float, longitude: float, 
                         tolerance_km: float = 50.0) -> Optional[str]:
    """
    Get US state from coordinates with optional border tolerance.
    
    Args:
        latitude, longitude: Coordinates to check
        tolerance_km: If in Canada/Mexico, search within this distance for US match
    """
    try:
        results = rg.search((latitude, longitude))
        country = results[0].get('cc', '')
        state = results[0].get('admin1', '')
        
        # If already in US, return the state
        if country == 'US':
            return state
        
        # If in Canada or Mexico and tolerance enabled, search nearby
        if country in ['CA', 'MX'] and tolerance_km > 0:
            # Search multiple nearby points (crude approximation)
            # 1 degree latitude ≈ 111 km
            offset_deg = tolerance_km / 111.0
            
            nearby_points = [
                (latitude - offset_deg, longitude),  # South
                (latitude + offset_deg, longitude),  # North
                (latitude, longitude - offset_deg),  # West
                (latitude, longitude + offset_deg),  # East
            ]
            
            for lat, lon in nearby_points:
                nearby_results = rg.search((lat, lon))
                nearby_country = nearby_results[0].get('cc', '')
                nearby_state = nearby_results[0].get('admin1', '')
                
                if nearby_country == 'US':
                    logger.info(f"Border tolerance: ({latitude}, {longitude}) "
                              f"mapped to {nearby_state} (originally {country})")
                    return nearby_state
        
        return None
        
    except Exception as e:
        logger.warning(f"Geocoding error: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════
# LCOE CALCULATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def calculate_npv(cash_flows: list, discount_rate: float, start_year: int = 0) -> float:
   """Calculate net present value of a cash flow series mid year calc"""
   npv = 0
   for i, cf in enumerate(cash_flows):
       year = start_year + i
       npv += cf / ((1 + discount_rate) ** (year+.5))
   return npv

def calculate_solar_storage_lcoe(
   system_type: str,  # "ac_coupled" or "dc_coupled"
   solar_mw: float,
   battery_mw: float,
   battery_mwh: float,
   land_acres: float,
   sim_year_0: SimulationResult,
   sim_year_13: SimulationResult,
   sim_year_14: SimulationResult,
   sim_year_25: SimulationResult,
   year_0_stats: Dict,
   construction_years: float,
   required_uptime_pct: float,
   config: Config
) -> LCOEResult:
   """Calculate LCOE for solar+storage system using degradation analysis."""
   
   # Select appropriate BOS costs based on system type
   if system_type == "dc_coupled":
       solar_bos_cost = config.costs.solar_bos_cost_y0_dc
       battery_bos_cost = config.costs.battery_bos_cost_y0_dc
   else:
       solar_bos_cost = config.costs.solar_bos_cost_y0_ac
       battery_bos_cost = config.costs.battery_bos_cost_y0_ac
   
   # Year 0 capital costs
   solar_capex = solar_mw * 1000 * config.costs.solar_cost_y0
   battery_capex = battery_mw * 1000 * config.costs.bess_cost_y0
   solar_bos_capex = solar_mw * 1000 * solar_bos_cost
   battery_bos_capex = battery_mw * 1000 * battery_bos_cost
   land_capex = land_acres * SQ_KM_PER_ACRE * config.costs.land_cost_per_sq_km
   
   total_initial_capex = solar_capex + battery_capex + solar_bos_capex + battery_bos_capex + land_capex
   
   # Construction cash flows
   capex_schedule = get_construction_schedule(construction_years)
   capex_npv = 0
   for year, fraction in capex_schedule:
       discount_year = year + 0.5
       capex_npv += (total_initial_capex * fraction) / ((1 + config.financial.discount_rate) ** discount_year)
   
   # Get annual energy production with degradation
   annual_energy_array = interpolate_annual_energy(
       sim_year_0, sim_year_13, sim_year_14, sim_year_25,
       construction_years, year_0_stats
   )
   
   # Annual O&M costs
   annual_om = ((solar_mw * config.costs.solar_fixed_om) + (battery_mw * config.costs.storage_fixed_om)) * 1000
   
   # Battery replacement at configured year
   battery_replacement_capex = battery_mw * 1000 * config.costs.bess_cost_y15
   battery_replacement_year = int(construction_years) + config.design.battery_lifespan_years
   
   # Build cash flow series
   capex_flows = [0] * config.financial.evaluation_years
   opex_flows = [0] * config.financial.evaluation_years
   energy_flows = [0] * config.financial.evaluation_years
   
   # Operations start after construction
   ops_start_year = int(np.ceil(construction_years))
   
   # Fill operational years
   for year in range(config.financial.evaluation_years):
       if year >= ops_start_year:
           opex_flows[year] = annual_om
           energy_flows[year] = annual_energy_array[year]
   
   # Battery replacement
   if battery_replacement_year < config.financial.evaluation_years:
       capex_flows[battery_replacement_year] = battery_replacement_capex
   
   # Calculate NPVs
   capex_npv += calculate_npv(capex_flows, config.financial.discount_rate)
   opex_npv = calculate_npv(opex_flows, config.financial.discount_rate)
   energy_npv = calculate_npv(energy_flows, config.financial.discount_rate)
   
   # Calculate LCOE
   total_cost_npv = capex_npv + opex_npv
   lcoe = (total_cost_npv / energy_npv if energy_npv > 0 else float('inf')) / 1000  # Convert to $/kWh
   
   return LCOEResult(
       system_type=system_type,
       capex_npv=capex_npv,
       opex_npv=opex_npv,
       energy_npv=energy_npv,
       lcoe=lcoe,
       construction_years=construction_years,
       nameplate_capacity_mw=solar_mw + battery_mw
   )


def calculate_gas_system_lcoe(
    plant_config: 'PlantConfiguration',
    gas_price: float,
    facility_load: FacilityLoad,
    config: Config,
    construction_years: Optional[float] = None
) -> 'LCOEResult':
    """
    Calculate LCOE for a natural gas + diesel backup system with proper energy accounting.
    NG generation is based on degraded capacity × availability × hours/year.
    Diesel generation fills the shortfall up to the pre-sized annual EUE.
    """

    from degradation_model import get_gas_degradation_factors

    # ─────────────────────────────────────────────
    # Setup and constants
    # ─────────────────────────────────────────────
    if construction_years is None:
        construction_years = plant_config.construction_timeline['construction_years']

    # --- START OF REFACTORED LOGIC ---
    
    power_analyzer = PowerFlowAnalyzer(config)
    # 1. Get the new flat dictionary of multipliers
    mult = power_analyzer.get_bus_architecture_multipliers("natural_gas")

    # 2. Calculate the total annual energy required AT THE BUS
    total_bus_demand_mwh = (
        facility_load.annual_it_energy_mwh * mult['bus_to_it'] +
        facility_load.annual_cooling_energy_mwh * mult['bus_to_cooling']
    )
    
    # 3. "Gross up" the bus demand to find the required generation AT THE SOURCE
    required_at_generator_mwh = total_bus_demand_mwh * mult['grid_to_bus']

    # This value represents the demand side of the LCOE calculation
    required_at_datacenter_mwh = facility_load.annual_facility_energy_mwh

    

    # Diesel design parameters (already sized upstream)
    diesel_design = getattr(plant_config, "diesel_design", None)
    diesel_capex = diesel_design.total_capex if diesel_design else 0.0
    diesel_fixed_om = diesel_design.annual_fixed_om if diesel_design else 0.0
    
    # ─────────────────────────────────────────────
    # CAPEX NPV
    # ─────────────────────────────────────────────
    total_capex = (plant_config.total_capacity_mw * 1000 * plant_config.capex_per_kw) + diesel_capex
    capex_schedule = get_construction_schedule(construction_years)

    capex_npv = 0.0
    for year, fraction in capex_schedule:
        discount_year = year + 0.5
        capex_npv += (total_capex * fraction) / ((1 + config.financial.discount_rate) ** discount_year)

    # ─────────────────────────────────────────────
    # OPEX and Energy Flows
    # ─────────────────────────────────────────────
    opex_flows = [0.0] * config.financial.evaluation_years
    energy_flows = [0.0] * config.financial.evaluation_years

    ops_start_year, first_year_fraction = get_operations_start_info(construction_years)

    for year in range(ops_start_year, config.financial.evaluation_years):
        operational_year = year - ops_start_year

        # Apply degradation factors
        cap_factor, eff_factor = get_gas_degradation_factors(
            operational_year,
            plant_config.turbine_class,
            config
        )

        # NG capacity available this year (MW)
        degraded_capacity_mw = plant_config.total_capacity_mw * cap_factor
        ng_possible_mwh = degraded_capacity_mw * plant_config.availability * HOURS_PER_YEAR
        ng_generation_mwh = min(ng_possible_mwh, required_at_generator_mwh)

        # Diesel fills the shortfall
        shortfall_mwh = max(0.0, required_at_generator_mwh - ng_generation_mwh)
        planned_maint_mwh = getattr(plant_config, "eue_maint_mwh", 0.0)
        diesel_capacity_mw = plant_config.diesel_design.total_capacity_mw if plant_config.diesel_design else 0.0
        testing_mwh = diesel_capacity_mw * config.costs.diesel_test_hours_per_year
        diesel_floor_mwh = testing_mwh + planned_maint_mwh
        diesel_generation_mwh = max(shortfall_mwh, diesel_floor_mwh)

        # Fuel costs
        heat_rate_btu_per_kwh = 3412 / (plant_config.efficiency * eff_factor)
        gas_fuel_cost = ng_generation_mwh * (heat_rate_btu_per_kwh / 1000) * gas_price
        diesel_fuel_cost = diesel_generation_mwh * (config.costs.diesel_eff / 1000) * config.costs.diesel_cost

        # O&M costs
        gas_var_om = ng_generation_mwh * plant_config.var_om_per_mwh
        diesel_var_om = diesel_generation_mwh * config.costs.diesel_var_om_per_mwh
        total_fixed_om = (plant_config.total_capacity_mw * 1000 * plant_config.fixed_om_per_kw_yr) + diesel_fixed_om

        # Partial first year adjustment
        year_fraction = first_year_fraction if (year == ops_start_year and first_year_fraction < 1.0) else 1.0

        # Store flows
        opex_flows[year] = (gas_fuel_cost + diesel_fuel_cost +
                            gas_var_om + diesel_var_om +
                            total_fixed_om) * year_fraction

        # LCOE Denominator: Delivered energy to the datacenter
        energy_flows[year] = required_at_datacenter_mwh * year_fraction

    # ─────────────────────────────────────────────
    # NPV and LCOE calculations
    # ─────────────────────────────────────────────
    opex_npv = calculate_npv(opex_flows, config.financial.discount_rate)
    energy_npv = calculate_npv(energy_flows, config.financial.discount_rate)

    total_cost_npv = capex_npv + opex_npv
    lcoe = (total_cost_npv / energy_npv if energy_npv > 0 else float('inf')) / 1000

    return LCOEResult(
        system_type="natural_gas",
        capex_npv=capex_npv,
        opex_npv=opex_npv,
        energy_npv=energy_npv,
        lcoe=lcoe,
        construction_years=construction_years,
        nameplate_capacity_mw=plant_config.total_capacity_mw
    )

def calculate_grid_baseline_lcoe(
   annual_energy_mwh: float,
   required_uptime_pct: float,
   config: Config,
   location: Optional[Tuple[float, float]] = None
) -> LCOEResult:
   """Calculate LCOE for grid electricity baseline"""
   
   # Determine grid parameters based on location
   if location:
       state = get_state_from_coords(location[0], location[1])
       if state and state in GRID_BASELINE_DATA:
           rto, interconnect_years, price_cents, gas_price = GRID_BASELINE_DATA[state]
           electricity_price = price_cents / 100.0  # Convert to $/kWh
           logger.info(f"Using grid data for {state} ({rto}): "
                      f"{interconnect_years} year interconnection, ${electricity_price:.4f}/kWh")
       else:
           # Fallback to defaults
           rto, interconnect_years, price_cents, gas_price = GRID_DEFAULT
           electricity_price = price_cents / 100.0
           logger.info(f"Location not found in US grid data, using defaults: "
                      f"{interconnect_years} year interconnection, ${electricity_price:.4f}/kWh")
   else:
       # No location provided, use defaults
       rto, interconnect_years, price_cents, gas_price = GRID_DEFAULT
       electricity_price = price_cents / 100.0
       logger.info(f"No location provided, using default grid data: "
                  f"{interconnect_years} year interconnection, ${electricity_price:.4f}/kWh")
   
   # No capital costs for grid connection
   capex_npv = 0
   
   # Annual electricity costs
   annual_cost = annual_energy_mwh * electricity_price * 1000  # Convert $/kWh to $/MWh
   
   # Operations start immediately upon interconnection completion
   ops_start_year, first_year_fraction = get_operations_start_info(interconnect_years)
   
   # Build cash flow series
   opex_flows = [0] * config.financial.evaluation_years
   energy_flows = [0] * config.financial.evaluation_years
   
   # First partial year
   if ops_start_year < config.financial.evaluation_years:
       partial_cost = annual_cost * first_year_fraction
       partial_energy = annual_energy_mwh * (required_uptime_pct / 100) * first_year_fraction
       opex_flows[ops_start_year] = partial_cost
       energy_flows[ops_start_year] = partial_energy
   
   # Full operational years
   for year in range(ops_start_year + 1, config.financial.evaluation_years):
       opex_flows[year] = annual_cost
       energy_flows[year] = annual_energy_mwh * (required_uptime_pct / 100)
   
   opex_npv = calculate_npv(opex_flows, config.financial.discount_rate)
   energy_npv = calculate_npv(energy_flows, config.financial.discount_rate)
   
   # LCOE calculation
   lcoe = electricity_price  # Already in $/kWh
   
   return LCOEResult(
       system_type="grid",
       capex_npv=capex_npv,
       opex_npv=opex_npv,
       energy_npv=energy_npv,
       lcoe=lcoe,
       construction_years=interconnect_years,
       nameplate_capacity_mw=0
   )

def calculate_gpu_idling_costs(
   total_gpus: int,
   construction_years: float,
   config: Config
) -> float:
   """Calculate NPV of GPU idling costs during construction/interconnection"""
   
   annual_idling_cost = total_gpus * config.costs.gpu_hour_spot_price * HOURS_PER_YEAR
   
   # Calculate NPV of idling costs
   idling_npv = 0
   for year in range(int(construction_years)):
       idling_npv += annual_idling_cost / ((1 + config.financial.discount_rate) ** year)
   
   # Handle fractional year
   if construction_years % 1 > 0:
       fractional_cost = annual_idling_cost * (construction_years % 1)
       idling_npv += fractional_cost / ((1 + config.financial.discount_rate) ** int(construction_years))
   
   return idling_npv



def get_construction_schedule(construction_years: float) -> List[Tuple[float, float]]:
    """
    Returns construction cash flow schedule.
    Total fractions always sum to 1.0.
    
    Returns: List of (year, fraction) tuples
    """
    if construction_years <= 1.0:
        return [(0.0, 1.0)]
    
    elif construction_years <= 2.0:
        # Always use 40/60 split for anything between 1-2 years
        return [(0.0, 0.4), (1.0, 0.6)]
    
    elif construction_years <= 3.0:
        # Always use 30/40/30 split for anything between 2-3 years
        return [(0.0, 0.3), (1.0, 0.4), (2.0, 0.3)]
    
    else:
        # Longer projects: spread evenly across all years
        full_years = int(construction_years)
        final_fraction = construction_years - full_years
        
        if final_fraction > 0:
            # Fractional final year: distribute evenly across all time
            fraction_per_year = 1.0 / construction_years
            schedule = []
            for year in range(full_years):
                schedule.append((float(year), fraction_per_year))
            # Final partial year gets proportional amount
            schedule.append((float(full_years), fraction_per_year * final_fraction))
        else:
            # Exact number of years: distribute evenly
            fraction_per_year = 1.0 / construction_years
            schedule = [(float(year), fraction_per_year) for year in range(full_years)]
        
        return schedule

def get_operations_start_info(construction_years: float) -> Tuple[int, float]:
    """
    Calculate when operations start and partial year fraction (0-indexed)
    
    Operations begin when construction completes.
    
    Examples:
        construction_years = 1.0 → operations start year 1, full year (1.0)
        construction_years = 1.5 → operations start year 1, half year (0.5)
        construction_years = 2.0 → operations start year 2, full year (1.0)
        construction_years = 2.3 → operations start year 2, partial year (0.7)
    
    Returns:
        (start_year, first_year_fraction)
    """
    completion_year = int(np.ceil(construction_years))  # Year construction finishes (rounded up)
    completion_fraction = construction_years - int(construction_years)  # Fractional part
    
    # Remaining fraction of the year after construction completes
    first_year_fraction = 1.0 - completion_fraction if completion_fraction > 0 else 1.0
    
    return completion_year, first_year_fraction

# ═══════════════════════════════════════════════════════════════════════════
# MAIN COMPARISON FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def compare_datacenter_power_systems(
   total_gpus: int,
   required_uptime_pct: float = 99,
   location: Tuple[float, float] = (31.77, -106.46),  # Default: El Paso, TX
   gas_price: Optional[float] = None,
   config: Optional[Config] = None
) -> SystemComparison:
   """
   Compare LCOE across different power system options for a datacenter.
   
   Args:
       total_gpus: Total number of GPUs in the datacenter
       required_uptime_pct: Required system uptime percentage
       location: (latitude, longitude) tuple
       gas_price: Natural gas price in $/MMBtu
       config: Configuration object
   
   Returns:
       SystemComparison object with results for all systems
   """
   # Load config if not provided
   cfg = config or load_config()
   
   analyzer = DatacenterAnalyzer(
       latitude=location[0],
       longitude=location[1],
       total_gpus=total_gpus
   )
   
   logger.info(f"Comparing power systems for {total_gpus:,} GPU datacenter")
   
   # Get facility load with location-specific hourly PUE
   facility = analyzer.calculate_facility_load(required_uptime_pct=required_uptime_pct)
   
   facility_load_mw = facility.facility_load_design_mw
   annual_energy_mwh = facility.annual_facility_energy_mwh
   
   logger.info(f"Facility requirements: {facility_load_mw:.1f} MW, {annual_energy_mwh:,.0f} MWh/year")
   logger.info(f"Location-specific cooling: Annual PUE={facility.pue:.3f}")
   
   # Step 2: Create SystemCosts objects for different architectures
   ac_costs = SystemCosts(
       solar_cost_per_kw=cfg.costs.solar_cost_y0,
       battery_cost_per_kw=cfg.costs.bess_cost_y0,
       solar_bos_cost_per_kw=cfg.costs.solar_bos_cost_y0_ac,
       battery_bos_cost_per_kw=cfg.costs.battery_bos_cost_y0_ac
   )
   
   dc_costs = SystemCosts(
       solar_cost_per_kw=cfg.costs.solar_cost_y0,
       battery_cost_per_kw=cfg.costs.bess_cost_y0,
       solar_bos_cost_per_kw=cfg.costs.solar_bos_cost_y0_dc,
       battery_bos_cost_per_kw=cfg.costs.battery_bos_cost_y0_dc
   )
   
   # Step 3: Optimize AC-coupled solar+storage for LCOE
   logger.info("Optimizing AC-coupled solar+storage system for minimum LCOE...")
   
   ac_optimizer = MicrogridOptimizer(
       latitude=location[0],
       longitude=location[1],
       facility_load=facility,
       required_uptime_pct=required_uptime_pct,
       costs=ac_costs,
       architecture="ac_coupled",
       efficiency_params=cfg,
       verbose=False
   )
   
   ac_config = ac_optimizer.optimize()
   
   # Step 4: Optimize DC-coupled solar+storage for LCOE
   logger.info("Optimizing DC-coupled solar+storage system for minimum LCOE...")
   
   dc_optimizer = MicrogridOptimizer(
       latitude=location[0],
       longitude=location[1],
       facility_load=facility,
       required_uptime_pct=required_uptime_pct,
       costs=dc_costs,
       architecture="dc_coupled",
       efficiency_params=cfg,
       verbose=False
   )
   
   dc_config = dc_optimizer.optimize()

   # AC-coupled LCOE
   ac_lcoe = calculate_solar_storage_lcoe(
       system_type="ac_coupled",
       solar_mw=ac_config.solar_mw,
       battery_mw=ac_config.battery_mw,
       battery_mwh=ac_config.battery_mwh,
       land_acres=ac_config.land_area_acres,
       sim_year_0=ac_config.sim_year_0,
       sim_year_13=ac_config.sim_year_13,
       sim_year_14=ac_config.sim_year_14,
       sim_year_25=ac_config.sim_year_25,
       year_0_stats=ac_config.year_0_stats,
       construction_years=cfg.design.solar_construction_years,
       required_uptime_pct=required_uptime_pct,
       config=cfg
   )

   # DC-coupled LCOE
   dc_lcoe = calculate_solar_storage_lcoe(
       system_type="dc_coupled",
       solar_mw=dc_config.solar_mw,
       battery_mw=dc_config.battery_mw,
       battery_mwh=dc_config.battery_mwh,
       land_acres=dc_config.land_area_acres,
       sim_year_0=dc_config.sim_year_0,
       sim_year_13=dc_config.sim_year_13,
       sim_year_14=dc_config.sim_year_14,
       sim_year_25=dc_config.sim_year_25,
       year_0_stats=dc_config.year_0_stats,
       construction_years=cfg.design.solar_construction_years,
       required_uptime_pct=required_uptime_pct,
       config=cfg
   )
   
   # Step 5: Optimize natural gas system with location-based pricing
   logger.info("Optimizing natural gas system...")
   from microgrid_optimizer import PowerSystemOptimizer
    
   power_optimizer = PowerSystemOptimizer(
       latitude=location[0],
       longitude=location[1],
       facility_load=facility,
       required_uptime_pct=required_uptime_pct,
       costs=ac_costs,
       efficiency_params=cfg
   )

   # Determine location-specific gas price
   if gas_price is None:
       gas_price = cfg.costs.default_gas_price_mmbtu
   
   location_gas_price = gas_price
   if location:
       state = get_state_from_coords(location[0], location[1])
       if state and state in GRID_BASELINE_DATA:
           _, _, _, state_gas_price = GRID_BASELINE_DATA[state]
           if state_gas_price is not None:
               location_gas_price = state_gas_price
               logger.info(f"Using location-specific gas price for {state}: ${location_gas_price:.2f}/MMBtu")
           else:
               logger.info(f"No gas price data for {state}, using default: ${gas_price:.2f}/MMBtu")
       else:
           logger.info(f"Location not found, using provided gas price: ${gas_price:.2f}/MMBtu")

   # Use location-specific gas price for optimization
   gas_optimization_result, best_gas_config = power_optimizer.optimize_natural_gas(location_gas_price)
   actual_gas_lcoe = calculate_gas_system_lcoe(
                    plant_config=best_gas_config,
                    gas_price=location_gas_price,
                    facility_load=facility,
                    config=cfg
                )
   
   # Use fractional construction years and immediate operational start (I think all this is redundant now)
   construction_years = best_gas_config.construction_timeline['total_months'] / 12.0
   ops_start_year, first_year_fraction = get_operations_start_info(construction_years)

  
   ng_lcoe = LCOEResult(
        system_type="natural_gas",
        capex_npv=actual_gas_lcoe.capex_npv,  # ← Use proper values
        opex_npv=actual_gas_lcoe.opex_npv,    # ← Use proper values  
        energy_npv=actual_gas_lcoe.energy_npv, # ← Use proper values
        lcoe=actual_gas_lcoe.lcoe,
        construction_years=actual_gas_lcoe.construction_years,
        nameplate_capacity_mw=best_gas_config.total_capacity_mw
    )
   
   # Step 6: Calculate grid baseline
   grid_lcoe = calculate_grid_baseline_lcoe(
       annual_energy_mwh, 
       required_uptime_pct,
       cfg,
       location=location
   )
   
   # Log results
   logger.info(f"\nLCOE Results:")
   logger.info(f"  AC Solar+Storage: ${ac_lcoe.lcoe:.4f}/kWh")
   logger.info(f"  DC Solar+Storage: ${dc_lcoe.lcoe:.4f}/kWh")
   logger.info(f"  Natural Gas: ${ng_lcoe.lcoe:.4f}/kWh")
   logger.info(f"  Grid: ${grid_lcoe.lcoe:.4f}/kWh")
   
   return SystemComparison(
       ac_solar=ac_lcoe,
       dc_solar=dc_lcoe,
       natural_gas=ng_lcoe,
       grid_baseline=grid_lcoe,
       total_gpus=total_gpus,
       facility_load_mw=facility_load_mw,
       annual_energy_gwh=annual_energy_mwh / 1000,
       ac_solar_mw=ac_config.solar_mw,
       ac_battery_mw=ac_config.battery_mw,
       ac_battery_mwh=ac_config.battery_mwh,
       dc_solar_mw=dc_config.solar_mw,
       dc_battery_mw=dc_config.battery_mw,
       dc_battery_mwh=dc_config.battery_mwh,
       ng_nameplate_mw=best_gas_config.total_capacity_mw,
       ng_configuration=f"{best_gas_config.n_units}× {best_gas_config.turbine_model} {best_gas_config.cycle_type}"
   )

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def print_lcoe_comparison_table(comparison: SystemComparison, config: Optional[Config] = None):
   """Print table comparing LCOE with and without GPU idling costs"""
   
   cfg = config or load_config()
   
   # Calculate idling costs for each system
   systems = [
       ('AC Solar+Storage', comparison.ac_solar),
       ('DC Solar+Storage', comparison.dc_solar),
       ('Natural Gas', comparison.natural_gas),
       ('Grid', comparison.grid_baseline)
   ]
   
   # Print header
   print(f"\n{'='*80}")
   print(f"LCOE COMPARISON RESULTS")
   print(f"{'='*80}")
   print(f"Datacenter Size: {comparison.total_gpus:,} GPUs")
   print(f"Facility Load: {comparison.facility_load_mw:.1f} MW")
   print(f"Annual Energy: {comparison.annual_energy_gwh:.1f} GWh")
   
   # Print system capacities
   print(f"\nSYSTEM CAPACITIES:")
   print(f"AC Solar+Storage: {comparison.ac_solar_mw:.0f} MW solar / {comparison.ac_battery_mw:.0f} MW ({comparison.ac_battery_mwh:.0f} MWh) battery")
   print(f"DC Solar+Storage: {comparison.dc_solar_mw:.0f} MW solar / {comparison.dc_battery_mw:.0f} MW ({comparison.dc_battery_mwh:.0f} MWh) battery")
   print(f"Natural Gas: {comparison.ng_nameplate_mw:.0f} MW - {comparison.ng_configuration}")
   print(f"Grid: N/A (utility connection)")
   
   # Calculate adjusted LCOEs with idling costs
   print(f"\nLCOE COMPARISON TABLE:")
   print(f"{'-'*80}")
   print(f"{'System':<20} {'Construction':<12} {'Base LCOE':<15} {'Idling Cost':<15} {'Total LCOE':<15}")
   print(f"{'Type':<20} {'Years':<12} {'($/kWh)':<15} {'($M NPV)':<15} {'($/kWh)':<15}")
   print(f"{'-'*80}")
   
   for name, result in systems:
       # Calculate idling cost impact
       idling_cost_npv = calculate_gpu_idling_costs(
           comparison.total_gpus,
           result.construction_years,
           cfg
       )
       
       # Add idling cost to total cost and recalculate LCOE
       if result.energy_npv > 0:
           adjusted_lcoe = (result.capex_npv + result.opex_npv + idling_cost_npv) / result.energy_npv / 1000  # Convert to $/kWh
       else:
           adjusted_lcoe = float('inf')
       
       print(f"{name:<20} {result.construction_years:<12.1f} ${result.lcoe:<14.4f} ${idling_cost_npv/1e6:<14.1f} ${adjusted_lcoe:<14.4f}")
   
   print(f"{'-'*80}")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
   # Load configuration
   config = load_config()
   
   # Example: Compare systems for 10,000 GPU datacenter in El Paso
   comparison = compare_datacenter_power_systems(
       total_gpus=10_000,
       required_uptime_pct=99,
       location=(31.77, -106.46),  # El Paso, TX
       config=config
   )
   
   # Print results
   print_lcoe_comparison_table(comparison, config)