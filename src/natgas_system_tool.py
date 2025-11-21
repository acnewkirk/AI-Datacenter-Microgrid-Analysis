"""
Natural Gas Power Plant Calculator - Refactored to use config.py
This module calculates natural gas plant configurations
"""

import logging
import numpy as np
import math
from math import comb
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from power_systems_estimator import PowerFlowAnalyzer
from it_facil import FacilityLoad
from config import Config, load_config
from degradation_model import get_temperature_derating

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────
# Data Structures
# ───────────────────────────────────────────────────────────────

@dataclass
class TurbineConfig:
   """Base turbine configuration with reliability analysis capabilities"""
   model: str
   turbine_class: str  # 'aero', 'f_class', 'h_class'
   capacity_mw: float
   efficiency: float
   availability: float  # This represents EAF (Equivalent Availability Factor)
   
   def get_maintenance_hours(self, config: Config) -> float:
       """Get maintenance hours from config"""
       return config.design.gas_maintenance_hours[self.turbine_class]
   
   def get_reliability_components(self, config: Config) -> Dict[str, float]:
       """Decompose EAF into forced and planned components"""
       maint_hours = self.get_maintenance_hours(config)
       pof = maint_hours / 8760.0  # Planned Outage Factor
       efor = max(0.0, 1.0 - self.availability - pof)  # Equivalent Forced Outage Rate
       forced_availability = 1.0 - efor
       return {
           'planned_outage_factor': pof,
           'equivalent_forced_outage_rate': efor,
           'forced_availability': forced_availability,
           'maintenance_hours_per_year': maint_hours
       }


@dataclass
class PlantConfiguration:
   """Complete plant configuration with reliability analysis"""
   turbine_model: str
   turbine_class: str
   n_units: int
   cycle_type: str  # 'SC' or 'CC'
   unit_capacity_mw: float
   total_capacity_mw: float
   efficiency: float
   availability: float
   capex_per_kw: float
   fixed_om_per_kw_yr: float
   var_om_per_mwh: float
   nrel_reference: str
   scaling_factors: Dict
   
   # New reliability fields
   maint_hours_per_unit: float = 0.0
   forced_availability: float = 0.0
   eue_forced_mwh: float = 0.0
   eue_maint_mwh: float = 0.0
   eue_total_mwh: float = 0.0
   prob_all_units: float = 0.0  # P(k=N) - all units running
   prob_one_down: float = 0.0   # P(k=N-1) - one unit down
   fuel_btu_yr: float = 0.0
   diesel_design: Optional['SimpleBackupSystemDesign'] = None
   
   # Construction timeline field
   construction_timeline: Optional[Dict[str, int]] = None
   
   def __post_init__(self):
       """Calculate construction timeline after creation if not provided"""
       if self.construction_timeline is None:
           # Will be set by calculate_construction_timeline function
           pass

@dataclass
class FilterConfig:
   """Configuration for deterministic pre-filter"""
   beta_low: float = 0.30   # Min design load factor
   beta_high: float = 0.90  # Max design load factor
 

@dataclass
class SimpleBackupSystemDesign:
   """Simplified backup system design for compatibility"""
   n_gensets_required: int
   n_gensets_total: int
   total_capacity_mw: float
   fuel_storage_gallons: int
   runtime_hours_at_full_load: float
   total_capex: float
   annual_fixed_om: float
   annual_testing_hours: float = 24 # monthly 2 hour run under load 

# ───────────────────────────────────────────────────────────────
# Turbine Library
# ───────────────────────────────────────────────────────────────
TURBINE_LIBRARY = {
   # Aeroderivatives (SC only)
   'SGT_A05': TurbineConfig(model='Siemens SGT-A05 KB7S', turbine_class='aero',
                            capacity_mw = 5.8, efficiency=.323, availability=0.92),

   'LM2500_G4': TurbineConfig(model='GE LM2500+G4', turbine_class='aero', 
                              capacity_mw=35, efficiency=0.39, availability=0.95),
   'Siemens_SGT_A45': TurbineConfig(model='Siemens SGT-A45', turbine_class='aero', 
                            capacity_mw=44, efficiency=0.404, availability=0.94),
   'LM6000_PF': TurbineConfig(model='GE LM6000PC', turbine_class='aero', 
                              capacity_mw=51, efficiency=0.397, availability=0.94),
   'SGT_800': TurbineConfig(model='Siemens SGT-800', turbine_class='aero', 
                            capacity_mw=62, efficiency=0.411, availability=0.92),
   'PW_FT4000': TurbineConfig(model='P&W FT4000 SwiftPac 70', turbine_class='aero', 
                              capacity_mw=71, efficiency=0.410, availability=0.95),
   'LMS100': TurbineConfig(model='GE LMS100', turbine_class='aero', 
                           capacity_mw=100, efficiency=0.395, availability=0.93),

   # F-class Industrial (SC and CC capable)
   'F_class_M501': TurbineConfig(model='Mitsubishi M501F', turbine_class='f_class', 
                                capacity_mw=185.4, efficiency=0.37, availability=0.91),
   'GE_6F.03': TurbineConfig(model='GE 6F.03', turbine_class='f_class', 
                             capacity_mw=88, efficiency=0.368, availability=0.93),
   'GE_7F.04': TurbineConfig(model='GE 7F.04', turbine_class='f_class', 
                             capacity_mw=202, efficiency=0.375, availability=0.93),
   'Siemens_SGT6-5000F': TurbineConfig(model='Siemens SGT6-5000F', turbine_class='f_class', 
                                       capacity_mw=260, efficiency=0.40, availability=0.93),

   # H-class (CC preferred)
   'GE_7HA.01': TurbineConfig(model='GE 7HA.01', turbine_class='h_class', 
                            capacity_mw=290, efficiency=0.42, availability=0.92),
   'H_siemens': TurbineConfig(model='Siemens SGT-8000H', turbine_class='h_class', 
                              capacity_mw=310, efficiency=0.404, availability=0.91),
   'GE_9HA.01': TurbineConfig(model='GE 9HA.01', turbine_class='h_class', 
                              capacity_mw=448, efficiency=0.429, availability=0.91),
   'GE_7HA.02': TurbineConfig(model='GE 7HA.02', turbine_class='h_class', 
                              capacity_mw=384, efficiency=0.426, availability=0.91),
   'GE_9HA.02': TurbineConfig(model='GE 9HA.02', turbine_class='h_class', 
                              capacity_mw=571, efficiency=0.44, availability=0.91),
}

# ───────────────────────────────────────────────────────────────
# STAGE 1: Engineering Pre-Filter Functions
# ───────────────────────────────────────────────────────────────

def passes_engineering_filter(
    plant_config: PlantConfiguration, 
    target_mw: float, 
    filter_config: FilterConfig,
    design_ambient_temp_c: float,  # Correct position
    require_n_minus_1: bool = False,
    avg_demand_mw: Optional[float] = None,
    check_diesel_runtime: bool = False,
    config: Optional[Config] = None
) -> bool:
    """
    Engineering pre-filter with optional diesel runtime check.
    
    Can be called in two modes:
    1. Initial screening (check_diesel_runtime=False) - before EUE calculation
    2. Final validation (check_diesel_runtime=True) - after EUE calculation
    """
    
    # ─────────────────────────────────────────────────────
    # STAGE 1: Basic engineering constraints (no EUE needed)
    # ─────────────────────────────────────────────────────
    
    # Rule 1: Load factor bounds (30-90% at ISO conditions)
    design_load_factor = target_mw / plant_config.total_capacity_mw
    if not (filter_config.beta_low <= design_load_factor <= filter_config.beta_high):
        return False

    # Rule 2: Unit count limits by turbine class and size
    if plant_config.turbine_class == 'aero':
        if plant_config.n_units > 8:
            return False
       
    
    elif plant_config.turbine_class in ['f_class', 'h_class']:
        if plant_config.unit_capacity_mw < 50:
            max_units = 7
        elif plant_config.unit_capacity_mw < 100:
            max_units = 5
        else:
            max_units = 3
        
        if plant_config.n_units > max_units:
            return False
   
     
 
    # Rule 4: Temperature derating check (NEW)
    temp_derate_factor = get_temperature_derating(
            plant_config.turbine_class, 
            design_ambient_temp_c,
            config
        )
    derated_capacity = plant_config.total_capacity_mw * temp_derate_factor
    if derated_capacity < target_mw:
        return False

    # Rule 5: N-1 requirement (MODIFIED to include derating)
    if require_n_minus_1:
        # Check both nominal and derated N-1 capacity
        nominal_n_minus_1_capacity = (plant_config.n_units - 1) * plant_config.unit_capacity_mw
        derated_n_minus_1_capacity = nominal_n_minus_1_capacity * temp_derate_factor

        if nominal_n_minus_1_capacity < target_mw or derated_n_minus_1_capacity < target_mw:
            return False
    
    # ─────────────────────────────────────────────────────
    # STAGE 2: Diesel runtime constraint (requires EUE) (rule 6)
    # ─────────────────────────────────────────────────────
    
    if check_diesel_runtime and plant_config.eue_total_mwh > 0:
        # Only check if we have EUE calculated
        if avg_demand_mw is None:
            avg_demand_mw = target_mw  # Fallback
        
        if config is None:
            from config import load_config
            config = load_config()
        
        # Get max diesel hours (from filter_config or config)
        max_diesel_hours = getattr(
            filter_config, 
            'max_diesel_runtime_hours',
            getattr(config.design, 'max_diesel_runtime_hours', 500)
        )
        
        testing_hours = getattr(config.design, 'diesel_testing_hours', 24)
        
        # Estimate diesel runtime
        if avg_demand_mw > 0:
            diesel_runtime_hours = plant_config.eue_total_mwh / avg_demand_mw
            total_diesel_hours = diesel_runtime_hours + testing_hours
            
            if total_diesel_hours > max_diesel_hours:
                logger.debug(f"Diesel runtime filter: {plant_config.n_units}× "
                           f"{plant_config.turbine_model} exceeds {max_diesel_hours}h limit "
                           f"({total_diesel_hours:.0f}h)")
                return False
    
    return True


# ───────────────────────────────────────────────────────────────
# STAGE 2: EUE Calculation Functions
# ───────────────────────────────────────────────────────────────
def calculate_eue_forced(plant_config: PlantConfiguration, target_mw: float, config: Config) -> float:
   """
   Calculate EUE from random forced outages using proper reliability theory.
   """
   # Find turbine spec by model name
   turbine_spec = None
   for key, turbine in TURBINE_LIBRARY.items():
       if turbine.model == plant_config.turbine_model:
           turbine_spec = turbine
           break
   
   if turbine_spec is None:
       raise KeyError(f"Turbine model '{plant_config.turbine_model}' not found in library")
   
   rel = turbine_spec.get_reliability_components(config)
   
   N = plant_config.n_units
   P_u = plant_config.unit_capacity_mw
   m = math.ceil(target_mw / P_u)  # Units needed online
   A_forced = rel['forced_availability']
   
   eue_forced = 0.0
   
   # Sum over all shortage states (k < m available units)
   for k in range(0, min(m, N + 1)):
       # Probability of exactly k units available
       prob_k = comb(N, k) * (A_forced**k) * ((1-A_forced)**(N-k))
       
       # Shortage magnitude (MW)
       if k < m:
           available_capacity = k * P_u
           required_capacity = min(target_mw, available_capacity + P_u)
           shortfall_mw = max(0, required_capacity - available_capacity)
       else:
           shortfall_mw = 0
       
       # Expected time in this state = probability × total time
       expected_hours_in_state = prob_k * 8760
       
       # Energy unserved in this state
       eue_this_state = shortfall_mw * expected_hours_in_state
       eue_forced += eue_this_state
   
   return eue_forced  # MWh/year

def calculate_eue_planned(plant_config: PlantConfiguration, target_mw: float, config: Config) -> float:
   """Calculate EUE from planned maintenance (deterministic, staggered)"""
   # Find turbine spec by model name
   turbine_spec = None
   for key, turbine in TURBINE_LIBRARY.items():
       if turbine.model == plant_config.turbine_model:
           turbine_spec = turbine
           break
   
   if turbine_spec is None:
       raise KeyError(f"Turbine model '{plant_config.turbine_model}' not found in library")
   
   maint_hours_per_unit = turbine_spec.get_maintenance_hours(config)
   units_needed = math.ceil(target_mw / plant_config.unit_capacity_mw)
   
   if plant_config.n_units <= units_needed:
       # No spare capacity during maintenance
       shortfall_per_maintenance = min(plant_config.unit_capacity_mw, target_mw)
       # Staggered maintenance: each unit down for maint_hours_per_unit
       eue_maintenance = shortfall_per_maintenance * maint_hours_per_unit * plant_config.n_units
   else:
       # Have spare capacity for maintenance
       eue_maintenance = 0.0
   
   return eue_maintenance  # MWh/year

def calculate_two_state_probabilities(plant_config: PlantConfiguration, config: Config) -> Tuple[float, float]:
   """Calculate two-state operating probabilities for fuel modeling"""
   # Find turbine spec by model name
   turbine_spec = None
   for key, turbine in TURBINE_LIBRARY.items():
       if turbine.model == plant_config.turbine_model:
           turbine_spec = turbine
           break
   
   if turbine_spec is None:
       raise KeyError(f"Turbine model '{plant_config.turbine_model}' not found in library")
   
   rel = turbine_spec.get_reliability_components(config)
   A_forced = rel['forced_availability']
   N = plant_config.n_units
   
   # State 1: All units running
   prob_all_units = A_forced ** N
   
   # State 2: One unit down (probability of exactly N-1 units running)
   prob_one_down = N * (1 - A_forced) * (A_forced ** (N-1))
   
   return prob_all_units, prob_one_down

# ───────────────────────────────────────────────────────────────
# Part-Load Performance Curves
# ───────────────────────────────────────────────────────────────

def calculate_part_load_efficiency(
    turbine_class: str, 
    load_factor: float,
    config: Optional[Config] = None
) -> float:
    """Calculate part-load efficiency multiplier based on turbine class"""
    cfg = config or load_config()
    load_factor = max(0.3, min(1.0, load_factor))  # Clamp to operating range
    
    if turbine_class == 'aero':
        penalty = cfg.gas_turbine.part_load_penalty_aero
    elif turbine_class == 'f_class':
        penalty = cfg.gas_turbine.part_load_penalty_f_class
    elif turbine_class == 'h_class':
        penalty = cfg.gas_turbine.part_load_penalty_h_class
    else:
        raise ValueError(f"Unknown turbine class: {turbine_class}")
    
    return 1.0 - penalty * (1.0 - load_factor)

def calculate_cc_capacity(gt_capacity_mw: float, gt_efficiency: float, config: Config) -> float:
    """
    Calculate combined cycle output from gas turbine specs.
    
    For combined cycle plants:
    1. Gas turbine converts fuel to electricity at its stated efficiency
    2. Exhaust heat is recovered by HRSG (typically 80-85% recovery)
    3. Steam cycle converts recovered heat to electricity at ~38% efficiency
    """
    # Calculate fuel input and waste heat
    fuel_input_mw = gt_capacity_mw / gt_efficiency
    gt_exhaust_heat_mw = fuel_input_mw - gt_capacity_mw
    
    # HRSG effectiveness depends on turbine class and exhaust temperature in practice, could improve this later
    # Typical exhaust temps:
    # H-class turbines have higher exhaust temps (~630°C) allowing better heat recovery
    # F-class turbines have moderate exhaust temps (~580°C)
    # Aeroderivatives have lower exhaust temps (~450-500°C)
    
    # You could make this configurable or turbine-class specific
    hrsg_effectiveness = config.efficiency.hrsg_effectiveness  # Conservative estimate for mixed fleet (.80 and .34)
    
    # Calculate steam turbine output
    recovered_heat_mw = gt_exhaust_heat_mw * hrsg_effectiveness
    steam_output_mw = recovered_heat_mw * config.efficiency.steam_cycle_efficiency
    
    return gt_capacity_mw + steam_output_mw

def map_to_nrel_reference(turbine_class: str, cycle_type: str, 
                        plant_capacity: float) -> str:
   """Map configuration to NREL reference"""
   if turbine_class == 'aero':
       return 'aero_sc'
   elif turbine_class == 'f_class':
       return 'f_class_sc' if cycle_type == 'SC' else 'f_class_cc'
   elif turbine_class == 'h_class':
       return 'h_class_cc'
   else:
       raise ValueError(f"Unknown turbine class: {turbine_class}")

def calculate_scaled_costs(base_costs: Dict, plant_capacity: float, 
                         reference_capacity: float, n_units: int) -> Dict:
   """Apply scaling laws and multi-unit penalties to base costs"""
   size_ratio = plant_capacity / reference_capacity
   
   if n_units <= 2:
       multi_unit_penalty = 1.0  # Standard configuration
   elif n_units <= 4:
       multi_unit_penalty = 1.3  # Moderate complexity
   else:
       multi_unit_penalty = 1.8  # High complexity - rare in practice
   
   # Scale capex with 0.67 power law
   scaled_capex = base_costs['capex_per_kw'] * (size_ratio ** -0.33) * multi_unit_penalty
   
   # Scale fixed O&M with 0.8 power law
   scaled_fixed_om = base_costs['fixed_om_per_kw'] * (size_ratio ** -0.2) * multi_unit_penalty
   
   # Variable O&M doesn't scale with size
   scaled_var_om = base_costs['var_om_per_mwh']
   
   return {
       'capex_per_kw': scaled_capex,
       'fixed_om_per_kw': scaled_fixed_om,
       'var_om_per_mwh': scaled_var_om,
       'scaling_factors': {
           'size_ratio': size_ratio,
           'capex_scale_factor': size_ratio ** -0.33,
           'fixed_om_scale_factor': size_ratio ** -0.2,
           'multi_unit_penalty': multi_unit_penalty
       }
   }

# ───────────────────────────────────────────────────────────────
# Construction Timeline Calculation
# ───────────────────────────────────────────────────────────────

def calculate_construction_timeline(turbine_class: str, n_units: int, 
                                  combined_cycle: bool, config: Config) -> Dict[str, int]:
   """Calculate construction timeline with turbine lead times from config"""
   # Base construction time
   if combined_cycle:
       construction_time = 42  # 3.5 years for CC
       post_turbine_fraction = 0.30  # 30% must occur after turbine receipt
   else:
       construction_time = 40  # 40 months for SC
       post_turbine_fraction = 0.30  # 30% must occur after turbine receipt
   
   # Get turbine lead time from config, if it can't find the turbine class default to 18 months
   turbine_lead_time = config.design.turbine_lead_times.get(turbine_class, 18)
   
   # Calculate post-turbine construction time
   post_turbine_construction = construction_time * post_turbine_fraction
   
   # Total timeline is max of:
   # 1. Full construction time, or 
   # 2. Turbine lead time + post-turbine construction work
   total_months = max(construction_time, 
                     turbine_lead_time + post_turbine_construction)

   return {
       'turbine_lead_time': turbine_lead_time,
       'construction_time': construction_time,
       'total_months': total_months,
       'construction_years': total_months / 12
   }

# ───────────────────────────────────────────────────────────────
# Configuration Generation with Two-Stage Analysis
# ───────────────────────────────────────────────────────────────

def generate_plant_configurations(
    target_mw: float,
    turbine_library: Dict[str, TurbineConfig],
    design_ambient_temp_c: float,  
    require_n_minus_1: bool = True,
    filter_config: Optional[FilterConfig] = None,
    config: Optional[Config] = None,
    annual_energy_mwh: Optional[float] = None,  # for getting avg values
    apply_diesel_runtime_filter: bool = False,  # Enable diesel runtime filtering
    max_diesel_runtime_hours: Optional[float] = None  # Override config value if needed
) -> List[PlantConfiguration]:
    """
    Generate plant configurations using two-stage reliability analysis.
    Now includes diesel backup sizing and runtime filtering.
    """
    cfg = config or load_config()
    
    if filter_config is None:
        filter_config = FilterConfig()
    
    configurations = []
    
    # Calculate average MW from annual energy
    if annual_energy_mwh is not None:
        avg_demand_mw = annual_energy_mwh / 8760
    else:
        avg_demand_mw = target_mw  # Fallback to peak (backward compatibility)

    for turbine_id, turbine in turbine_library.items():
        # Determine valid cycle types
        if turbine.turbine_class == 'aero':
            cycle_types = ['SC']
        else:
            cycle_types = ['SC', 'CC']
        
        for cycle_type in cycle_types:
            # Calculate unit capacity
            if cycle_type == 'CC':
                unit_capacity = calculate_cc_capacity(turbine.capacity_mw, turbine.efficiency, cfg)
                fuel_input_mw = turbine.capacity_mw / turbine.efficiency
                cc_efficiency = unit_capacity / fuel_input_mw
            else:
                unit_capacity = turbine.capacity_mw
                cc_efficiency = turbine.efficiency
            
            # Generate configurations with different unit counts
            min_units = max(1, int(np.ceil(target_mw / unit_capacity)))
            
            for n_units in range(min_units, min_units + 8):
                total_capacity = n_units * unit_capacity
                
                # Stop at excessive oversizing
                if total_capacity > 2.0 * target_mw:
                    break
                
                # Calculate load factor and operating efficiency
                normal_load_factor = avg_demand_mw / total_capacity
                part_load_multiplier = calculate_part_load_efficiency(
                    turbine.turbine_class, normal_load_factor, cfg
                )
                operating_efficiency = cc_efficiency * part_load_multiplier
                
                # Map to NREL reference and calculate costs
                nrel_ref_key = map_to_nrel_reference(turbine.turbine_class, cycle_type, total_capacity)
                nrel_ref = cfg.costs.ng_costs[nrel_ref_key]
                scaled_costs = calculate_scaled_costs(nrel_ref, total_capacity, nrel_ref['capacity_mw'], n_units)
                
                # Create preliminary configuration
                plant_config = PlantConfiguration(
                    turbine_model=turbine.model,
                    turbine_class=turbine.turbine_class,
                    n_units=n_units,
                    cycle_type=cycle_type,
                    unit_capacity_mw=unit_capacity,
                    total_capacity_mw=total_capacity,
                    efficiency=operating_efficiency,
                    availability=turbine.availability,
                    capex_per_kw=scaled_costs['capex_per_kw'],
                    fixed_om_per_kw_yr=scaled_costs['fixed_om_per_kw'],
                    var_om_per_mwh=scaled_costs['var_om_per_mwh'],
                    nrel_reference=nrel_ref_key,
                    scaling_factors=scaled_costs['scaling_factors']
                )
                
                # Set construction timeline
                plant_config.construction_timeline = calculate_construction_timeline(
                    turbine.turbine_class, n_units, cycle_type == 'CC', cfg
                )
                
                # STAGE 1: Apply engineering pre-filter
                if not passes_engineering_filter(plant_config, target_mw, filter_config, design_ambient_temp_c, require_n_minus_1):
                    continue
                
                # STAGE 2: Calculate EUE components and reliability metrics
                rel_components = turbine.get_reliability_components(cfg)
                plant_config.maint_hours_per_unit = rel_components['maintenance_hours_per_year']
                plant_config.forced_availability = rel_components['forced_availability']
                plant_config.eue_forced_mwh = calculate_eue_forced(plant_config, avg_demand_mw, cfg) #may swap to target 
                plant_config.eue_maint_mwh = calculate_eue_planned(plant_config, avg_demand_mw, cfg)
                plant_config.eue_total_mwh = plant_config.eue_forced_mwh + plant_config.eue_maint_mwh
                
                # STAGE 3: Size diesel backup based on EUE (NEW - using config)
                plant_config.diesel_design = size_diesel_backup_from_eue(
                    max(plant_config.eue_total_mwh, 0.0),  # still use EUE for fuel sizing
                    target_mw,  # size to cover peak load
                    cfg
                )

                # STAGE 4: Apply diesel runtime filter if enabled
                if apply_diesel_runtime_filter:
                    max_hours = max_diesel_runtime_hours or getattr(cfg.design, 'max_diesel_runtime_hours', 200)
                    diesel_runtime_hours = plant_config.eue_total_mwh / avg_demand_mw
                    testing_hours = cfg.costs.diesel_test_hours_per_year
                    total_diesel_hours = diesel_runtime_hours + testing_hours

                    if total_diesel_hours > max_hours:
                        logger.debug(f"Rejected {n_units}× {turbine.model} {cycle_type}: "
                                     f"Diesel runtime {total_diesel_hours:.0f}h > {max_hours}h limit")
                        continue
                
                # Calculate two-state probabilities
                plant_config.prob_all_units, plant_config.prob_one_down = calculate_two_state_probabilities(plant_config, cfg)
                
                # Calculate expected fuel consumption (simplified)
                heat_rate_btu_kwh = 3412 / operating_efficiency
                plant_config.fuel_btu_yr = target_mw * 1000 * 8760 * heat_rate_btu_kwh
                
                # Add load factor info to scaling factors
                plant_config.scaling_factors['normal_load_factor'] = normal_load_factor
                plant_config.scaling_factors['part_load_multiplier'] = part_load_multiplier
                plant_config.scaling_factors['eue_total_mwh'] = plant_config.eue_total_mwh
                
                configurations.append(plant_config)
                
                # #logger.debug(f"Generated config: {n_units}× {turbine.model} {cycle_type} = "
                #            f"{total_capacity:.0f} MW @ {normal_load_factor:.1%} load, "
                #            f"EUE: {plant_config.eue_total_mwh:.0f} MWh/yr, "
                #            f"Diesel: {plant_config.diesel_design.total_capacity_mw if plant_config.diesel_design else 0:.0f} MW")
    
    return configurations

# ───────────────────────────────────────────────────────────────
# Diesel Backup System Design
# ───────────────────────────────────────────────────────────────


   
def size_diesel_backup_from_eue(
    eue_mwh_per_year: float, 
    target_mw: float,
    config: Optional[Config] = None
) -> SimpleBackupSystemDesign:
    """Size diesel backup based on Expected Unserved Energy analysis"""
    cfg = config or load_config()
    
    # Assume diesel gensets are 2 MW each (typical size)
    genset_capacity_mw = 2.0
    
    # Size for target power delivery
    n_gensets_required = math.ceil(target_mw / genset_capacity_mw)
    
    # Add one spare genset for N-1 capability
    n_gensets_total = n_gensets_required + 1
    total_capacity_mw = n_gensets_total * genset_capacity_mw
    
    # Size fuel storage for EUE coverage plus some margin
    # Use diesel efficiency from config
    gallons_per_mwh = cfg.costs.diesel_eff  # Now from config
    safety_factor = 2.0  # 2x margin for uncertainty
    fuel_storage_gallons = int(eue_mwh_per_year * gallons_per_mwh * safety_factor)
    
    # Minimum storage: 72 hours at full load
    min_storage_gallons = int(target_mw * 1000 * 72 * (cfg.costs.diesel_eff / 1000))
    fuel_storage_gallons = max(fuel_storage_gallons, min_storage_gallons)
    
    # Runtime at full load
    runtime_hours = fuel_storage_gallons / (target_mw * 1000 * (cfg.costs.diesel_eff / 1000))
    
    # Cost estimates using config values
    genset_cost_per_kw = cfg.costs.diesel_genset_cost_per_kw  # From config
    fuel_tank_cost = fuel_storage_gallons * cfg.costs.diesel_tank_cost_per_gallon  # From config
    total_capex = total_capacity_mw * 1000 * genset_cost_per_kw + fuel_tank_cost
    
    # O&M: 6% of capex annually (could also move this to config)
    annual_fixed_om = total_capex * 0.06
    
    return SimpleBackupSystemDesign(
        n_gensets_required=n_gensets_required,
        n_gensets_total=n_gensets_total,
        total_capacity_mw=total_capacity_mw,
        fuel_storage_gallons=fuel_storage_gallons,
        runtime_hours_at_full_load=runtime_hours,
        total_capex=total_capex,
        annual_fixed_om=annual_fixed_om
    )

# ───────────────────────────────────────────────────────────────
# Natural Gas Power Plant Calculator
# ───────────────────────────────────────────────────────────────

class NGPowerPlantCalculator:
   """Natural gas power plant sizing and configuration calculator."""
   
   def __init__(
       self, 
       facility_load: FacilityLoad,
       required_uptime_pct: float = 99.0,
       gas_price_mmbtu: Optional[float] = None,
       include_backup: bool = True,
       filter_config: Optional[FilterConfig] = None,
       efficiency_params: Optional[Config] = None
   ):
       """
       Initialize calculator with FacilityLoad object.
       
       Args:
           facility_load: FacilityLoad object from datacenter analysis
           required_uptime_pct: Required system uptime percentage
           gas_price_mmbtu: Natural gas price in $/MMBtu
           include_backup: Whether to include diesel backup sizing
           filter_config: Configuration for deterministic pre-filter
           efficiency_params: Config object with efficiency parameters
       """
       # Load config
       self.config = efficiency_params or load_config()
       
       power_analyzer = PowerFlowAnalyzer(self.config)
       mult = power_analyzer.get_bus_architecture_multipliers("natural_gas")

       # Calculate the total power demanded at bus
       total_bus_demand_mw = (facility_load.it_load_design_mw * mult['bus_to_it'] + 
                              facility_load.cooling_load_design_mw * mult['bus_to_cooling'])
       
       # "Gross up" the bus demand to find the required generation at source
       self.required_generation_mw = total_bus_demand_mw * mult['grid_to_bus']

       # Do the analogous calculation for annual energy
       total_bus_demand_mwh = (facility_load.annual_it_energy_mwh * mult['bus_to_it'] + 
                               facility_load.annual_cooling_energy_mwh * mult['bus_to_cooling'])
       
       self.annual_energy_mwh = total_bus_demand_mwh * mult['grid_to_bus']
       
       self.required_uptime_pct = required_uptime_pct
       self.gas_price_mmbtu = gas_price_mmbtu or self.config.costs.default_gas_price_mmbtu
       self.include_backup = include_backup
       self.filter_config = filter_config or FilterConfig()
       
       # Validate inputs
       # if self.required_generation_mw <= 0:
       #     raise ValueError("Required generation must be positive")
       # if not 90 <= self.required_uptime_pct <= 99.99:
       #     raise ValueError("Uptime percentage must be between 90% and 99.99%")
   
   def calculate_plant_parameters(self, turbine_model: str, combined_cycle: bool = False,
                                gas_price_mmbtu: Optional[float] = None) -> Dict:
       """
       Calculate plant parameters for specific configuration (backward compatibility).
       """
       if turbine_model not in TURBINE_LIBRARY:
           raise KeyError(f"Turbine model '{turbine_model}' not found")
   
       turbine = TURBINE_LIBRARY[turbine_model]
       cycle_type = 'CC' if combined_cycle else 'SC'
       gas_price = gas_price_mmbtu or self.gas_price_mmbtu
   
       # Check if configuration is valid
       if turbine.turbine_class == 'aero' and combined_cycle:
           raise ValueError("Aeroderivative turbines cannot operate in combined cycle")
   
       # Calculate unit capacity
       if combined_cycle:
           unit_capacity = calculate_cc_capacity(turbine.capacity_mw, turbine.efficiency, self.config)
       else:
           unit_capacity = turbine.capacity_mw
   
       # Use EUE analysis to determine optimal unit count
       configs = []
       min_units = max(1, int(np.ceil(self.required_generation_mw / unit_capacity)))
   
       for n_units in range(min_units, min_units + 6):
           total_capacity = n_units * unit_capacity
           normal_load_factor = self.required_generation_mw / total_capacity
       
           if combined_cycle:
               fuel_input_mw = turbine.capacity_mw / turbine.efficiency
               cc_efficiency = unit_capacity / fuel_input_mw
           else:
               cc_efficiency = turbine.efficiency
       
           part_load_multiplier = calculate_part_load_efficiency(
               turbine.turbine_class, normal_load_factor
           )
           operating_efficiency = cc_efficiency * part_load_multiplier
       
           # Map to NREL reference and calculate costs
           nrel_ref_key = map_to_nrel_reference(turbine.turbine_class, cycle_type, total_capacity)
           nrel_ref = self.config.costs.ng_costs[nrel_ref_key]
           scaled_costs = calculate_scaled_costs(nrel_ref, total_capacity, nrel_ref['capacity_mw'], n_units)
       
           config = PlantConfiguration(
               turbine_model=turbine.model,
               turbine_class=turbine.turbine_class,
               n_units=n_units,
               cycle_type=cycle_type,
               unit_capacity_mw=unit_capacity,
               total_capacity_mw=total_capacity,
               efficiency=operating_efficiency,
               availability=turbine.availability,
               capex_per_kw=scaled_costs['capex_per_kw'],
               fixed_om_per_kw_yr=scaled_costs['fixed_om_per_kw'],
               var_om_per_mwh=scaled_costs['var_om_per_mwh'],
               nrel_reference=nrel_ref_key,
               scaling_factors=scaled_costs['scaling_factors']
           )
           
           # Set construction timeline
           config.construction_timeline = calculate_construction_timeline(
               turbine.turbine_class, n_units, combined_cycle, self.config
           )
       
           # Apply engineering filter
           filter_cfg = FilterConfig()
           if passes_engineering_filter(config, self.required_generation_mw, filter_cfg):
               # Calculate EUE components
               rel_components = turbine.get_reliability_components(self.config)
               config.forced_availability = rel_components['forced_availability']
               config.eue_forced_mwh = calculate_eue_forced(config, self.required_generation_mw, self.config)
               config.eue_maint_mwh = calculate_eue_planned(config, self.required_generation_mw, self.config)
               config.eue_total_mwh = config.eue_forced_mwh + config.eue_maint_mwh
               config.prob_all_units, config.prob_one_down = calculate_two_state_probabilities(config, self.config)
           
               configs.append(config)
   
       if not configs:
           raise ValueError("No valid configurations found for specified turbine")
   
       # Select configuration with minimum EUE (most reliable)
       optimal_config = min(configs, key=lambda c: c.eue_total_mwh)
   
       # Calculate fuel cost (basic calculation without LCOE)
       heat_rate_btu_kwh = 3412 / optimal_config.efficiency
       fuel_cost_per_mwh = (heat_rate_btu_kwh / 1000) * gas_price
   
       # Size diesel backup
       if self.include_backup:
           diesel_design = size_diesel_backup_from_eue(
               optimal_config.eue_total_mwh,
               self.required_generation_mw, 
               self.config
           )
           optimal_config.diesel_design = diesel_design
   
       # Build result dictionary (maintaining backward compatibility)
       result = {
           'fleet_description': f"{optimal_config.n_units}× {optimal_config.turbine_model} {cycle_type}",
           'nameplate_capacity_mw': optimal_config.total_capacity_mw,
           'required_capacity_mw': self.required_generation_mw,
           'efficiency_hhv': optimal_config.efficiency / self.config.efficiency.lhv_to_hhv,
           'efficiency_lhv': optimal_config.efficiency,
           'unit_capacity_mw': optimal_config.unit_capacity_mw,
           'n_units': optimal_config.n_units,
           'capex_per_kw': optimal_config.capex_per_kw,
           'fixed_om_per_kw_yr': optimal_config.fixed_om_per_kw_yr,
           'var_om_per_mwh': optimal_config.var_om_per_mwh,
           'fuel_cost_per_mwh': fuel_cost_per_mwh,
           'heat_rate_btu_kwh': heat_rate_btu_kwh,
           'combined_cycle': combined_cycle,
           'total_nominal_capex': optimal_config.capex_per_kw * optimal_config.total_capacity_mw * 1000,
           'system_reliability': optimal_config.availability * 100,
           'nrel_reference': optimal_config.nrel_reference,
           'scaling_applied': optimal_config.scaling_factors,
           'construction_years': optimal_config.construction_timeline['construction_years'],
           'construction_months': optimal_config.construction_timeline['total_months'],
           'turbine_lead_time_months': optimal_config.construction_timeline['turbine_lead_time'],
       
           # New EUE-based reliability metrics
           'eue_forced_mwh_per_year': optimal_config.eue_forced_mwh,
           'eue_maintenance_mwh_per_year': optimal_config.eue_maint_mwh,
           'eue_total_mwh_per_year': optimal_config.eue_total_mwh,
           'forced_availability': optimal_config.forced_availability,
           'prob_all_units_running': optimal_config.prob_all_units,
           'prob_one_unit_down': optimal_config.prob_one_down,
       }
   
       # Add diesel backup info if included
       if self.include_backup and optimal_config.diesel_design:
           result.update({
               'diesel_gensets_required': optimal_config.diesel_design.n_gensets_required,
               'diesel_gensets_total': optimal_config.diesel_design.n_gensets_total,
               'diesel_capacity_mw': optimal_config.diesel_design.total_capacity_mw,
               'diesel_fuel_storage_gallons': optimal_config.diesel_design.fuel_storage_gallons,
               'diesel_runtime_hours': optimal_config.diesel_design.runtime_hours_at_full_load,
               'diesel_capex': optimal_config.diesel_design.total_capex,
               'diesel_annual_om': optimal_config.diesel_design.annual_fixed_om,
           })
   
       result['delivered_to_it_mw'] = self.facility_design_mw
       result['required_generation_mw'] = self.required_generation_mw
   
       return result

# ───────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ───────────────────────────────────────────────────────────────

# Add this as the main function at the bottom of natgas_system_tool.py

if __name__ == "__main__":
    """Verify the EXACT demand values being used for EUE calculations"""
    
    print("\n" + "="*70)
    print("TEST: Verify Exact Demand Values Used in EUE")
    print("="*70)
    
    from config import load_config
    
    # Create a simple plant config
    plant = PlantConfiguration(
        turbine_model="Test",
        turbine_class="f_class",
        n_units=2,
        unit_capacity_mw=200,
        total_capacity_mw=400,
        cycle_type="SC",
        efficiency=0.40,
        availability=0.97,
        forced_availability=0.97,
        capex_per_kw=1000,
        fixed_om_per_kw_yr=25,
        var_om_per_mwh=5
    )
    
    # Test with known values
    peak_mw = 300.0
    avg_mw = 225.0  # 75% of peak
    
    cfg = load_config()
    
    # Direct EUE calculations
    eue_at_peak = calculate_eue_forced(plant, peak_mw, cfg)
    eue_at_avg = calculate_eue_forced(plant, avg_mw, cfg)
    
    print(f"\nDirect EUE calculation test:")
    print(f"  Plant: {plant.n_units}×{plant.unit_capacity_mw} MW = {plant.total_capacity_mw} MW")
    print(f"  Peak demand: {peak_mw} MW → EUE = {eue_at_peak:.0f} MWh/yr")
    print(f"  Avg demand:  {avg_mw} MW → EUE = {eue_at_avg:.0f} MWh/yr")
    print(f"  Reduction: {(1-eue_at_avg/eue_at_peak)*100:.1f}%")
    
    # Now test through generate_plant_configurations
    annual_mwh = avg_mw * 8760
    
    print(f"\nThrough generate_plant_configurations:")
    print(f"  If annual_energy_mwh = {annual_mwh:.0f}")
    print(f"  Then avg_demand_mw should = {annual_mwh/8760:.0f} MW")
    
    # The key question: Does generate_plant_configurations use this correctly?
    # We can't easily test this without modifying the function to log/return the value
    
    print("\n⚠️  To fully verify, we'd need to add logging inside")
    print("    generate_plant_configurations to confirm avg_demand_mw value")