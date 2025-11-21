"""
Configuration module for datacenter power systems analysis.
Centralizes all modeling parameters and assumptions.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import json


@dataclass
class CostConfig:
   """Capital and operating cost parameters (y2022 USD)"""
   # Solar costs -- sourced from NREL 2024 ATB
   solar_cost_y0: float = 402  # $/kW DC for modules
   solar_bos_cost_y0_ac: float = 851.64  # $/kW for solar share of BOS in AC systems
   solar_bos_cost_y0_dc: float = 598.74  # $/kW for solar share of BOS in DC systems
   
   # Battery costs
   bess_cost_y0: float = 826  # $/kW for energy storage system
   battery_bos_cost_y0_ac: float = 195.62  # $/kW for battery share of BOS in AC systems
   battery_bos_cost_y0_dc: float = 195.62  # $/kW for battery share of BOS in DC systems
   
   # Future costs (Year 15 moderate projection from NREL)
   solar_cost_y15: float = 247.4109  # $/kW DC
   bess_cost_y15: float = 508.1235  # $/kW
   bos_cost_y15: float = 644.4656  # $/kW combined
   
   # O&M costs The NREL ATB value includes the cost of battery replacement in y15 so we decompose it using the PNNL value for the LFP and Ramasamy et al. value for the PV
   solar_fixed_om: float = 16.58  # $/kW DC-yr includes the new inverters but since they don't decompse cost don't want to try to break it out and it shouldn't matter 
   storage_fixed_om: float = 4.37 # $/kW-yr for 4 hr lfp battery storage (PNNL 2024)
   # Land costs
   land_cost_per_sq_km: float = 150000  # $/km^2 (Year 0)
   
   # Natural gas costs (NREL 2024 ATB, or EIA Capital Costs and Performance Characteristics )
   ng_costs: Dict[str, Dict] = field(default_factory=lambda: {
       'aero_sc': {'capacity_mw': 211, 'capex_per_kw': 1510, 'fixed_om_per_kw': 35, 'var_om_per_mwh': 8},
       'f_class_sc': {'capacity_mw': 233, 'capex_per_kw': 1319, 'fixed_om_per_kw': 25.7, 'var_om_per_mwh': 6.94},
       'f_class_cc': {'capacity_mw': 727, 'capex_per_kw': 1455, 'fixed_om_per_kw': 33, 'var_om_per_mwh': 2.1},
       'h_class_cc': {'capacity_mw': 800, 'capex_per_kw': 1600, 'fixed_om_per_kw': 36, 'var_om_per_mwh': 2.1}
   })
   
   # Fuel and opportunity costs
   default_gas_price_mmbtu: float = 3.5  # $/MMBtu natural gas
   gpu_hour_spot_price: float = 2.40  # $/GPU-hour
   dgx_node_price: float = 350000  # $/DGX H100 node (8 GPUs)
   
   diesel_genset_cost_per_kw: float = 800
   diesel_tank_cost_per_gallon: float = 3.00  # $/gallon for diesel storage tank
   diesel_cost = 3.40 # $/gallon, USA avg diesel price as of july 2025, adjusted to 2022 dollars. 
   diesel_eff = 250 # gallons per MWh, average diesel generator efficiency NREL backup gen doc
   diesel_var_om_per_mwh   = 12.0    # $/MWh non-fuel
   diesel_test_hours_per_year: float = 24.0  # default annual testing hours per diesel unit. Based on 2 hours of monthly testing per gen



@dataclass
class EfficiencyConfig:
   """Power conversion and system efficiency parameters"""
   # Power conversion stages
   dc_dc: float = 0.97  # DC-DC conversion efficiency
   ac_dc: float = 0.96  # AC-DC rectification efficiency
   dc_ac: float = 0.96  # DC-AC inversion efficiency
   transformer: float = 0.98
   switchgear: float = 0.99
   ups: float = 0.94  # Double conversion UPS (critical for both IT and cooling)
   
   # IT-specific efficiencies
   dc_psu_efficiency: float = 0.98  # OCP-style DC PSUs
   ac_psu_efficiency: float = 0.94  # Traditional AC PSUs
   
   # Cooling-specific efficiencies
   vfd_efficiency: float = 0.97  # Variable Frequency Drive for AC cooling motors
   dc_motor_controller_efficiency: float = 1  # Electronic speed controller for DC motors
   
   # Distribution
   cable_efficiency: float = 0.99  # Distribution losses
   pdu_efficiency: float = 0.99  # Power Distribution Unit efficiency
   
   # Battery and solar
   battery_rte: float = 0.90  # Round-trip efficiency for LFP batteries
   inverter_load_ratio: float = 1.2 # ILR sourced from NREL ATB
   inverter_efficiency: float = 0.96  # Separate out inverter efficiency
   mppt_efficiency: float = 0.99  # MPPT efficiency
   battery_converter: float = 0.98  # Battery DC-DC converter efficiency
   
   
   # Natural gas efficiencies 
   steam_cycle_efficiency: float = 0.34
   lhv_to_hhv: float = 1.108
   hrsg_effectiveness: float = 0.80

@dataclass
class GasTurbineConfig:
    """Gas turbine performance parameters"""
    # Part-load efficiency penalty coefficients
    # At full load: efficiency = baseline
    # At reduced load: efficiency = baseline × (1 - penalty × (1 - load_factor))
    part_load_penalty_aero: float = 0.10      # Aeroderivatives: best part-load
    part_load_penalty_f_class: float = 0.15   # F-class: moderate penalty
    part_load_penalty_h_class: float = 0.20   # H-class: optimized for baseload
    
    # Temperature derating coefficients (capacity loss per °C above 15°C ISO baseline)
    temp_derating_per_c_aero: float = 0.010    # 1.0% per °C
    temp_derating_per_c_f_class: float = 0.008 # 0.8% per °C
    temp_derating_per_c_h_class: float = 0.007 # 0.7% per °C


@dataclass
class ITLoadConfig:
   """IT facility load parameters"""
   gpus_per_node: int = 8  # Number of GPUs per compute node (e.g., DGX H100)
   node_power_avg_kw: float = 7.3  # Average power per node during training (includes interconnect, based on Newkirk et al.)
   node_power_max_kw: float = 8.5  # Maximum power per node during training (includes interconnect)
   design_contingency_factor: float = 1.05  # 5% design margin for facility sizing
   default_pue: float = 1.2  # Default Power Usage Effectiveness
   default_required_uptime_pct: float = 99  # Default required uptime percentage


@dataclass
class DesignConfig:
   """System design and layout parameters"""
   # Land use factors
   solar_acres_per_mw: float = 4.17  # Typical for single-axis tracking (LBNL paper)
   battery_acres_per_mwh: float = 0.25  # Typical for grid-scale BESS
   
   # Construction timelines
   solar_construction_years: float = 2
   battery_lifespan_years: int = 13  # Battery replacement schedule
   
   # Natural gas maintenance schedules (hours/year)
   gas_maintenance_hours: Dict[str, int] = field(default_factory=lambda: {
       'aero': 130,
       'f_class': 190,
       'h_class': 200
   })
   
   # Turbine lead times (months, based on conservative industry estimates) probably more like 5-7 
   # Costa spoke to GE and they said 36 across the board
   turbine_lead_times: Dict[str, int] = field(default_factory=lambda: {
       'aero': 36,
       'f_class': 36,
       'h_class': 36
   })


@dataclass
class DegradationConfig:
   """Equipment degradation parameters"""
   # Solar PV degradation
   solar_first_year: float = 0.01  # 1% first year degradation
   solar_annual: float = 0.0055  # 0.55% annual degradation thereafter
   
   # Battery degradation models
  
   fade_model_path: str =  "output_tables/fade_surrogate.pkl"


   
   # Battery thermal model
   battery_rt_eff: float = 0.90  # Battery efficiency for thermal calculations
   battery_mth_per_mwh: float = 2.3  # Thermal mass per MWh (°C·h/kW)
   battery_t_min: float =   23# Minimum battery temperature in climate control (°C)
   battery_t_max: float = 27  # Maximum battery temperature in climate contorl (°C)
   
   # Gas turbine degradation (capacity%, efficiency% per year)
   gas_degradation_rates: Dict[str, tuple] = field(default_factory=lambda: {
       'aero': (0.002, 0.003),
       'f_class': (0.0015, 0.0025),
       'h_class': (0.0013, 0.0027)
   })


@dataclass
class FinancialConfig:
   """Financial analysis parameters"""
   discount_rate: float = 0.07  # 7% discount rate
   evaluation_years: int = 27  # Project lifetime for solar systems


@dataclass
class Config:
   """Complete configuration for datacenter power systems analysis"""
   costs: CostConfig = field(default_factory=CostConfig)
   efficiency: EfficiencyConfig = field(default_factory=EfficiencyConfig)
   it_load: ITLoadConfig = field(default_factory=ITLoadConfig)
   design: DesignConfig = field(default_factory=DesignConfig)
   degradation: DegradationConfig = field(default_factory=DegradationConfig)
   financial: FinancialConfig = field(default_factory=FinancialConfig)
   gas_turbine: GasTurbineConfig = field(default_factory=GasTurbineConfig)


def load_config(path: Optional[str] = None) -> Config:
   """Load configuration from JSON file or return defaults"""
   if path is None:
       return Config()
   
   with open(path, 'r') as f:
       data = json.load(f)
   
   return Config(
       costs=CostConfig(**data.get('costs', {})),
       efficiency=EfficiencyConfig(**data.get('efficiency', {})),
       it_load=ITLoadConfig(**data.get('it_load', {})),
       design=DesignConfig(**data.get('design', {})),
       degradation=DegradationConfig(**data.get('degradation', {})),
       financial=FinancialConfig(**data.get('financial', {}))
   )


def save_config(config: Config, path: str):
   """Save configuration to JSON file"""
   data = {
       'costs': config.costs.__dict__,
       'efficiency': config.efficiency.__dict__,
       'it_load': config.it_load.__dict__,
       'design': config.design.__dict__,
       'degradation': config.degradation.__dict__,
       'financial': config.financial.__dict__
   }
   
   with open(path, 'w') as f:
       json.dump(data, f, indent=2)
