"""
    Degradation Models for Datacenter Power Generation Systems
    -----------------------------------------------------------------
    This file provides a single source of truth for all component
    degradation.  The battery-fade surrogate is a grey-box model:
        deterministic physics scaffold  +  GP residuals.
    We keep the scaffold in the *Kelvin-coefficient* form used by
    BLAST-Lite for perfect back-compatibility.

    Notation
    --------
        Calendar:  k_cal ∝ exp( p₂ / T )
                   p₂  < 0  conventional Arrhenius (rate ↑ with T)

        Cycling :  k_cyc ∝ exp( p₇ / T )
                   p₇  > 0  inverse* Arrhenius (BLAST choice in their modelling)

    Conversion to physical activation energy
    ----------------------------------------
        E_cal = |p₂| · R          (≈ 43 kJ mol⁻¹)
        E_cyc =  p₇  · R          (≈ 18 kJ mol⁻¹)

"""

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from pvstoragesim import SimulationResult
from config import Config, load_config
import logging
import sys

# expose this module under a stable name so pickle can resolve classes
sys.modules.setdefault("degradation_model", sys.modules[__name__])

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
#  Kelvin-coefficient constants  (sourced from BLAST-Lite v1.1)
# ----------------------------------------------------------------------
T_MIN, T_MAX = 15, 30          # clip range for numerical stability  [°C]

P2_CAL_K = -5_210.0            # calendar Kelvin coefficient 
P7_CYC_K =  2_190.0            # cycle    Kelvin coefficient 

# Exponent baselines for ageing exponents β and α
BETA0  = 0.526                 # BLAST-Lite reference for calendar
ALPHA0 = 0.828                 # BLAST-Lite reference for cycling

# ----------------------------------------------------------------------
#  Scaffold helper functions
# ----------------------------------------------------------------------
def cal_scaffold(X, A, beta_shift, gamma):
    """
    Calendar-fade scaffold      [% capacity loss per year]

        Parameters in BLAST form:
            p₂  = P2_CAL_K  (fixed)
            β   = BETA0 + beta_shift

        A, beta_shift, gamma are fitted.
    """
    t, soc, T_C = X.T
    beta  = BETA0 + beta_shift
    T_K   = np.clip(T_C, T_MIN, T_MAX) + 273.15   # clamp for safety to boundaries of our training data
    arr   = np.exp(P2_CAL_K / T_K)                # note: P2_CAL_K < 0 → normal Arrhenius
    soc_term = 1 + gamma * np.abs(soc - 0.5)
    return A * (t ** beta) * arr * soc_term


def cyc_scaffold(X, B, alpha_shift):
    """
    Cycle-fade scaffold         [% capacity loss per year]

        p₇ = P7_CYC_K  (positive)  → inverse Arrhenius
        α  = ALPHA0 + alpha_shift
    """
    efc, soc, T_C = X.T
    alpha = ALPHA0 + alpha_shift
    efc   = np.maximum(efc, 1e-6)                 # guard against 0 ** α
    T_K   = np.clip(T_C, T_MIN, T_MAX) + 273.15
    arr   = np.exp(P7_CYC_K / T_K)                # inverse Arrhenius
    return B * (efc ** alpha) * arr

# ----------------------------------------------------------------------
#  Fade-surrogate class
# ----------------------------------------------------------------------
class FadeSurrogate:
    """
    Grey-box battery fade surrogate:
        scaffold (above)  +  Gaussian-process residuals.

    The class is serialised via pickle; keep it importable from this module!
    """

    def __init__(self, p_cal, p_cyc, gp_cal, gp_cyc):
        self.p_cal, self.p_cyc = p_cal, p_cyc
        self.gp_cal, self.gp_cyc = gp_cal, gp_cyc

    # ---- internal convenience methods -----------------------------------
    def delta_q_cal(self, X):
        return cal_scaffold(X, *self.p_cal) + self.gp_cal.predict(X)

    def delta_q_cyc(self, X):
        return cyc_scaffold(X, *self.p_cyc) + self.gp_cyc.predict(X)

    # ---- public API ------------------------------------------------------
    def __call__(self, Xcal, Xcyc):
        return self.delta_q_cal(Xcal), self.delta_q_cyc(Xcyc)

    def predict_marginal_fade(self, year, mean_soc, mean_temp, efc_cum):
        """
        Wrapper used by production code: returns (calendar, cycle) fade
        in % for the specified operating year and stressors.
        """
        X_cal = np.array([[year,       mean_soc, mean_temp]])
        X_cyc = np.array([[efc_cum,    mean_soc, mean_temp]])
        return self.__call__(X_cal, X_cyc)

# ----------------------------------------------------------------------
#  Cached loader for the pickled surrogate
# ----------------------------------------------------------------------
_FADE_MODEL = None

def load_fade_surrogate(cfg: Config):
    """
    Load & cache fade_surrogate.pkl specified in the configuration.
    The class must be importable under 'degradation_model.FadeSurrogate'
    (hence the `sys.modules.setdefault` trick above).
    """
    global _FADE_MODEL
    if _FADE_MODEL is None:
        with open(Path(cfg.degradation.fade_model_path), "rb") as f:
            _FADE_MODEL = pickle.load(f)
        logger.info("Grey-box battery fade surrogate loaded")
    return _FADE_MODEL

# ═══════════════════════════════════════════════════════════════════════════
# SOLAR DEGRADATION
# ═══════════════════════════════════════════════════════════════════════════

def get_solar_capacity_factor(operational_year: int, config: Optional[Config] = None) -> float:
   """
   Solar capacity factor at start of operational year.
   
   Args:
       operational_year: Years since start of operations (0 = first year)
       config: Configuration object
   
   Returns:
       Capacity multiplier (0-1)
   """
   cfg = config or load_config()
   
   if operational_year == 0:
       return 1.0
   elif operational_year == 1:
       return 1.0 - cfg.degradation.solar_first_year
   else:
       cap = 1.0 - cfg.degradation.solar_first_year
       cap *= (1.0 - cfg.degradation.solar_annual) ** (operational_year - 1)
       return max(0.0, cap)

# ═══════════════════════════════════════════════════════════════════════════
# BATTERY THERMAL MODEL
# ═══════════════════════════════════════════════════════════════════════════

def calculate_thermal_profile(
   sim_result: SimulationResult,
   facility_load,
   config: Optional[Config] = None,
   init_T: Optional[float] = None
) -> Dict:
   """
   Calculate battery thermal profile and annual thermal load.
   
   Args:
       sim_result: SimulationResult from pvstoragesim with hourly_data
       facility_load: FacilityLoad object with hourly_pue
       config: Configuration object
       init_T: Initial battery temperature (°C)
   
   Returns:
       Dict with battery_temperature array, annual_thermal_kwh, and mean_temperature
   """
   cfg = config or load_config()
   
   # Use config values or defaults
   init_T = init_T if init_T is not None else 20.0
   rt_eff = cfg.degradation.battery_rt_eff
   mth_per_MWh = cfg.degradation.battery_mth_per_mwh
   T_min = cfg.degradation.battery_t_min
   T_max = cfg.degradation.battery_t_max
   
   if sim_result.hourly_data is None:
       logger.warning("No hourly data available for thermal calculation")
       return {
           'battery_temperature': np.array([init_T]),
           'annual_thermal_kwh': 0.0,
           'mean_temperature': init_T,
           'temperature_p90': init_T,
           'temperature_p10': init_T
       }
   
   hd = sim_result.hourly_data
   battery_capacity_mwh = hd['battery_soc_mwh'].max()
   
   if battery_capacity_mwh == 0:
       logger.warning("No battery capacity detected")
       return {
           'battery_temperature': np.array([init_T]),
           'annual_thermal_kwh': 0.0,
           'mean_temperature': init_T,
           'temperature_p90': init_T,
           'temperature_p10': init_T
       }
   
   n_hours = len(hd)
   T = np.empty(n_hours)
   T[0] = init_T
   
   heat_kw = np.zeros(n_hours)
   cool_kw = np.zeros(n_hours)
   
   # Battery heat generation from inefficiency
   one_way_loss = 1 - np.sqrt(rt_eff)  # One-way trip efficiency loss
   thermal_mass = mth_per_MWh * battery_capacity_mwh
   
   # Get hourly PUE for cooling efficiency
   pue_hourly = np.asarray(facility_load.hourly_pue)
   
   for h in range(1, n_hours):
       # Heat generation from battery charge/discharge
       charge_mw = abs(hd['battery_charge_mw'].iloc[h])
       discharge_mw = abs(hd['battery_discharge_mw'].iloc[h])
       heat_generation_kw = (charge_mw + discharge_mw) * one_way_loss * 1000
       
       # Predicted temperature without thermal management
       temp_predicted = T[h-1] + heat_generation_kw / thermal_mass
       
       # Thermal management (heating/cooling)
       heating_kw = cooling_kw = 0.0
       
       if temp_predicted > T_max:
           # Cooling required
           excess_heat_kw = (temp_predicted - T_max) * thermal_mass
           cooling_kw = excess_heat_kw / (1 / (pue_hourly[h] - 1 + 1e-6))
           cool_kw[h] = cooling_kw
           T[h] = T_max  # Maintain at max temperature
           
       elif temp_predicted < T_min and heat_generation_kw < 1.0:
           # Heating required (only when minimal battery activity)
           heat_deficit_kw = (T_min - temp_predicted) * thermal_mass
           heating_kw = heat_deficit_kw
           heat_kw[h] = heating_kw
           T[h] = T_min  # Maintain at min temperature
       else:
           # No thermal management needed
           T[h] = temp_predicted
   
   # Total annual thermal load
   annual_thermal_kwh = heat_kw.sum() + cool_kw.sum()
   
   # Calculate temperature statistics
   mean_temp = float(np.mean(T))
   temp_p90 = float(np.percentile(T, 90))
   temp_p10 = float(np.percentile(T, 10))
   
   logger.debug(f"Battery thermal profile: mean={mean_temp:.1f}°C, "
               f"P10={temp_p10:.1f}°C, P90={temp_p90:.1f}°C, "
               f"thermal load={annual_thermal_kwh:.0f} kWh/year")
   
   return {
       'battery_temperature': T,
       'annual_thermal_kwh': annual_thermal_kwh,
       'mean_temperature': mean_temp,
       'temperature_p90': temp_p90,
       'temperature_p10': temp_p10
   }

# ═══════════════════════════════════════════════════════════════════════════
# SOPHISTICATED BATTERY DEGRADATION MODEL (GP-BASED)
# ═══════════════════════════════════════════════════════════════════════════

def extract_battery_year_0_stats(
    sim_result: SimulationResult,
    facility_load,
    config: Optional[Config] = None
) -> Dict:
    """
    Extract Year 0 battery statistics for degradation model.
    Now includes thermal statistics from the profile calculation.
    """
    cfg = config or load_config()
    
    if sim_result.hourly_data is None:
        logger.warning("No hourly data available, using fallback estimates")
        return {
            'mean_soc0_pct': 65.0,
            'efc0': sim_result.battery_cycles_per_year if sim_result else 250.0,
            'soc_p90': 0.75,
            'mean_temperature': 25.0,
            'temperature_p90': 30.0,
            'thermal_parasitic_fraction': 0.01
        }
    
    hd = sim_result.hourly_data
    battery_capacity_mwh = hd['battery_soc_mwh'].max()
    
    if battery_capacity_mwh == 0:
        logger.warning("No battery capacity detected, using fallback estimates")
        return {
            'mean_soc0_pct': 65.0,
            'efc0': 0.0,
            'soc_p90': 0.75,
            'mean_temperature': 25.0,
            'temperature_p90': 30.0,
            'thermal_parasitic_fraction': 0.0
        }
    
    # Extract features matching training approach
    sf = sim_result.stress_features or {}
    
    # Mean SOC (from stress features, matching training)
    mean_soc = sf.get("mean_soc", 0.5)
    mean_soc0_pct = mean_soc * 100
    
    # SOC P90 (directly from hourly data)
    soc_array = hd['battery_soc_mwh'] / battery_capacity_mwh
    soc_p90 = float(np.percentile(soc_array, 90))
    
    # EFC from throughput
    throughput_mwh = sf.get("throughput_mwh", 0.0)
    efc0 = throughput_mwh / battery_capacity_mwh if battery_capacity_mwh > 0 else 0.0
    
    # Calculate thermal profile and extract statistics
    thermal_data = calculate_thermal_profile(sim_result, facility_load, cfg)
    mean_temperature = thermal_data['mean_temperature']
    temperature_p90 = thermal_data['temperature_p90']
    
    # Thermal parasitic load
    thermal_parasitic_kwh = thermal_data['annual_thermal_kwh']
    delivered_kwh = sim_result.load_served_mwh * 1000
    thermal_parasitic_fraction = (
        thermal_parasitic_kwh / delivered_kwh if delivered_kwh > 0 else 0.0
    )
    
    logger.debug(f"Battery Year 0 stats: mean_SOC={mean_soc:.2f}, SOC_P90={soc_p90:.2f}, "
                f"EFC={efc0:.1f}, mean_temp={mean_temperature:.1f}°C, "
                f"Thermal parasitic={thermal_parasitic_fraction*100:.2f}%")
    
    return {
        'mean_soc0_pct': float(mean_soc0_pct),
        'mean_soc': float(mean_soc),  # Store normalized version too
        'efc0': float(efc0),
        'soc_p90': float(soc_p90),
        'mean_temperature': float(mean_temperature),
        'temperature_p90': float(temperature_p90),
        'thermal_parasitic_kwh': float(thermal_parasitic_kwh),
        'thermal_parasitic_fraction': float(thermal_parasitic_fraction)
    }

def predict_capacity_trajectory(
        mean_soc0_pct: float,
        efc0: float,
        mean_temperature: float = 25.0,
        n_years: int = 25,
        config: Optional[Config] = None
    ) -> Dict:
        """
        Predict battery capacity trajectory using the grey-box fade surrogate.
        """

        cfg = config or load_config()
        fade = load_fade_surrogate(cfg)

        mean_soc = mean_soc0_pct / 100.0
        mean_temp_clamped = np.clip(mean_temperature, 10.0, 40.0)

        trajectory = [100.0]          # capacity in %
        annual_fades = []
        capacity = 1.0

        for year in range(1, n_years + 1):
            efc_cum = (year - 1) * efc0
            dq_cal, dq_cyc = fade.predict_marginal_fade(
                year, mean_soc, mean_temp_clamped, efc_cum
            )
            fade_pct = dq_cal.item() + dq_cyc.item()
            annual_fades.append(fade_pct)
            capacity *= (1 - fade_pct / 100.0)
            trajectory.append(capacity * 100.0)

        return {
            'trajectory': trajectory,
            'annual_fades': annual_fades,
            'final_capacity': trajectory[-1],
            'temperature_used': mean_temp_clamped
        }



def get_battery_capacity_at_year(
    operational_year: int,
    year_0_stats: Dict,
    config: Optional[Config] = None
) -> float:
    """
    Capacity after `operational_year` years, using the grey-box surrogate.
    """

    if operational_year == 0:
        return 1.0

    cfg   = config or load_config()
    fade  = load_fade_surrogate(cfg)

    mean_soc        = year_0_stats.get('mean_soc',
                                       year_0_stats.get('mean_soc0_pct', 65.0) / 100.0)
    
    efc0            = year_0_stats.get('efc0', 250.0)
    mean_temperature= year_0_stats.get('mean_temperature', 25.0)
    mean_temp_clamped = np.clip(mean_temperature, 10.0, 40.0)

    capacity = 1.0
    for yr in range(1, operational_year + 1):
        efc_cum = (yr - 1) * efc0
        dq_cal, dq_cyc = fade.predict_marginal_fade(
            yr, mean_soc, mean_temp_clamped, efc_cum
        )
        # -------- extract scalars --------------------------------------
        fade_pct = dq_cal.item() + dq_cyc.item()
        capacity *= (1 - fade_pct / 100.0)

    return capacity

def get_battery_capacity_with_replacement(
    operational_year: int,
    year_0_stats: Dict,
    replacement_year: int = 13,
    config: Optional[Config] = None
) -> float:
    """
    Get battery capacity accounting for mid-life replacement.
    
    Args:
        operational_year: Years since start of operations
        year_0_stats: Dictionary from extract_battery_year_0_stats
        replacement_year: Year when battery is replaced (default 13)
        config: Configuration object
    
    Returns:
        Battery capacity factor (0-1)
    """
    if operational_year < replacement_year:
        # Before replacement: use original battery degradation
        return get_battery_capacity_at_year(operational_year, year_0_stats, config)
    else:
        # After replacement: fresh battery with adjusted cycling
        # Assume cycling reduces proportionally with solar degradation
        solar_factor = get_solar_capacity_factor(replacement_year + 1, config)
        
        # Create adjusted stats for replacement battery
        replacement_stats = {
            'mean_soc': year_0_stats.get('mean_soc', 0.65),
            'soc_p90': year_0_stats.get('soc_p90', 0.75),
            'efc0': year_0_stats.get('efc0', 250.0) * solar_factor,
            'mean_temperature': year_0_stats.get('mean_temperature', 25.0)
        }
        
        # Years since replacement
        years_since_replacement = operational_year - replacement_year
        return get_battery_capacity_at_year(years_since_replacement, replacement_stats, config)

# ═══════════════════════════════════════════════════════════════════════════
# GAS TURBINE DEGRADATION
# ═══════════════════════════════════════════════════════════════════════════

def get_gas_degradation_factors(
   operational_year: int,
   turbine_class: str,
   config: Optional[Config] = None
) -> Tuple[float, float]:
   """
   Gas turbine capacity and efficiency factors.
   
   Args:
       operational_year: Years since start of operations
       turbine_class: 'aero', 'f_class', or 'h_class'
       config: Configuration object
   
   Returns:
       Tuple of (capacity_factor, efficiency_factor)
   """
   if operational_year == 0:
       return 1.0, 1.0
   
   cfg = config or load_config()
   
   # Get degradation rates from config
   rates = cfg.degradation.gas_degradation_rates
   cap_rate, eff_rate = rates.get(turbine_class, rates['f_class'])
   
   capacity_factor = (1.0 - cap_rate) ** operational_year
   efficiency_factor = (1.0 - eff_rate) ** operational_year
   
   return capacity_factor, efficiency_factor

def get_temperature_derating(
   turbine_class: str,
   ambient_temp_c: float,
   config: Optional[Config] = None
) -> float:
   """
   Temperature derating factor for gas turbine capacity.
   
   Args:
       turbine_class: 'aero', 'f_class', or 'h_class'
       ambient_temp_c: Ambient temperature in Celsius
       config: Configuration object
   
   Returns:
       Capacity derating factor
   """
   cfg = config or load_config()
   
   baseline_temp = 15.0  # ISO conditions
   temp_delta = ambient_temp_c - baseline_temp
   
   # Get derating rate from config
   if turbine_class == 'aero':
       rate = cfg.gas_turbine.temp_derating_per_c_aero
   elif turbine_class == 'f_class':
       rate = cfg.gas_turbine.temp_derating_per_c_f_class
   elif turbine_class == 'h_class':
       rate = cfg.gas_turbine.temp_derating_per_c_h_class
   else:
       raise ValueError(f"Unknown turbine class: {turbine_class}")
   
   # Only apply derating for temperatures above baseline
   if temp_delta <= 0:
       return 1.0
   
   derating_factor = 1.0 - (rate * temp_delta)
   return max(0.7, derating_factor)  # Floor at 70% capacity

# ═══════════════════════════════════════════════════════════════════════════
# ENERGY INTERPOLATION WITH THERMAL EFFECTS
# ═══════════════════════════════════════════════════════════════════════════

def interpolate_annual_energy(
   sim_year_0: SimulationResult,
   sim_year_13: SimulationResult, 
   sim_year_14: SimulationResult,
   sim_year_25: SimulationResult,
   construction_years: float,
   year_0_stats: Dict
) -> np.ndarray:
   """
   Interpolate annual energy production over 27-year project lifetime using anchor point simulations.
   
   Uses four anchor points:
   1. First operational year (fresh system)
   2. Year before battery replacement (degraded system) - operational year 13
   3. First year with new battery (solar degraded, battery fresh) - operational year 14
   4. Final year (both degraded)
   
   Args:
       sim_year_0: Simulation with fresh components
       sim_year_13: Simulation with degraded solar & battery (pre-replacement)
       sim_year_14: Simulation with degraded solar & fresh battery (post-replacement) 
       sim_year_25: Simulation with degraded solar & aged replacement battery
       construction_years: Duration of construction phase
       year_0_stats: Battery statistics including thermal_parasitic_fraction
       
   Returns:
       Array of annual energy production (MWh) for calendar years 0-26
   """
   energy = np.zeros(27)
   
   # Apply thermal parasitic reduction to all anchor points
   thermal_parasitic = year_0_stats.get('thermal_parasitic_kwh', 0.0)/1000  # Convert to MWh
   
   anchor_energies = {
       0: sim_year_0.load_served_mwh - thermal_parasitic,
       12: sim_year_13.load_served_mwh  - thermal_parasitic, 
       13: sim_year_14.load_served_mwh - thermal_parasitic,
       24: sim_year_25.load_served_mwh  - thermal_parasitic
   }
   
   # Fill construction years with zero
   ops_start_year = int(np.ceil(construction_years))
   for year in range(ops_start_year):
       energy[year] = 0.0
   
   # Map operational years to calendar years and set anchor points
   calendar_anchors = {}
   for ops_year, energy_val in anchor_energies.items():
       calendar_year = ops_start_year + ops_year
       if calendar_year < 27:
           calendar_anchors[calendar_year] = energy_val
           energy[calendar_year] = energy_val
   
   # Interpolate between anchors
   anchor_years = sorted(calendar_anchors.keys())
   
   for i in range(len(anchor_years) - 1):
       start_year = anchor_years[i]
       end_year = anchor_years[i + 1]
       start_energy = calendar_anchors[start_year]
       end_energy = calendar_anchors[end_year]
       
       # Linear interpolation between anchor points
       for year in range(start_year + 1, min(end_year, 27)):
           fraction = (year - start_year) / (end_year - start_year)
           energy[year] = start_energy * (1 - fraction) + end_energy * fraction
   
   # Handle any remaining years after last anchor
   if anchor_years and anchor_years[-1] < 26:
       final_energy = calendar_anchors[anchor_years[-1]]
       for year in range(anchor_years[-1] + 1, 27):
           energy[year] = final_energy  # Hold constant after last anchor
   
   return energy

# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION AND TESTING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

