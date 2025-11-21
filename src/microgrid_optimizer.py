"""
Microgrid System Optimizer - Refactored to use config.py
Finds optimal solar + battery configurations using CapEx optimization (proven LCOE proxy).
Handles degradation analysis and provides clean interface to other modules.
"""

from re import S
from tkinter import ARC
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
from scipy.optimize import differential_evolution
import warnings
from config import Config, load_config

# Required imports for simulation and degradation
from power_systems_estimator import PowerFlowAnalyzer
from pvstoragesim import evaluate_system, get_solar_generation
from degradation_model import get_solar_capacity_factor, get_battery_capacity_at_year, extract_battery_year_0_stats
from it_facil import FacilityLoad

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES (Required by other modules)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SystemCosts:
   """Cost parameters for system components."""
   solar_cost_per_kw: float  # $/kW DC for modules
   battery_cost_per_kw: float  # $/kWh for ESS
   solar_bos_cost_per_kw: float  # $/kW for solar BOS
   battery_bos_cost_per_kw: float  # $/kW for battery BOS
   battery_hours: float = 4.0  # Duration of battery storage
   
   def calculate_system_cost(self, solar_mw: float, battery_mw: float) -> float:
       """Calculate total system cost in millions of dollars."""
       solar_cost = solar_mw * 1000 * self.solar_cost_per_kw / 1e6
       battery_cost = battery_mw * 1000 * self.battery_cost_per_kw / 1e6
       solar_bos_cost = solar_mw * 1000 * self.solar_bos_cost_per_kw / 1e6
       battery_bos_cost = battery_mw * 1000 * self.battery_bos_cost_per_kw / 1e6
       return solar_cost + battery_cost + solar_bos_cost + battery_bos_cost

@dataclass
class OptimizationResult:
   """Results from optimization with degradation analysis."""
   solar_mw: float
   battery_mw: float
   battery_mwh: float
   total_cost_million: float
   land_area_acres: float
   optimization_type: str
   meet_requirement: bool
   
   # Degradation analysis results (required by lcoe_calc.py)
   sim_year_0: 'SimulationResult'
   sim_year_13: 'SimulationResult'  
   sim_year_14: 'SimulationResult'
   sim_year_25: 'SimulationResult'
   year_0_stats: Dict
   
   # Optimization metadata
   optimization_success: bool = True
   optimization_message: str = ""
   function_evaluations: int = 0
   
   # Convenience properties
   @property
   def uptime_pct(self) -> float:
       return self.sim_year_0.uptime_pct if self.sim_year_0 else 99.5
   
   @property 
   def energy_served_pct(self) -> float:
       return self.sim_year_0.energy_served_pct if self.sim_year_0 else 100.0

# ═══════════════════════════════════════════════════════════════════════════
# CORE OPTIMIZER CLASS
# ═══════════════════════════════════════════════════════════════════════════

class MicrogridOptimizer:
   """
   Solar+battery system optimizer using CapEx minimization (proven LCOE proxy).
   Handles 4-year degradation analysis and provides clean interface.
   """
   
   def __init__(
       self,
       latitude: float,
       longitude: float,
       facility_load: FacilityLoad,
       required_uptime_pct: float,
       costs: SystemCosts,
       architecture: str = "ac_coupled",
       efficiency_params: Optional[Config] = None,
       verbose: bool = False
   ):
       """
       Initialize optimizer.
       
       Args:
           latitude, longitude: Location coordinates
           facility_load: FacilityLoad object with datacenter requirements
           required_uptime_pct: Minimum uptime requirement (%)
           costs: SystemCosts object with component pricing
           architecture: "ac_coupled" or "dc_coupled"
           efficiency_params: Config object with efficiency parameters
          
       """
       self.latitude = latitude
       self.longitude = longitude
       self.facility_load = facility_load
       self.required_uptime_pct = required_uptime_pct
       self.costs = costs
       self.architecture = architecture
       self.config = efficiency_params or load_config()
       
       
       # Derived parameters
       self.facility_load_mw = facility_load.facility_load_design_mw
       self.hourly_pue = getattr(facility_load, "hourly_pue", None)
       
       # Optimization tracking
       self._simulation_cache: Dict[str, Dict] = {}
       self.function_evaluations = 0
       
       # Get solar profile once and cache (with facility_load for TMY caching)
       logger.info(f"Fetching solar data for ({latitude:.3f}, {longitude:.3f})")
       self.solar_profile = get_solar_generation(
           latitude, longitude, facility_load=facility_load, architecture=architecture
       )
       
       # Calculate optimization bounds
       self.bounds = [
           (1.0 * self.facility_load_mw, 15.0 * self.facility_load_mw),  # Solar MW
           (1.0 * self.facility_load_mw, 10.0 * self.facility_load_mw)   # Battery MW
       ]
       
       logger.info(f"Initialized {architecture} optimizer for {self.facility_load_mw:.1f} MW load")
   
   def _get_cache_key(self, solar_mw: float, battery_mw: float) -> str:
       """Generate cache key with appropriate tolerance."""
       tolerance = 1 # 1 MW tolerance for rounding
       solar_rounded = round(solar_mw / tolerance) * tolerance
       battery_rounded = round(battery_mw / tolerance) * tolerance
       return f"{solar_rounded:.1f}_{battery_rounded:.1f}"
   
   def _run_degradation_analysis(self, solar_mw: float, battery_mw: float, fast_mode: bool = False):
       """
       Run degradation analysis with optional fast mode.
       fast_mode=True: Year 0 only, no hourly data (for Stage 1 screening)
       fast_mode=False: Full 4-year analysis (for Stage 2 optimization)
       """
       try:
           if fast_mode:
               # STAGE 1: Fast screening (Year 0 only, no hourly data)
               sim_year_0 = evaluate_system(
                   self.latitude, self.longitude, solar_mw, battery_mw,
                   self.facility_load, hourly_pue=self.hourly_pue,
                   architecture=self.architecture, efficiency_params=self.config,
                   solar_profile=self.solar_profile, return_hourly=False
               )
               if sim_year_0.uptime_pct < self.required_uptime_pct:
                   return None
           
               return {'sim_year_0': sim_year_0}
       
           else:
               # STAGE 2: Full degradation analysis (all 4 years)
               # Year 0: Fresh system with hourly data
               sim_year_0 = evaluate_system(
                   self.latitude, self.longitude, solar_mw, battery_mw,
                   self.facility_load, hourly_pue=self.hourly_pue,
                   architecture=self.architecture, efficiency_params=self.config,
                   solar_profile=self.solar_profile, return_hourly=True
               )
           
               if sim_year_0.uptime_pct < self.required_uptime_pct:
                   return None
           
               # Extract battery statistics
               year_0_stats = extract_battery_year_0_stats(sim_year_0, self.facility_load)
           
               # Year 13: Pre-replacement degradation
               solar_factor_13 = get_solar_capacity_factor(12)
               battery_factor_13 = get_battery_capacity_at_year(12, year_0_stats)
           
               sim_year_13 = evaluate_system(
                   self.latitude, self.longitude, 
                   solar_mw * solar_factor_13, battery_mw * battery_factor_13,
                   self.facility_load, hourly_pue=self.hourly_pue,
                   architecture=self.architecture, efficiency_params=self.config,
                   solar_profile=self.solar_profile, return_hourly=True
               )
           
               if sim_year_13.uptime_pct < self.required_uptime_pct:
                   return None
           
               # Year 14: Post-replacement
               solar_factor_14 = get_solar_capacity_factor(13)
           
               sim_year_14 = evaluate_system(
                   self.latitude, self.longitude,
                   solar_mw * solar_factor_14, battery_mw,
                   self.facility_load, hourly_pue=self.hourly_pue,
                   architecture=self.architecture, efficiency_params=self.config,
                   solar_profile=self.solar_profile, return_hourly=True
               )
           
               if sim_year_14.uptime_pct < self.required_uptime_pct:
                   return None
           
               # Year 25: End of life
               solar_factor_25 = get_solar_capacity_factor(24)
               replacement_stats = {
                    'mean_soc0_pct': year_0_stats['mean_soc0_pct'], 
                    'efc0': year_0_stats['efc0'],  # <- Keep original EFC0
                    'mean_soc': year_0_stats.get('mean_soc', year_0_stats['mean_soc0_pct'] / 100),
                    'soc_p90': year_0_stats.get('soc_p90', 0.75)  # Add missing features
                }

               battery_factor_25 = get_battery_capacity_at_year(12, replacement_stats)
           
               sim_year_25 = evaluate_system(
                   self.latitude, self.longitude,
                   solar_mw * solar_factor_25, battery_mw * battery_factor_25,
                   self.facility_load, hourly_pue=self.hourly_pue,
                   architecture=self.architecture, efficiency_params=self.config,
                   solar_profile=self.solar_profile, return_hourly=True
               )
           
               if sim_year_25.uptime_pct < self.required_uptime_pct:
                   return None
           
               return {
                   'sim_year_0': sim_year_0,
                   'sim_year_13': sim_year_13,
                   'sim_year_14': sim_year_14,
                   'sim_year_25': sim_year_25,
                   'year_0_stats': year_0_stats
               }
           
       except Exception as e:
           if self.verbose:
               logger.debug(f"Analysis failed for {solar_mw:.1f}MW solar, {battery_mw:.1f}MW battery: {e}")
           return None
   

   def _evaluate_configuration(self, solar_mw: float, battery_mw: float, fast_mode: bool = False) -> Optional[Dict]:
       """Cached configuration evaluation with optional fast mode."""
       # Use different cache for fast vs full mode
       cache_suffix = "_fast" if fast_mode else "_full"
       cache_key = self._get_cache_key(solar_mw, battery_mw) + cache_suffix
   
       if cache_key in self._simulation_cache:
           return self._simulation_cache[cache_key]
   
       # Count function evaluation
       self.function_evaluations += 1
   
       # Run degradation analysis
       result = self._run_degradation_analysis(solar_mw, battery_mw, fast_mode=fast_mode)
   
       # Cache result (even if None)
       self._simulation_cache[cache_key] = result
   
       return result

   
   

   
   def calculate_land_area(self, solar_mw: float, battery_mw: float) -> float:
       """Calculate total land area in acres."""
       solar_acres = solar_mw * self.config.design.solar_acres_per_mw
       battery_mwh = battery_mw * self.costs.battery_hours
       battery_acres = battery_mwh * self.config.design.battery_acres_per_mwh
       return solar_acres + battery_acres
   
   def optimize(self, *, method: str = "global") -> OptimizationResult:
    """
    Two-stage optimization with progressive bounds expansion for robustness.
    """
    logger.info(f"Starting 2-stage optimization for {self.architecture}")
    self.function_evaluations = 0

    # ──────────────────────────────────────────────────────────────
    #  1. Stage-1 — Feasibility screening
    # ──────────────────────────────────────────────────────────────
    solar_min, solar_max = self.bounds[0]
    batt_min, batt_max = self.bounds[1]

    # Adaptive sample count
    box_area = (solar_max / solar_min) * (batt_max / batt_min)
    n_screen = max(80, int(40 + 20 * np.log10(box_area)))

    def latin_samples(n: int) -> list[tuple[float, float]]:
        rng = np.random.default_rng()
        u1 = rng.random(n)
        u2 = rng.random(n)
        rng.shuffle(u1)
        rng.shuffle(u2)

        log_smin, log_smax = np.log10([solar_min, solar_max])
        log_bmin, log_bmax = np.log10([batt_min, batt_max])

        return [
            (10 ** (log_smin + u1[i]*(log_smax-log_smin)),
             10 ** (log_bmin + u2[i]*(log_bmax-log_bmin)))
            for i in range(n)
        ]

    feasible: list[dict] = []
    attempt = 0
    while not feasible:
        attempt += 1
        logger.info(f"Stage-1 screening attempt {attempt}: {n_screen} samples")
        for solar_mw, battery_mw in latin_samples(n_screen):
            res = self._evaluate_configuration(solar_mw, battery_mw, fast_mode=True)
            if res is None:
                continue

            cost = self.costs.calculate_system_cost(solar_mw, battery_mw)
            feasible.append(dict(
                solar_mw=solar_mw,
                battery_mw=battery_mw,
                cost=cost,
                result=res
            ))

        if feasible:
            break

        logger.warning("No feasible designs – expanding bounds by ×1.5")
        solar_max *= 1.5
        batt_max *= 1.5
        self.bounds = [(solar_min, solar_max), (batt_min, batt_max)]
        n_screen = int(n_screen * 1.4)

    feasible.sort(key=lambda d: d["cost"])
    logger.info(f"Stage-1: {len(feasible)}/{n_screen} feasible "
                f"(best ${feasible[0]['cost']:.1f} M)")

    # ──────────────────────────────────────────────────────────────
    #  2. Stage-2 — Progressive bounds expansion with tracking
    # ──────────────────────────────────────────────────────────────
    stage2_attempts = 0
    max_attempts = 3
    expansion_factor = 1.15  # Start with 15% buffer
    
    s_opt = None
    b_opt = None
    final_res = None
    result_de = None
    
    while stage2_attempts < max_attempts:
        stage2_attempts += 1
        
        # Calculate bounds with progressive expansion
        buf = expansion_factor ** (stage2_attempts - 1)
        stage2_bounds = [
            (max(solar_min, min(d['solar_mw'] for d in feasible) / buf),
             min(solar_max, max(d['solar_mw'] for d in feasible) * buf)),
            (max(batt_min, min(d['battery_mw'] for d in feasible) / buf),
             min(batt_max, max(d['battery_mw'] for d in feasible) * buf)),
        ]
        
        logger.info(f"Stage-2 attempt {stage2_attempts}/{max_attempts} - "
                   f"Solar [{stage2_bounds[0][0]:.1f}–{stage2_bounds[0][1]:.1f}] MW, "
                   f"Battery [{stage2_bounds[1][0]:.1f}–{stage2_bounds[1][1]:.1f}] MW")
        
        # Track valid configurations found during DE
        valid_configs = []
        
        def stage2_obj_with_tracking(x: np.ndarray) -> float:
            s_mw, b_mw = x
            res = self._evaluate_configuration(s_mw, b_mw, fast_mode=False)
            if res is None:
                return 1e9
            
            # Track this valid configuration
            cost = self.costs.calculate_system_cost(s_mw, b_mw)
            valid_configs.append((s_mw, b_mw, cost, res))
            return cost
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result_de = differential_evolution(
                    stage2_obj_with_tracking,
                    stage2_bounds,
                    maxiter=25,
                    popsize=10,
                    disp=False
                )
            
            # Check if DE result is valid
            s_opt, b_opt = result_de.x
            final_res = self._evaluate_configuration(s_opt, b_opt, fast_mode=False)
            
            if final_res is not None:
                logger.info(f"Stage-2 succeeded on attempt {stage2_attempts}")
                break
            elif valid_configs:
                # DE converged to invalid, but we found valid configs
                logger.warning("DE converged to invalid, using best valid config found")
                valid_configs.sort(key=lambda x: x[2])  # Sort by cost
                s_opt, b_opt, _, final_res = valid_configs[0]
                result_de.fun = valid_configs[0][2]
                break
                
        except Exception as e:
            logger.warning(f"Stage-2 attempt {stage2_attempts} failed: {e}")
            
            # Check if we found any valid configs before failure
            if valid_configs:
                logger.info("Using best valid config found before failure")
                valid_configs.sort(key=lambda x: x[2])
                s_opt, b_opt, _, final_res = valid_configs[0]
                result_de = type('dummy', (), {
                    'fun': valid_configs[0][2], 
                    'x': np.array([s_opt, b_opt]),
                    'message': f"Recovered from: {e}"
                })
                break
    
    # ──────────────────────────────────────────────────────────────
    #  3. Final fallback if all Stage-2 attempts failed
    # ──────────────────────────────────────────────────────────────
    if final_res is None:
        logger.warning("All Stage-2 attempts failed, trying best Stage-1 with full degradation")
        
        # Try best Stage-1 candidate with full degradation
        bf = feasible[0]
        final_res = self._evaluate_configuration(
            bf['solar_mw'], bf['battery_mw'], fast_mode=False
        )
        
        if final_res is not None:
            s_opt, b_opt = bf['solar_mw'], bf['battery_mw']
            result_de = type('dummy', (), {
                'fun': bf['cost'],
                'x': np.array([s_opt, b_opt]),
                'message': "Used Stage-1 best"
            })
        else:
            # Last resort: oversized system
            logger.error("Even Stage-1 best fails, using oversized system")
            s_opt = self.facility_load_mw * 10
            b_opt = self.facility_load_mw * 8
            final_res = self._evaluate_configuration(s_opt, b_opt, fast_mode=False)
            
            if final_res is None:
                raise ValueError("Cannot find feasible configuration even with 10x oversizing")
                
            result_de = type('dummy', (), {
                'fun': self.costs.calculate_system_cost(s_opt, b_opt),
                'x': np.array([s_opt, b_opt]),
                'message': "Emergency oversizing"
            })

    # ──────────────────────────────────────────────────────────────
    #  4. Assemble return object
    # ──────────────────────────────────────────────────────────────
    opt = OptimizationResult(
        solar_mw=s_opt,
        battery_mw=b_opt,
        battery_mwh=b_opt * self.costs.battery_hours,
        total_cost_million=result_de.fun,
        land_area_acres=self.calculate_land_area(s_opt, b_opt),
        optimization_type="Screening + DE (progressive expansion)",
        meet_requirement=True,
        optimization_success=True,
        optimization_message=(
            f"{len(feasible)} feasible from {n_screen} samples → "
            f"Stage-2 attempts: {stage2_attempts} ({getattr(result_de, 'message', 'converged')})"
        ),
        function_evaluations=self.function_evaluations,
        sim_year_0=final_res['sim_year_0'],
        sim_year_13=final_res['sim_year_13'],
        sim_year_14=final_res['sim_year_14'],
        sim_year_25=final_res['sim_year_25'],
        year_0_stats=final_res['year_0_stats']
    )

    logger.info(f"Optimized: {opt.solar_mw:.1f} MW PV, "
               f"{opt.battery_mw:.1f} MW / {opt.battery_mwh:.0f} MWh "
               f"(${opt.total_cost_million:.1f} M) "
               f"after {self.function_evaluations} evaluations")

    return opt


# ═══════════════════════════════════════════════════════════════════════════
# NATURAL GAS OPTIMIZATION (Separate class for LCOE optimization)
# ═══════════════════════════════════════════════════════════════════════════

class PowerSystemOptimizer:
   """
   Orchestrates optimization across solar+storage (CapEx) and natural gas (LCOE).
   Required by lcoe_calc.py and other modules.
   """
   
   def __init__(
       self, 
       latitude: float, 
       longitude: float, 
       facility_load: FacilityLoad,
       required_uptime_pct: float, 
       costs: SystemCosts,
       efficiency_params: Optional[Config] = None
   ):
       self.latitude = latitude
       self.longitude = longitude  
       self.facility_load = facility_load
       self.required_uptime_pct = required_uptime_pct
       self.costs = costs
       self.config = efficiency_params or load_config()
   
   def optimize_solar_storage(self, architecture: str = "ac_coupled") -> OptimizationResult:
       """Optimize solar+storage using CapEx minimization."""
       optimizer = MicrogridOptimizer(
           latitude=self.latitude,
           longitude=self.longitude,
           facility_load=self.facility_load,
           required_uptime_pct=self.required_uptime_pct,
           costs=self.costs,
           architecture=architecture,
           efficiency_params=self.config
       )
       return optimizer.optimize()
   
   def optimize_natural_gas(self, gas_price: float = 3.5) -> Tuple[OptimizationResult, 'PlantConfiguration']:
       """Optimize natural gas system using LCOE minimization."""
       from natgas_system_tool import NGPowerPlantCalculator, generate_plant_configurations, TURBINE_LIBRARY
       from lcoe_calc import calculate_gas_system_lcoe
       
       ng_calculator = NGPowerPlantCalculator(
           facility_load=self.facility_load,
           include_backup= True,
           efficiency_params=self.config
       )

       configs = generate_plant_configurations(
           ng_calculator.required_generation_mw,
           TURBINE_LIBRARY,
           self.facility_load.design_ambient_temp_c,
           require_n_minus_1=False, 
           config=self.config,
           annual_energy_mwh=ng_calculator.annual_energy_mwh  #
       )

       if not configs:
           raise ValueError("No feasible natural gas configurations found")

       # Find configuration with minimum LCOE
       best_lcoe = float('inf')
       best_config = None

       for config in configs:
           try:
               lcoe = calculate_gas_system_lcoe(
                  plant_config= config,
                  gas_price= gas_price,
                  facility_load = self.facility_load,
                  config =  self.config
               )
               
               if lcoe.lcoe < best_lcoe:
                   best_lcoe = lcoe.lcoe
                   best_config = config
                   
           except Exception as e:
               logger.warning(f"Failed to calculate LCOE for {config.turbine_model}: {e}")
               continue

       if best_config is None:
           raise ValueError("Could not find optimal natural gas configuration")

       # Convert to OptimizationResult format
       optimization_result = OptimizationResult(
           solar_mw=0,
           battery_mw=0,
           battery_mwh=0,
           total_cost_million=best_config.capex_per_kw * best_config.total_capacity_mw / 1000,
           land_area_acres=10,  # Simplified assumption
           optimization_type="Natural Gas LCOE",
           meet_requirement=True,
           # Dummy values for required fields
           sim_year_0=None,
           sim_year_13=None,
           sim_year_14=None,
           sim_year_25=None,
           year_0_stats={}
       )
   
       return optimization_result, best_config

