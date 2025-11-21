"""
IT Facility Load Calculator - Refactored to use config.py
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional
from config import Config, load_config

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CSV LOADING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def load_hourly_load_data(csv_path: str = "output_tables/hourly_load_data.csv") -> np.ndarray:
   """
   Load hourly IT load shape from CSV file.
   
   Args:
       csv_path: Path to the hourly load data CSV file
       
   Returns:
       8760-element numpy array of normalized load multipliers
   """
   try:
       df = pd.read_csv(csv_path)
       
       # Validate structure
       required_cols = ['date', 'hour', 'it_load_avg', 'it_load_norm']
       if not all(col in df.columns for col in required_cols):
           raise ValueError(f"CSV missing required columns. Found: {df.columns.tolist()}")
       
       if len(df) != 8760:
           raise ValueError(f"CSV must have exactly 8760 rows, found {len(df)}")
       
       # Extract normalized load values
       load_shape = df['it_load_norm'].values
       
       # Validate data
       if np.any(load_shape <= 0):
           raise ValueError("All load values must be positive")
       
       logger.info(f"Loaded hourly load data from {csv_path}")
       logger.info(f"Load shape stats: mean={load_shape.mean():.3f}, "
                  f"min={load_shape.min():.3f}, max={load_shape.max():.3f}")
       
       return load_shape
       
   except FileNotFoundError:
       logger.warning(f"Hourly load data file not found: {csv_path}")
       logger.warning("Using flat load profile instead")
       return np.ones(8760)
   except Exception as e:
       logger.error(f"Error loading hourly load data: {e}")
       logger.warning("Using flat load profile instead")
       return np.ones(8760)

# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FacilityLoad:
   """Single-source-of-truth for all demand-side data of the data-centre."""
   # ─────────────── Input parameters ────────────────
   total_gpus: int
   total_nodes: int
   node_power_avg_kw: float
   node_power_max_kw: float
   design_contingency_factor: float
   pue: float
   required_uptime_pct: float

   # ───────── Calculated scalar values ─────────────
   it_load_avg_mw: float
   it_load_max_mw: float
   facility_load_avg_mw: float
   facility_load_max_mw: float
   facility_load_design_mw: float
   design_ambient_temp_c: float
   # NEW: Disaggregated design loads
   it_load_design_mw: float
   cooling_load_design_mw: float

   annual_it_energy_mwh: float
   annual_facility_energy_mwh: float
   annual_facility_energy_gwh: float
   # NEW: Disaggregated annual energy
   annual_cooling_energy_mwh: float
   
   # ───────── Hourly arrays (optional on construction) ─────────
   hourly_pue: Optional[np.ndarray] = None
   hourly_it_load_mw: Optional[np.ndarray] = None
   hourly_facility_load_mw: Optional[np.ndarray] = field(init=False, default=None)
   # NEW: Disaggregated hourly load
   hourly_cooling_load_mw: Optional[np.ndarray] = field(init=False, default=None)

   # ───────── Weather cache ─────────
   tmy_weather: Optional[pd.DataFrame] = None
   

   # =============================================================
   # Post-processing / helpers
   # =============================================================
   def __post_init__(self) -> None:
       """Initialise missing arrays and derived hourly facility-load profile."""
       # 1. Hourly PUE: flat if not provided
       if self.hourly_pue is None:
           self.hourly_pue = np.full(8760, self.pue, dtype=float)

       # 2. Hourly IT load: flat if not provided
       if self.hourly_it_load_mw is None:
           self.hourly_it_load_mw = np.full(
               8760, self.it_load_avg_mw, dtype=float
           )

       # 3. Sanity checks
       if self.hourly_it_load_mw.shape != (8760,):
           raise ValueError("hourly_it_load_mw must have 8760 values")
       if self.hourly_pue.shape != (8760,):
           raise ValueError("hourly_pue must have 8760 values")

       # 4. Derived total-facility and cooling profile cache
       self.hourly_facility_load_mw = (
           self.hourly_it_load_mw * self.hourly_pue
       )
       self.hourly_cooling_load_mw = self.hourly_facility_load_mw - self.hourly_it_load_mw


   # -------------------------------------------------------------
   # Public helpers
   # -------------------------------------------------------------
   def total_load_profile(self) -> np.ndarray:
       """Return the 8760-hour total facility-load array (IT × PUE)."""
       if self.hourly_facility_load_mw is None:
           self.hourly_facility_load_mw = (
               self.hourly_it_load_mw * self.hourly_pue
           )
       return self.hourly_facility_load_mw

   def set_load_shape(self, shape: np.ndarray) -> None:
       """
       Replace the IT-load shape while preserving the same *annual average*.

       The supplied `shape` is normalised internally so that `shape.mean()==1`
       before it is multiplied by `it_load_avg_mw`.
       """
       shape = np.asarray(shape, dtype=float)
       if shape.shape != (8760,):
           raise ValueError("load shape must have 8760 elements")
       shape = shape / shape.mean()  # keep average unchanged
       self.hourly_it_load_mw = self.it_load_avg_mw * shape
       self.hourly_facility_load_mw = self.hourly_it_load_mw * self.hourly_pue
       self.hourly_cooling_load_mw = self.hourly_facility_load_mw - self.hourly_it_load_mw

   # -------------------------------------------------------------
   # String repr for CLI prints
   # -------------------------------------------------------------
   def __str__(self) -> str:
       return (
           f"\nFACILITY LOAD CALCULATION RESULTS\n"
           f"{'='*60}\n"
           f"Configuration:\n"
           f"  Total GPUs: {self.total_gpus:,}\n"
           f"  Total Nodes: {self.total_nodes:,}\n"
           f"  PUE (annual avg): {self.pue:.2f}\n"
           f"  Required Uptime: {self.required_uptime_pct:.1f}%\n"
           f"\nIT Load:\n"
           f"  Average: {self.it_load_avg_mw:.1f} MW\n"
           f"  Maximum: {self.it_load_max_mw:.1f} MW\n"
           f"\nFacility Load (IT + Cooling):\n"
           f"  Average: {self.facility_load_avg_mw:.1f} MW\n"
           f"  Maximum: {self.facility_load_max_mw:.1f} MW\n"
           f"  Design (+{(self.design_contingency_factor-1)*100:.0f}%): "
           f"{self.facility_load_design_mw:.1f} MW\n"
           f"    ↳ IT Design Load:     {self.it_load_design_mw:.1f} MW\n"
           f"    ↳ Cooling Design Load: {self.cooling_load_design_mw:.1f} MW\n"
           f"\nAnnual Energy Consumption:\n"
           f"  IT Energy: {self.annual_it_energy_mwh:,.0f} MWh\n"
           f"  Cooling Energy: {self.annual_cooling_energy_mwh:,.0f} MWh\n"
           f"  Total Facility Energy: {self.annual_facility_energy_mwh:,.0f} MWh "
           f"({self.annual_facility_energy_gwh:.1f} GWh)"
       )

# ═══════════════════════════════════════════════════════════════════════════
# CALCULATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def calculate_facility_load(
   total_gpus: int,
   config: Optional[Config] = None,
   pue: Optional[float] = None,
   required_uptime_pct: Optional[float] = None,
   hourly_pue: Optional[np.ndarray] = None,
   hourly_it_load_mw: Optional[np.ndarray] = None,
   tmy_weather: Optional[pd.DataFrame] = None,
   use_hourly_load_csv: bool = True,
   hourly_load_csv_path: str = "output_tables/hourly_load_data.csv"
) -> FacilityLoad:
   """
   Return a populated FacilityLoad object.
   
   Args:
       total_gpus: Total number of training GPUs
       config: Configuration object (if None, loads default)
       pue: Power Usage Effectiveness (overrides config if provided)
       required_uptime_pct: Required facility uptime percentage (overrides config if provided)
       hourly_pue: Optional 8760-hour PUE array
       hourly_it_load_mw: Optional 8760-hour IT load array (takes precedence over CSV)
       tmy_weather: Optional weather data
       use_hourly_load_csv: Whether to load hourly load shape from CSV
       hourly_load_csv_path: Path to hourly load CSV file
   """
   cfg = config or load_config()
   it_params = cfg.it_load
   
   pue = pue if pue is not None else it_params.default_pue
   required_uptime_pct = required_uptime_pct if required_uptime_pct is not None else it_params.default_required_uptime_pct
   
   # 1. Node count
   total_nodes = total_gpus // it_params.gpus_per_node
   if total_gpus % it_params.gpus_per_node != 0:
       logger.warning(
           "GPU count %d not divisible by %d; using %d full nodes.",
           total_gpus, it_params.gpus_per_node, total_nodes,
       )

   # 2. IT loads
   it_load_avg_mw = (total_nodes * it_params.node_power_avg_kw) / 1000.0
   it_load_max_mw = (total_nodes * it_params.node_power_max_kw) / 1000.0

   # 3. Facility loads (Total and Disaggregated)
   facility_load_avg_mw = it_load_avg_mw * pue
   facility_load_max_mw = it_load_max_mw * pue
   facility_load_design_mw = facility_load_max_mw * it_params.design_contingency_factor
   it_load_design_mw = it_load_max_mw * it_params.design_contingency_factor
   cooling_load_design_mw = (facility_load_max_mw - it_load_max_mw) * it_params.design_contingency_factor

   # 4. Energy (Total and Disaggregated)
   uptime_frac = required_uptime_pct / 100.0
   operating_hours = uptime_frac * 8760
   annual_it_energy_mwh = it_load_avg_mw * operating_hours
   annual_facility_energy_mwh = facility_load_avg_mw * operating_hours
   annual_cooling_energy_mwh = (facility_load_avg_mw - it_load_avg_mw) * operating_hours
   annual_facility_energy_gwh = annual_facility_energy_mwh / 1000.0

   # 5. Process hourly IT load array
   processed_hourly_it_load = None
   if hourly_it_load_mw is not None:
       if len(hourly_it_load_mw) != 8760:
           raise ValueError(f"hourly_it_load_mw must have 8760 elements")
       
       mean = hourly_it_load_mw.mean()
       if mean == 0:
           raise ValueError("hourly_it_load_mw mean must be > 0")
       
       normalized_shape = hourly_it_load_mw / mean
       processed_hourly_it_load = it_load_avg_mw * normalized_shape
       
   elif use_hourly_load_csv:
       load_shape = load_hourly_load_data(hourly_load_csv_path)
       processed_hourly_it_load = it_load_avg_mw * load_shape


   # Calculate design ambient temperature from TMY data
   if tmy_weather is not None and 'temp_air' in tmy_weather.columns:
       design_ambient_temp_c = np.percentile(tmy_weather['temp_air'].values, 99.0)
   else:
       design_ambient_temp_c = 40.0  # Default fallback for hot climate

   return FacilityLoad(
       total_gpus=total_gpus,
       total_nodes=total_nodes,
       node_power_avg_kw=it_params.node_power_avg_kw,
       node_power_max_kw=it_params.node_power_max_kw,
       pue=pue,
       design_contingency_factor=it_params.design_contingency_factor,
       required_uptime_pct=required_uptime_pct,
       it_load_avg_mw=it_load_avg_mw,
       it_load_max_mw=it_load_max_mw,
       facility_load_avg_mw=facility_load_avg_mw,
       facility_load_max_mw=facility_load_max_mw,
       facility_load_design_mw=facility_load_design_mw,
       it_load_design_mw=it_load_design_mw,
       cooling_load_design_mw=cooling_load_design_mw,
       annual_it_energy_mwh=annual_it_energy_mwh,
       annual_facility_energy_mwh=annual_facility_energy_mwh,
       annual_cooling_energy_mwh=annual_cooling_energy_mwh,
       annual_facility_energy_gwh=annual_facility_energy_gwh,
       hourly_it_load_mw=processed_hourly_it_load,
       hourly_pue=hourly_pue,
       design_ambient_temp_c=design_ambient_temp_c,  # NEW
       tmy_weather=tmy_weather, 
   )

# ... (The rest of the file remains the same) ...

def calculate_facility_load_with_csv(
   total_gpus: int,
   csv_path: str = "output_tables/hourly_load_data.csv",
   config: Optional[Config] = None,
   **kwargs
) -> FacilityLoad:
   """
   Convenience function to calculate facility load using CSV hourly data.
   
   Args:
       total_gpus: Total number of GPUs
       csv_path: Path to hourly load CSV file
       config: Configuration object
       **kwargs: Additional arguments passed to calculate_facility_load
       
   Returns:
       FacilityLoad object with hourly load profile from CSV
   """
   return calculate_facility_load(
       total_gpus=total_gpus,
       config=config,
       use_hourly_load_csv=True,
       hourly_load_csv_path=csv_path,
       **kwargs
   )


def get_annual_energy_gwh(
   total_gpus: int,
   config: Optional[Config] = None,
   pue: Optional[float] = None,
   required_uptime_pct: Optional[float] = None
) -> float:
   """
   Quick helper function to get annual energy consumption in GWh.
   
   Args:
       total_gpus: Total number of training GPUs
       config: Configuration object
       pue: Power Usage Effectiveness (overrides config if provided)
       required_uptime_pct: Required facility uptime percentage (overrides config if provided)
   
   Returns:
       Annual facility energy consumption in GWh
   """
   cfg = config or load_config()
   it_params = cfg.it_load
   
   # Use provided values or fall back to config
   pue = pue if pue is not None else it_params.default_pue
   required_uptime_pct = required_uptime_pct if required_uptime_pct is not None else it_params.default_required_uptime_pct
   
   total_nodes = total_gpus // it_params.gpus_per_node
   it_load_avg_mw = (total_nodes * it_params.node_power_avg_kw) / 1000
   facility_load_avg_mw = it_load_avg_mw * pue
   uptime_fraction = required_uptime_pct / 100.0
   operating_hours = 8760 * uptime_fraction
   annual_facility_energy_mwh = facility_load_avg_mw * operating_hours
   return annual_facility_energy_mwh / 1000


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
   # Example: Calculate load for a 10,000 GPU training cluster
   example_gpus = 10_000
   
   # Calculate with hourly load profile from CSV
   result = calculate_facility_load(
       total_gpus=example_gpus,
       required_uptime_pct=99,
       use_hourly_load_csv=True
   )
   
   print(result)
   
   # Show load profile statistics
   if result.hourly_it_load_mw is not None:
       print(f"\n{'='*60}")
       print("Hourly Load Profile Statistics:")
       print(f"Average IT Load: {result.hourly_it_load_mw.mean():.1f} MW")
       print(f"Minimum IT Load: {result.hourly_it_load_mw.min():.1f} MW")
       print(f"Maximum IT Load: {result.hourly_it_load_mw.max():.1f} MW")
       print(f"Load Factor: {result.hourly_it_load_mw.min()/result.hourly_it_load_mw.max():.3f}")
   
   # Quick calculations for different cluster sizes
   print(f"\n{'='*60}")
   print("Quick Reference - Facility Design Loads:")
   print(f"{'GPUs':<10} {'Nodes':<10} {'Design Load (MW)':<20}")
   print(f"{'-'*40}")
   
   cfg = load_config()
   for gpu_count in [1_000, 5_000, 10_000, 20_000, 50_000, 100_000]:
       facility = calculate_facility_load(gpu_count, config=cfg)
       design_load = facility.facility_load_design_mw
       nodes = gpu_count // cfg.it_load.gpus_per_node
       print(f"{gpu_count:<10,} {nodes:<10,} {design_load:<20.1f}")
   
   print(f"\n{'='*60}")
   print("Annual Energy Consumption at 99.9% Uptime:")
   print(f"{'GPUs':<10} {'Energy (GWh/year)':<20}")
   print(f"{'-'*30}")
   
   for gpu_count in [1_000, 5_000, 10_000, 20_000, 50_000, 100_000]:
       annual_gwh = get_annual_energy_gwh(gpu_count, config=cfg, required_uptime_pct=99.9)
       print(f"{gpu_count:<10,} {annual_gwh:<20.1f}")