"""
Datacenter Analyzer - Coordination Layer - Refactored to use config.py
Integrates location-specific PUE calculations with facility load modeling.
Provides PUE-aware facility load calculations for downstream tools.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from config import Config, load_config

# Import required modules
from pue_tool import select_optimal_cooling_system, fetch_weather_data
from it_facil import calculate_facility_load, FacilityLoad

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatacenterAnalyzer:
    """
    Analyzes datacenter facility requirements with location-specific cooling efficiency.
    Provides PUE-aware facility load calculations for downstream tools.
    """
    
    def __init__(
        self,
        latitude: float,
        longitude: float,
        total_gpus: int,
        case_numbers: Optional[List[int]] = None,
        lookup_dir: str = "output_tables",
        config: Optional[Config] = None
    ):
        """
        Initialize the datacenter analyzer.
        
        Args:
            latitude: Datacenter location latitude
            longitude: Datacenter location longitude
            total_gpus: Total number of GPUs in the facility
            case_numbers: List of cooling cases to evaluate (default: [1, 2, 14, 15, 16, 17])
            lookup_dir: Directory containing PUE lookup tables
            config: Configuration object
        """
        self.latitude = latitude
        self.longitude = longitude
        self.total_gpus = total_gpus
        self.case_numbers = case_numbers or [1, 2, 14, 15, 16, 17]
        self.lookup_dir = lookup_dir
        self.config = config or load_config()
        
        # Storage for results
        self.pue_results = None
        self.facility_load = None
        self.annual_pue = None
        self.hourly_pue = None
        self.weather_df = None  # Cache for TMY data
        
        logger.info(f"Initialized DatacenterAnalyzer for location ({latitude:.3f}, {longitude:.3f}) "
                   f"with {total_gpus:,} GPUs")
    
    def analyze_cooling(self) -> Dict:
        """
        Analyze cooling options for the datacenter location and select optimal system.
        
        Returns:
            Dictionary containing PUE analysis results
        """
        logger.info("Analyzing cooling systems for location...")
        
        # STEP 1: Fetch FULL TMY data once and cache it
        if self.weather_df is None:
            logger.info("Fetching full TMY weather data...")
            # Fetch the FULL PVGIS TMY dataset (not the reduced PUE version)
            from pvlib import iotools
            self.weather_df = iotools.get_pvgis_tmy(self.latitude, self.longitude)[0]
        
            # Create the reduced version for PUE tool
            pue_weather_df = pd.DataFrame({
                'hour': range(len(self.weather_df)),
                'temperature_c': self.weather_df['temp_air'].values,
                'humidity_pct': self.weather_df['relative_humidity'].values
            })
    
        # STEP 2: Get optimal cooling system using the reduced TMY data
        self.pue_results = select_optimal_cooling_system(
            latitude=self.latitude,
            longitude=self.longitude,
            case_numbers=self.case_numbers,
            lookup_dir=self.lookup_dir,
            weather_df=pue_weather_df  # Pass reduced version to PUE tool
        )

        
        # Extract key values
        self.annual_pue = self.pue_results['optimal_annual_pue']
        optimal_case = self.pue_results['optimal_case']
        
        # Get hourly PUE array for the optimal case
        if optimal_case is not None:
            self.hourly_pue = np.array(
                self.pue_results['hourly_data']['pue_profiles'][optimal_case]
            )
            logger.info(f"Selected cooling system: Case {optimal_case} with annual PUE={self.annual_pue:.3f}")
        else:
            # Fallback to default if no optimal case found
            logger.warning("No optimal cooling case found, using default PUE")
            self.annual_pue = self.config.it_load.default_pue
            self.hourly_pue = np.full(8760, self.config.it_load.default_pue)
        
        return self.pue_results
    
    def calculate_facility_load(
        self,
        gpus_per_node: Optional[int] = None,
        node_power_avg_kw: Optional[float] = None,
        node_power_max_kw: Optional[float] = None,
        design_contingency_factor: Optional[float] = None,
        required_uptime_pct: Optional[float] = None,
        design_pue_percentile: float = 99.0
    ) -> FacilityLoad:
        """
        Calculate facility loads using location-specific PUE.
        
        Args:
            gpus_per_node: Number of GPUs per compute node
            node_power_avg_kw: Average power per node (kW)
            node_power_max_kw: Maximum power per node (kW)
            design_contingency_factor: Safety factor for design
            required_uptime_pct: Required uptime percentage
            design_pue_percentile: Percentile of PUE to use for design (default 99th)
            
        Returns:
            FacilityLoad object with calculated values
        """
        # Use provided values or fall back to config
        gpus_per_node = gpus_per_node or self.config.it_load.gpus_per_node
        node_power_avg_kw = node_power_avg_kw or self.config.it_load.node_power_avg_kw
        node_power_max_kw = node_power_max_kw or self.config.it_load.node_power_max_kw
        design_contingency_factor = design_contingency_factor or self.config.it_load.design_contingency_factor
        required_uptime_pct = required_uptime_pct or self.config.it_load.default_required_uptime_pct
        
        # Ensure cooling analysis has been run
        if self.annual_pue is None:
            self.analyze_cooling()
        
        # Calculate design PUE using specified percentile
        design_pue = np.percentile(self.hourly_pue, design_pue_percentile)
        
        logger.info(f"Calculating facility loads:")
        logger.info(f"  Annual average PUE: {self.annual_pue:.3f}")
        logger.info(f"  {design_pue_percentile}th percentile PUE: {design_pue:.3f}")
        
        # Create FacilityLoad with all cached data injected
        self.facility_load = calculate_facility_load(
            total_gpus=self.total_gpus,
            config=self.config,
            pue=self.annual_pue,  # Use annual PUE for average calculations
            required_uptime_pct=required_uptime_pct,
            # Inject cached data:
            hourly_pue=self.hourly_pue,      # Hourly PUE profile
            tmy_weather=self.weather_df      # Cached TMY weather data
        )
        
               # Recalculate and update all design loads using the percentile-based PUE
        it_load_max_mw = self.facility_load.it_load_max_mw

        # Recalculate all three components consistently
        it_design_mw = it_load_max_mw * design_contingency_factor
        cooling_design_mw = it_load_max_mw * (design_pue - 1.0) * design_contingency_factor
        facility_design_mw = it_design_mw + cooling_design_mw

        # Update the facility load object with the new, consistent design values
        self.facility_load.it_load_design_mw = it_design_mw
        self.facility_load.cooling_load_design_mw = cooling_design_mw
        self.facility_load.facility_load_design_mw = facility_design_mw
        
        logger.info(f"  Design load: {facility_design_mw:.1f} MW "
                   f"(using {design_pue_percentile}th percentile PUE)")
        
        return self.facility_load
    
    def get_cached_weather_data(self) -> Optional[pd.DataFrame]:
        """
        Get cached TMY weather data for downstream tools.
        
        Returns:
            DataFrame with TMY weather data or None if not yet fetched
        """
        return self.weather_df
    
    def get_hourly_pue_profile(self) -> np.ndarray:
        """
        Get hourly PUE values for the optimal cooling system.
        This can be used by downstream tools like pvstoragesim.
        
        Returns:
            Array of 8760 hourly PUE values
        """
        if self.hourly_pue is None:
            self.analyze_cooling()
        
        return self.hourly_pue.copy()
    
    def get_annual_pue(self) -> float:
        """
        Get annual average PUE for the optimal cooling system.
        
        Returns:
            Annual average PUE value
        """
        if self.annual_pue is None:
            self.analyze_cooling()
        
        return self.annual_pue
    
    def get_cooling_system_info(self) -> Dict:
        """
        Get information about the selected cooling system.
        
        Returns:
            Dictionary with cooling system details
        """
        if self.pue_results is None:
            self.analyze_cooling()
        
        return {
            'optimal_case': self.pue_results['optimal_case'],
            'annual_pue': self.annual_pue,
            'max_pue': self.pue_results['optimal_max_pue'],
            'all_cases_evaluated': self.pue_results['all_cases'],
            'weather_stats': self.pue_results['weather_stats']
        }
    
    def get_summary(self) -> Dict:
        """
        Get a summary of facility analysis results.
        
        Returns:
            Dictionary containing facility analysis summary
        """
        # Ensure analyses have been run
        if self.facility_load is None:
            self.calculate_facility_load()
        
        summary = {
            'location': {
                'latitude': self.latitude,
                'longitude': self.longitude
            },
            'datacenter_config': {
                'total_gpus': self.total_gpus,
                'total_nodes': self.facility_load.total_nodes
            },
            'cooling': {
                'system': self.pue_results['optimal_case'] if self.pue_results else None,
                'annual_pue': self.annual_pue,
                'weather_conditions': self.pue_results['weather_stats'] if self.pue_results else None
            },
            'power_requirements': {
                'it_load_avg_mw': self.facility_load.it_load_avg_mw,
                'it_load_max_mw': self.facility_load.it_load_max_mw,
                'facility_load_avg_mw': self.facility_load.facility_load_avg_mw,
                'facility_load_max_mw': self.facility_load.facility_load_max_mw,
                'facility_load_design_mw': self.facility_load.facility_load_design_mw
            },
            'energy_consumption': {
                'annual_it_energy_mwh': self.facility_load.annual_it_energy_mwh,
                'annual_facility_energy_mwh': self.facility_load.annual_facility_energy_mwh,
                'annual_facility_energy_gwh': self.facility_load.annual_facility_energy_gwh
            }
        }
        
        return summary


# Example usage
if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Example: Analyze a 94,000 GPU datacenter in Phoenix, AZ
    analyzer = DatacenterAnalyzer(
        latitude=33.4484,
        longitude=-112.0740,
        total_gpus=94000,
        config=config
    )
    
    # Calculate facility requirements with location-specific cooling
    facility_load = analyzer.calculate_facility_load(required_uptime_pct=99.9)
    
    print(f"\n{'='*60}")
    print("DATACENTER FACILITY ANALYSIS")
    print(f"{'='*60}")
    
    # Get cooling system info
    cooling_info = analyzer.get_cooling_system_info()
    print(f"\nCooling System:")
    print(f"  Optimal Case: {cooling_info['optimal_case']}")
    print(f"  Annual PUE: {cooling_info['annual_pue']:.3f}")
    print(f"  Maximum PUE: {cooling_info['max_pue']:.3f}")
    
    # Display facility load results
    print(facility_load)
    
    # Show what's available for downstream tools
    print(f"\n{'='*60}")
    print("DATA AVAILABLE FOR DOWNSTREAM TOOLS:")
    print(f"{'='*60}")
    print(f"\n1. Facility Load Object:")
    print(f"   - Design capacity: {facility_load.facility_load_design_mw:.1f} MW")
    print(f"   - Annual energy: {facility_load.annual_facility_energy_gwh:.1f} GWh")
    print(f"   - Cached TMY data: {'Available' if facility_load.tmy_weather is not None else 'Not available'}")
    print(f"   - Hourly PUE profile: {'Available' if facility_load.hourly_pue is not None else 'Not available'}")
    
    print(f"\n2. Hourly PUE Profile:")
    hourly_pue = analyzer.get_hourly_pue_profile()
    print(f"   - Length: {len(hourly_pue)} hours")
    print(f"   - Mean: {np.mean(hourly_pue):.3f}")
    print(f"   - Std Dev: {np.std(hourly_pue):.3f}")
    print(f"   - Min: {np.min(hourly_pue):.3f}")
    print(f"   - Max: {np.max(hourly_pue):.3f}")