
"""
PUE Selection Tool for Data Center Cooling Systems
Selects optimal cooling system based on annual average PUE for a given location
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
from nsrdb_loader import get_nsrdb_tmy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_weather_data(latitude: float, longitude: float) -> pd.DataFrame:
    """
    Fetch hourly weather data from NSRDB TMY dataset.

    Args:
        latitude: Location latitude
        longitude: Location longitude

    Returns:
        DataFrame with columns: hour, temperature_c, humidity_pct
    """
    logger.info(f"Fetching weather data for ({latitude:.3f}, {longitude:.3f})")

    weather_data = get_nsrdb_tmy(latitude, longitude)
    logger.info(f"Successfully fetched NSRDB TMY data with {len(weather_data)} hours")

    result_df = pd.DataFrame({
        'hour': range(len(weather_data)),
        'temperature_c': weather_data['temp_air'].values,
        'humidity_pct': weather_data['relative_humidity'].values,
    })

    return result_df


def load_pue_lookup_table(case_number: int, lookup_dir: str = "output_tables") -> pd.DataFrame:
    """
    Load PUE lookup table for a specific cooling system case.
    
    Args:
        case_number: Cooling system case number
        lookup_dir: Directory containing lookup tables
        
    Returns:
        DataFrame with columns: T_oa, RH_oa, pue
    """
    lookup_file = Path(lookup_dir) / f"lookup_PUE_case{case_number}.csv"
    
    if not lookup_file.exists():
        raise FileNotFoundError(f"Lookup table not found: {lookup_file}")
    
    logger.info(f"Loading lookup table: {lookup_file}")
    df = pd.read_csv(lookup_file)
    
    # Verify required columns exist
    required_cols = ['T_oa', 'RH_oa', 'pue']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Lookup table missing required columns. Found: {df.columns.tolist()}")
    
    return df[required_cols]


def calculate_annual_pue(
    weather_df: pd.DataFrame,
    lookup_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate annual average PUE and statistics for a cooling system.

    Vectorized: rounds weather to lookup resolution, merges on (T_oa, RH_oa),
    then uses KDTree nearest-neighbor for any unmatched hours.

    Args:
        weather_df: Hourly weather data
        lookup_df: PUE lookup table

    Returns:
        Dictionary with annual_pue, max_pue, valid_hours, hourly_pue
    """
    from scipy.spatial import KDTree

    # Vectorized rounding to lookup table resolution
    rounded_temp = np.round(weather_df['temperature_c'].values * 2) / 2  # nearest 0.5°C
    rounded_humidity = np.round(weather_df['humidity_pct'].values)        # nearest 1%

    # Build a rounded weather frame and merge onto lookup table
    weather_rounded = pd.DataFrame({'T_oa': rounded_temp, 'RH_oa': rounded_humidity})
    merged = weather_rounded.merge(lookup_df, on=['T_oa', 'RH_oa'], how='left')
    hourly_pue = merged['pue'].values.copy()

    # Fill unmatched hours via KDTree nearest-neighbor (same weighting as old code)
    missing_mask = np.isnan(hourly_pue)
    if missing_mask.any():
        temp_weight = 1.0
        humidity_weight = 0.1
        lookup_coords = np.column_stack([
            lookup_df['T_oa'].values * temp_weight,
            lookup_df['RH_oa'].values * humidity_weight
        ])
        tree = KDTree(lookup_coords)

        query_coords = np.column_stack([
            rounded_temp[missing_mask] * temp_weight,
            rounded_humidity[missing_mask] * humidity_weight
        ])
        _, indices = tree.query(query_coords)
        hourly_pue[missing_mask] = lookup_df['pue'].values[indices]

    # Replace invalid PUE values (≤0) with penalty
    invalid_mask = hourly_pue <= 0
    hourly_pue[invalid_mask] = 10.0

    valid_hours = int(np.sum(~invalid_mask))

    if valid_hours == 0:
        logger.error("No valid PUE values found for entire year")
        return {
            'annual_pue': float('inf'),
            'max_pue': float('inf'),
            'valid_hours': 0
        }

    return {
        'annual_pue': np.mean(hourly_pue),
        'max_pue': np.max(hourly_pue[~invalid_mask]),  # Max of valid values only
        'valid_hours': valid_hours,
        'hourly_pue': hourly_pue  # Store for future integration
    }


def select_optimal_cooling_system(
    latitude: float,
    longitude: float,
    case_numbers: List[int] = [1,2, 14, 15, 16, 17],
    lookup_dir: str = "output_tables", 
    weather_df: Optional[pd.DataFrame] = None     # ← NEW
) -> Dict:
    """
    Select the data center cooling system with lowest annual average PUE.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude  
        case_numbers: List of cooling system cases to evaluate
        lookup_dir: Directory containing lookup tables
        
    Returns:
        Dictionary with results including optimal case, PUE values, and hourly data
    """
    logger.info(f"Evaluating cooling systems for location ({latitude:.3f}, {longitude:.3f})")
    
    # Fetch weather data once
    if weather_df is None:
        weather_df = fetch_weather_data(latitude, longitude)
    
    # Store results for all cases
    results = {
        'location': {'latitude': latitude, 'longitude': longitude},
        'weather_stats': {
            'mean_temperature_c': weather_df['temperature_c'].mean(),
            'min_temperature_c': weather_df['temperature_c'].min(),
            'max_temperature_c': weather_df['temperature_c'].max(),
            'mean_humidity_pct': weather_df['humidity_pct'].mean()
        },
        'all_cases': {},
        'hourly_data': {
            'temperature': weather_df['temperature_c'].tolist(),
            'humidity': weather_df['humidity_pct'].tolist(),
            'pue_profiles': {}
        }
    }
    
    best_case = None
    best_pue = float('inf')
    
    # Evaluate each cooling system case
    for case_num in case_numbers:
        try:
            # Load lookup table
            lookup_df = load_pue_lookup_table(case_num, lookup_dir)
            
            # Calculate annual PUE
            pue_stats = calculate_annual_pue(weather_df, lookup_df)
            
            # Store results
            results['all_cases'][case_num] = {
                'annual_pue': pue_stats['annual_pue'],
                'max_pue': pue_stats['max_pue'],
                'valid_hours': pue_stats['valid_hours']
            }
            
            # Store hourly PUE profile for future integration
            results['hourly_data']['pue_profiles'][case_num] = pue_stats['hourly_pue'].tolist()
            
            logger.info(f"Case {case_num}: Annual PUE = {pue_stats['annual_pue']:.3f}, "
                       f"Max PUE = {pue_stats['max_pue']:.3f}")
            
            # Track best case
            if pue_stats['annual_pue'] < best_pue:
                best_pue = pue_stats['annual_pue']
                best_case = case_num
                
        except Exception as e:
            logger.error(f"Error evaluating case {case_num}: {e}")
            results['all_cases'][case_num] = {
                'annual_pue': float('inf'),
                'max_pue': float('inf'),
                'valid_hours': 0,
                'error': str(e)
            }
    
    # Set optimal case results
    if best_case is not None:
        results['optimal_case'] = best_case
        results['optimal_annual_pue'] = results['all_cases'][best_case]['annual_pue']
        results['optimal_max_pue'] = results['all_cases'][best_case]['max_pue']
    else:
        results['optimal_case'] = None
        results['optimal_annual_pue'] = float('inf')
        results['optimal_max_pue'] = float('inf')
        
    logger.info(f"Optimal cooling system: Case {best_case} with annual PUE = {best_pue:.3f}")
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Test with a specific location (e.g., Phoenix, AZ)
    test_latitude = 33.4484
    test_longitude = -112.0740
    
    try:
        results = select_optimal_cooling_system(
            latitude=test_latitude,
            longitude=test_longitude,
            case_numbers=[1,2, 14,15, 16, 17],
            lookup_dir="output_tables"
        )
        
        print("\n=== PUE Selection Results ===")
        print(f"Location: ({test_latitude}, {test_longitude})")
        print(f"Weather stats: {results['weather_stats']}")
        print(f"\nOptimal cooling system: Case {results['optimal_case']}")
        print(f"Annual average PUE: {results['optimal_annual_pue']:.3f}")
        print(f"Maximum PUE: {results['optimal_max_pue']:.3f}")
        print("\nAll cases evaluated:")
        for case, stats in results['all_cases'].items():
            print(f"  Case {case}: Annual PUE = {stats['annual_pue']:.3f}, "
                  f"Max PUE = {stats['max_pue']:.3f}")
            
    except Exception as e:
        logger.error(f"Error in PUE selection: {e}")
        raise