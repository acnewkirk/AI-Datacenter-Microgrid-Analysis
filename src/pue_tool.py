
"""
PUE Selection Tool for Data Center Cooling Systems
Selects optimal cooling system based on annual average PUE for a given location
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from pvlib import iotools
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_weather_data(latitude: float, longitude: float) -> pd.DataFrame:
    """
    Fetch hourly weather data from PVGIS TMY dataset.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        
    Returns:
        DataFrame with columns: hour, temperature_c, humidity_pct
    """
    logger.info(f"Fetching weather data for ({latitude:.3f}, {longitude:.3f})")
    
    try:
        # Fetch TMY data from PVGIS (returns tuple of dataframe, metadata, and inputs)
        weather_data = iotools.get_pvgis_tmy(latitude, longitude)[0]
        logger.info(f"Successfully fetched PVGIS TMY data with {len(weather_data)} hours")
        
        # Extract temperature and humidity
        result_df = pd.DataFrame({
            'hour': range(len(weather_data)),
            'temperature_c': weather_data['temp_air'].values,  # Air temperature in Celsius
            'humidity_pct': weather_data['relative_humidity'].values  # Relative humidity in %
        })
        
        return result_df
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"Failed to fetch PVGIS data: {e}")
        raise ValueError(f"Could not fetch weather data for location ({latitude}, {longitude})")


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


def round_weather_conditions(temperature: float, humidity: float) -> Tuple[float, float]:
    """
    Round weather conditions to match lookup table resolution.
    
    Args:
        temperature: Temperature in Celsius
        humidity: Relative humidity in %
        
    Returns:
        Tuple of (rounded_temperature, rounded_humidity)
    """
    # Round temperature to nearest 0.5C
    rounded_temp = round(temperature * 2) / 2
    
    # Round humidity to nearest 1%
    rounded_humidity = round(humidity)
    
    return rounded_temp, rounded_humidity


def lookup_pue_value(
    lookup_df: pd.DataFrame, 
    temperature: float, 
    humidity: float
) -> Optional[float]:
    """
    Look up PUE value for given weather conditions with nearest neighbor fallback.
    
    Args:
        lookup_df: PUE lookup table DataFrame
        temperature: Temperature in Celsius
        humidity: Relative humidity in %
        
    Returns:
        PUE value (uses interpolation if exact match not found)
    """
    # Round to lookup table resolution
    temp_rounded, humidity_rounded = round_weather_conditions(temperature, humidity)
    
    # First try exact match
    mask = (lookup_df['T_oa'] == temp_rounded) & (lookup_df['RH_oa'] == humidity_rounded)
    matching_rows = lookup_df[mask]
    
    if len(matching_rows) > 0:
        if len(matching_rows) > 1:
            logger.warning(f"Multiple PUE values found for T={temp_rounded}C, RH={humidity_rounded}%, using first")
        return matching_rows.iloc[0]['pue']
    
    # No exact match found - use nearest neighbor interpolation
    logger.debug(f"No exact match for T={temp_rounded}C, RH={humidity_rounded}% - using nearest neighbor")
    
    # Calculate Euclidean distance to all points
    # Weight temperature more heavily than humidity (temperature typically more critical for cooling)
    temp_weight = 1.0
    humidity_weight = 0.1
    
    distances = np.sqrt(
        (temp_weight * (lookup_df['T_oa'] - temp_rounded))**2 + 
        (humidity_weight * (lookup_df['RH_oa'] - humidity_rounded))**2
    )
    
    # Find nearest neighbor
    nearest_idx = distances.idxmin()
    nearest_pue = lookup_df.loc[nearest_idx, 'pue']
    
    logger.debug(f"Using nearest neighbor: T={lookup_df.loc[nearest_idx, 'T_oa']}C, "
                f"RH={lookup_df.loc[nearest_idx, 'RH_oa']}%, PUE={nearest_pue:.3f}")
    
    return nearest_pue


def calculate_annual_pue(
    weather_df: pd.DataFrame,
    lookup_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate annual average PUE and statistics for a cooling system.
    
    Args:
        weather_df: Hourly weather data
        lookup_df: PUE lookup table
        
    Returns:
        Dictionary with annual_pue, max_pue, valid_hours
    """
    hourly_pue = []
    valid_hours = 0
    
    for _, row in weather_df.iterrows():
        pue = lookup_pue_value(lookup_df, row['temperature_c'], row['humidity_pct'])
        
        if pue is not None and pue > 0:  # Filter out invalid PUE values
            hourly_pue.append(pue)
            valid_hours += 1
        else:
            # Use a high penalty PUE for missing/invalid values
            hourly_pue.append(10.0)  # Penalty value
    
    if valid_hours == 0:
        logger.error("No valid PUE values found for entire year")
        return {
            'annual_pue': float('inf'),
            'max_pue': float('inf'),
            'valid_hours': 0
        }
    
    hourly_pue = np.array(hourly_pue)
    
    return {
        'annual_pue': np.mean(hourly_pue),
        'max_pue': np.max(hourly_pue[hourly_pue < 10.0]),  # Max of valid values only
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