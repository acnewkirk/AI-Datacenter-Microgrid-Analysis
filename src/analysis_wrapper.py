"""
Simple wrapper for datacenter power analysis
Minimal implementation to get LCOE comparison results
"""

from lcoe_calc import get_state_from_coords, GRID_BASELINE_DATA, compare_datacenter_power_systems, calculate_gpu_idling_costs
from config import load_config
from typing import Tuple, Optional

def analyze_datacenter_simple(
    gpus: int,
    latitude: float,
    longitude: float,
    required_uptime: float = 99.0,
    gas_price: Optional[float] = None,
):
    """
    Simple datacenter analysis that outputs LCOE comparison table
    
    Args:
        gpus: Number of GPUs
        latitude: Latitude of location
        longitude: Longitude of location  
        required_uptime: Required uptime percentage (default 99%)
        gas_price: Natural gas price $/MMBtu (default 3.5)
    """
    
    # Run the comprehensive comparison
    comparison = compare_datacenter_power_systems(
        total_gpus=gpus,
        required_uptime_pct=required_uptime,
        location=(latitude, longitude),
        gas_price=gas_price
    )
    
    # Get the actual gas price used
    config = load_config()
    actual_gas_price = gas_price or config.costs.default_gas_price_mmbtu
    state = get_state_from_coords(latitude, longitude)
    if gas_price is None and state and state in GRID_BASELINE_DATA:
        _, _, _, state_gas_price = GRID_BASELINE_DATA[state]
        if state_gas_price is not None:
            actual_gas_price = state_gas_price
    
    # Print formatted results
    print(f"\n{'='*80}")
    print(f"LCOE COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"Location: ({latitude:.2f}, {longitude:.2f})")
    if state:
        print(f"State: {state}")
    print(f"Natural Gas Price: ${actual_gas_price:.2f}/MMBtu")
    print(f"IT Facility Design Load: {comparison.facility_load_mw:.0f} MW")
    print(f"Required Uptime: {required_uptime}%")
    
    # System capacities with land footprint
    print(f"\nSYSTEM CAPACITIES:")
    
    # Calculate land areas in km
    ac_land_km2 = ((comparison.ac_solar_mw * 5) + (comparison.ac_battery_mwh * 0.25)) * 0.00404686
    dc_land_km2 = ((comparison.dc_solar_mw * 5) + (comparison.dc_battery_mwh * 0.25)) * 0.00404686
    
    print(f"AC Solar+Storage: {comparison.ac_solar_mw:.0f} MW solar / {comparison.ac_battery_mw:.0f} MW ({comparison.ac_battery_mwh:.0f} MWh) battery - {ac_land_km2:.1f}sq km")
    print(f"DC Solar+Storage: {comparison.dc_solar_mw:.0f} MW solar / {comparison.dc_battery_mw:.0f} MW ({comparison.dc_battery_mwh:.0f} MWh) battery - {dc_land_km2:.1f}sq km")
    print(f"Natural Gas: {comparison.ng_nameplate_mw:.0f} MW - {comparison.ng_configuration}")
    
    # LCOE comparison table with idling costs
    print(f"\nLCOE COMPARISON TABLE:")
    print(f"{'-'*80}")
    print(f"{'System':<20} {'Construction':<12} {'Base LCOE':<15} {'Idling Cost':<15} {'Total LCOE':<15}")
    print(f"{'Type':<20} {'Years':<12} {'($/kWh)':<15} {'($M NPV)':<15} {'($/kWh)':<15}")
    print(f"{'-'*80}")
    
    # Calculate and display for each system
    systems = [
        ('AC Solar+Storage', comparison.ac_solar),
        ('DC Solar+Storage', comparison.dc_solar),
        ('Natural Gas', comparison.natural_gas),
        ('Grid', comparison.grid_baseline)
    ]
    
    for name, result in systems:
        # Calculate idling cost - pass config object instead of discount_rate
        idling_cost_npv = calculate_gpu_idling_costs(
            comparison.total_gpus,
            result.construction_years,
            config  # Pass the config object
        )
        
        # Calculate total LCOE including idling
        if result.energy_npv > 0:
            total_lcoe = (result.capex_npv + result.opex_npv + idling_cost_npv) / result.energy_npv / 1000
        else:
            total_lcoe = float('inf')
        
        print(f"{name:<20} {result.construction_years:<12.2f} ${result.lcoe:<14.4f} ${idling_cost_npv/1e6:<14.1f} ${total_lcoe:<14.4f}")
    
    print(f"{'-'*80}")


# Quick location presets for convenience
LOCATIONS = {
    'el_paso': (31.77, -106.46),
    'phoenix': (33.45, -112.07),
    'dallas': (32.78, -96.80),
    'seattle': (47.61, -122.33),
    'chicago': (41.88, -87.63)
}


if __name__ == "__main__":
    import sys
    
    # Check if command line arguments provided
    if len(sys.argv) > 1:
        # Parse command line arguments
        try:
            gpus = int(sys.argv[1])
            
            # Check if location is a preset or coordinates
            if len(sys.argv) == 3 and sys.argv[2] in LOCATIONS:
                # Using preset location
                lat, lon = LOCATIONS[sys.argv[2]]
                print(f"Using preset location: {sys.argv[2]}")
            elif len(sys.argv) >= 4:
                # Using direct coordinates
                lat = float(sys.argv[2])
                lon = float(sys.argv[3])
            else:
                print("Usage: python analysis_wrapper.py <gpus> <location_preset>")
                print("   or: python analysis_wrapper.py <gpus> <latitude> <longitude> [uptime] [gas_price]")
                print(f"Available presets: {list(LOCATIONS.keys())}")
                sys.exit(1)
            
            # Optional parameters
            uptime = float(sys.argv[4]) if len(sys.argv) > 4 else 99.0
            gas_price = float(sys.argv[5]) if len(sys.argv) > 5 else None  # Use None for location-based pricing
            
            # Run analysis
            analyze_datacenter_simple(
                gpus=gpus,
                latitude=lat,
                longitude=lon,
                required_uptime=uptime,
                gas_price=gas_price
            )
            
        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
            print("Usage: python analysis_wrapper.py <gpus> <location_preset>")
            print("   or: python analysis_wrapper.py <gpus> <latitude> <longitude> [uptime] [gas_price]")
            sys.exit(1)
    else:
        # No arguments - run default example
        print("Running default example (10,000 GPUs in El Paso)...")
        print("For custom analysis, use: python analysis_wrapper.py <gpus> <latitude> <longitude>")
        analyze_datacenter_simple(
            gpus=10000,
            latitude=31.77,
            longitude=-106.46
        )