"""
PV Storage Simulation Engine 
Simulates hour-by-hour operation of solar+battery microgrids with architecture-specific power flow modeling.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import logging
import tzfpy  # Timezone lookup library
import rainflow  # For cycle counting
from pvlib import pvsystem, modelchain, location
from it_facil import FacilityLoad
from power_systems_estimator import PowerFlowAnalyzer
from config import Config, load_config
from nsrdb_loader import get_nsrdb_tmy

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

@dataclass
class SimulationResult:
    """Results from hourly simulation"""
    uptime_pct: float
    energy_served_pct: float
    solar_generation_mwh: float
    solar_curtailed_mwh: float
    load_served_mwh: float
    battery_charged_mwh: float
    battery_discharged_mwh: float
    battery_cycles_per_year: float
    unmet_load_mwh: float
    stress_features: Optional[Dict] = None 
    hourly_data: Optional[pd.DataFrame] = None


def get_solar_generation(
    latitude: float,
    longitude: float,
    system_type: str = "single-axis",
    surface_tilt: float = 20,
    surface_azimuth: float = 180,
    facility_load: Optional[FacilityLoad] = None,
) -> pd.DataFrame:
    """
    Fetch hourly solar generation profile from NSRDB or use cached TMY data.

    Returns a raw normalized profile, independent of architecture/topology;
    inverter clipping is applied downstream in evaluate_system.
    """
    logger.info(f"Fetching solar data for ({latitude:.3f}, {longitude:.3f})")
    
    if system_type.lower() == "fixed-tilt":
        mount = pvsystem.FixedMount(
            surface_tilt=surface_tilt, surface_azimuth=surface_azimuth
        )
    elif system_type.lower() == "single-axis":
        mount = pvsystem.SingleAxisTrackerMount()
    else:
        raise ValueError("system_type must be either 'fixed-tilt' or 'single-axis'")

    try:
        timezone_str = tzfpy.get_tz(longitude, latitude)
    except:
        timezone_str = 'UTC'
    
    pvlib_config = {
        "module_parameters": {"pdc0": 1, "gamma_pdc": -0.004},
        "inverter_parameters": {"pdc0": 1, "eta_inv_nom": 1},
        "temperature_model_parameters": {"a": -3.56, "b": -0.075, "deltaT": 3},
    }
    
    array = pvsystem.Array(
        mount=mount,
        module_parameters=pvlib_config["module_parameters"],
        temperature_model_parameters=pvlib_config["temperature_model_parameters"]
    )
    site = location.Location(latitude, longitude)

    pv_system = pvsystem.PVSystem(
        arrays=[array],
        inverter_parameters=pvlib_config["inverter_parameters"],
    )

    model = modelchain.ModelChain(
        pv_system, site, aoi_model="physical", spectral_model="no_loss"
    )

    weather_data = None
    if facility_load and hasattr(facility_load, 'tmy_weather') and facility_load.tmy_weather is not None:
        try:
            weather_data = facility_load.tmy_weather
            logger.info("Using cached TMY weather data for solar generation")
        except Exception as e:
            logger.warning(f"Failed to use cached TMY data: {e}, fetching fresh data")
            weather_data = None
    
    if weather_data is None:
        weather_data = get_nsrdb_tmy(latitude, longitude)
        logger.info("Successfully fetched fresh NSRDB TMY data")

    model.run_model(weather_data)

    solar_generation_df = model.results.ac.reset_index()
    
    if 'p_mp' in solar_generation_df.columns:
        dc_power_values = solar_generation_df['p_mp'].values
    else:
        power_cols = [col for col in solar_generation_df.columns if 'p' in col.lower()]
        if power_cols:
            dc_power_values = solar_generation_df[power_cols[0]].values
        else:
            raise ValueError("Cannot find power output column in results")
    
    max_dc = np.max(dc_power_values)
    if max_dc > 0:
        p_dc_normalized = dc_power_values / max_dc
    else:
        raise ValueError("Zero solar generation detected")

    solar_profile = pd.DataFrame({
        'hour': range(len(dc_power_values)),
        'p_dc': p_dc_normalized
    })
    
    capacity_factor = p_dc_normalized.mean()
    data_source = "cached TMY" if facility_load and hasattr(facility_load, 'tmy_weather') and facility_load.tmy_weather is not None else "fresh NSRDB"
    logger.info(f"Solar profile generated from {data_source} - capacity factor: {capacity_factor:.3f}")
    
    return solar_profile


def simulate_battery_operation(
    solar_dc_mw: np.ndarray,
    battery_power_mw: float,
    battery_energy_mwh: float,
    hourly_bus_load_mw: np.ndarray,
    solar_to_bus_mult: float,
    solar_to_battery_mult: float,
    battery_to_bus_mult: float,
    inverter_cap_mw_dc: Optional[float] = None,
    initial_soc: float = 50,
) -> Dict[str, np.ndarray]:
    """
    Vectorized one-year battery dispatch. Load is served first from solar; any
    surplus solar charges the battery drawn *upstream* of the load bus (via
    `solar_to_battery_mult`), and the battery discharges to the load bus (via
    `battery_to_bus_mult`).

    `inverter_cap_mw_dc` (mv_coupled AC only) caps the solar->load path at the
    inverter rating; PV above the cap is still available to the DC-coupled
    battery, so clipped energy is recaptured rather than discarded. Pass None
    (DC architectures, and AC lv_direct whose profile is pre-clipped) to leave
    the load path uncapped.
    """
    n_hours = len(solar_dc_mw)

    if len(hourly_bus_load_mw) != n_hours:
        raise ValueError(f"Bus load array length ({len(hourly_bus_load_mw)}) must match solar array length ({n_hours})")

    battery_soc = np.zeros(n_hours + 1)
    battery_soc[0] = (initial_soc / 100.0) * battery_energy_mwh
    battery_charge_mw = np.zeros(n_hours)
    battery_discharge_mw = np.zeros(n_hours)
    curtailed_solar_mw = np.zeros(n_hours)
    unmet_load_mw = np.zeros(n_hours)

    # Solar that can reach the load bus is inverter-limited when a cap is set
    # (mv_coupled AC). The DC-coupled battery can still absorb PV above the cap.
    if inverter_cap_mw_dc is not None:
        solar_dc_to_bus = np.minimum(solar_dc_mw, inverter_cap_mw_dc)
    else:
        solar_dc_to_bus = solar_dc_mw
    solar_at_bus_mw = solar_dc_to_bus / solar_to_bus_mult
    power_balance_at_bus = solar_at_bus_mw - hourly_bus_load_mw

    for t in range(n_hours):
        if power_balance_at_bus[t] > 0:
            # Load fully served by solar. Charge the battery from leftover PV
            # measured upstream (DC) — for mv_coupled AC this includes the
            # clipped portion above the inverter cap.
            dc_used_for_load = hourly_bus_load_mw[t] * solar_to_bus_mult
            leftover_pv_dc = solar_dc_mw[t] - dc_used_for_load
            max_charge_at_battery = min(battery_power_mw, battery_energy_mwh - battery_soc[t])
            charge_at_battery = min(leftover_pv_dc / solar_to_battery_mult, max_charge_at_battery)
            battery_charge_mw[t] = charge_at_battery
            battery_soc[t + 1] = battery_soc[t] + charge_at_battery
            dc_used_for_charging = charge_at_battery * solar_to_battery_mult
            curtailed_solar_mw[t] = leftover_pv_dc - dc_used_for_charging
        else:
            # Deficit: discharge to the bus. No charging this hour, so any
            # clipped PV during an inverter-capped deficit hour is curtailed
            # (a thin edge case in an overbuilt islanded system).
            deficit_at_bus = -power_balance_at_bus[t]
            max_discharge_from_battery = min(battery_power_mw, battery_soc[t])
            max_discharge_at_bus = max_discharge_from_battery / battery_to_bus_mult
            discharge_at_bus = min(deficit_at_bus, max_discharge_at_bus)
            discharge_from_battery = discharge_at_bus * battery_to_bus_mult
            battery_discharge_mw[t] = discharge_from_battery
            battery_soc[t + 1] = battery_soc[t] - discharge_from_battery
            unmet_load_mw[t] = deficit_at_bus - discharge_at_bus

    return {
        'battery_soc': battery_soc[:-1],
        'battery_charge_mw': battery_charge_mw,
        'battery_discharge_mw': battery_discharge_mw,
        'curtailed_solar_mw': curtailed_solar_mw,
        'unmet_load_mw': unmet_load_mw,
        'solar_at_load_mw': solar_at_bus_mw
    }


def extract_stress_features(hd: pd.DataFrame) -> Optional[Dict]:
    """Extract battery stress features for the degradation surrogate: mean SOC,
    rainflow half-cycle counts and EFC contributions binned by depth of
    discharge (right-closed bins (a, b] over [0, 1]), and discharge throughput.
    SOC is normalized by the maximum SOC reached during the year (intended
    simplification; equals nameplate whenever the battery fully charges)."""
    try:
        cap = hd['battery_soc_mwh'].max()
        if not np.isfinite(cap) or cap == 0:
            return None

        soc = hd['battery_soc_mwh'].to_numpy() / cap
        mean_soc = float(soc.mean())
        
        half_cycles = rainflow.extract_cycles(soc)
        
        dod_bins = np.linspace(0, 1.0, 11)
        cycle_counts_by_dod = np.zeros(len(dod_bins) - 1)
        efc_sum_by_dod = np.zeros(len(dod_bins) - 1)

        # Right-closed bins (a, b], matching pd.cut(right=True). Values outside
        # [0, 1] are dropped.
        do_ds_arr = np.fromiter((cycle[0] for cycle in half_cycles), dtype=float)
        if do_ds_arr.size:
            n_bins = len(dod_bins) - 1
            bin_idx = np.digitize(do_ds_arr, dod_bins, right=True) - 1
            valid = (bin_idx >= 0) & (bin_idx < n_bins)
            np.add.at(cycle_counts_by_dod, bin_idx[valid], 0.5)
            np.add.at(efc_sum_by_dod, bin_idx[valid], 0.5 * do_ds_arr[valid])

        discharge_throughput = float(np.abs(hd['battery_discharge_mw'].to_numpy()).sum())

        return {
            'mean_soc': mean_soc,
            'cycle_counts_by_dod': cycle_counts_by_dod.tolist(),
            'efc_sum_by_dod': efc_sum_by_dod.tolist(),
            'throughput_mwh': discharge_throughput
        }
    except Exception as e:
        logger.warning(f"Could not calculate stress features due to an error: {e}")
        return None

def evaluate_system(
    latitude: float,
    longitude: float,
    solar_capacity_mw: float,
    battery_power_mw: float,
    facility_load: FacilityLoad,
    hourly_pue: Optional[np.ndarray] = None,
    architecture: str = "ac_coupled",
    topology: str = "mv_coupled",
    efficiency_params: Optional[Config] = None,
    solar_profile: Optional[pd.DataFrame] = None,
    battery_duration_hours: float = 4.0,
    return_hourly: bool = False
) -> SimulationResult:
    """
    Simulate solar+battery system operation for one year.
    """
    cfg = efficiency_params or load_config()

    if solar_profile is None:
        solar_profile = get_solar_generation(
            latitude, longitude, facility_load=facility_load
        )

    hourly_it_load_mw = facility_load.hourly_it_load_mw
    hourly_cooling_load_mw = facility_load.hourly_cooling_load_mw

    analyzer = PowerFlowAnalyzer(cfg, topology=topology)
    mult = analyzer.get_bus_architecture_multipliers(architecture)

    hourly_bus_load_mw = (hourly_it_load_mw * mult['bus_to_it'] +
                          hourly_cooling_load_mw * mult['bus_to_cooling'])

    battery_energy_mwh = battery_power_mw * battery_duration_hours
    solar_dc_mw = solar_profile['p_dc'].values * solar_capacity_mw

    # Inverter clipping, by architecture/topology:
    #  - AC lv_direct: the battery sits behind the single inverter, so PV
    #    above the inverter rating is unrecoverable — clip the PV array.
    #  - AC mv_coupled: DC-coupled storage taps the PV DC link pre-inverter —
    #    cap only the solar->load path; the battery recaptures PV above it.
    #  - DC architectures: no inverter on the PV path; no cap.
    inverter_cap_mw_dc = None
    if architecture == "ac_coupled":
        inverter_limit_mw = solar_capacity_mw / cfg.efficiency.inverter_load_ratio
        if topology == "lv_direct":
            solar_dc_mw = np.minimum(solar_dc_mw, inverter_limit_mw)
        else:  # mv_coupled
            inverter_cap_mw_dc = inverter_limit_mw

    sim_results = simulate_battery_operation(
        solar_dc_mw=solar_dc_mw,
        battery_power_mw=battery_power_mw,
        battery_energy_mwh=battery_energy_mwh,
        hourly_bus_load_mw=hourly_bus_load_mw,
        solar_to_bus_mult=mult['solar_to_bus'],
        solar_to_battery_mult=mult['solar_to_battery'],
        battery_to_bus_mult=mult['battery_to_bus'],
        inverter_cap_mw_dc=inverter_cap_mw_dc,
        initial_soc=75.0
    )
    
    total_hours = len(solar_dc_mw)
    hours_online = np.sum(sim_results['unmet_load_mw'] < 0.001)
    
    total_load_mwh = np.sum(hourly_it_load_mw) + np.sum(hourly_cooling_load_mw)
    total_bus_load_mwh = np.sum(hourly_bus_load_mw)
    total_unmet_mwh_at_bus = np.sum(sim_results['unmet_load_mw'])

    avg_bus_to_load_multiplier = total_bus_load_mwh / total_load_mwh if total_load_mwh > 0 else 1
    total_unmet_mwh_at_load = total_unmet_mwh_at_bus / avg_bus_to_load_multiplier
    
    total_solar_generation_mwh = np.sum(solar_dc_mw)
    total_solar_curtailed_mwh = np.sum(sim_results['curtailed_solar_mw'])
    total_served_mwh = total_load_mwh - total_unmet_mwh_at_load
    total_battery_charged_mwh = np.sum(sim_results['battery_charge_mw'])
    total_battery_discharged_mwh = np.sum(sim_results['battery_discharge_mw'])
    
    battery_cycles = total_battery_discharged_mwh / battery_energy_mwh if battery_energy_mwh > 0 else 0
    
    uptime_pct = (hours_online / total_hours) * 100
    energy_served_pct = (total_served_mwh / total_load_mwh) * 100 if total_load_mwh > 0 else 0
    
    hourly_data = None
    stress_features = None

    if return_hourly:
        total_load_mw = hourly_it_load_mw + hourly_cooling_load_mw
        hourly_data = pd.DataFrame({
            'hour': range(total_hours),
            'solar_dc_mw': solar_dc_mw,
            'solar_at_load_mw': sim_results['solar_at_load_mw'],
            'it_load_mw': hourly_it_load_mw,
            'cooling_load_mw': hourly_cooling_load_mw,
            'total_load_mw': total_load_mw,
            'pue': facility_load.hourly_pue,
            'battery_soc_mwh': sim_results['battery_soc'],
            'battery_charge_mw': sim_results['battery_charge_mw'],
            'battery_discharge_mw': sim_results['battery_discharge_mw'],
            'curtailed_solar_mw': sim_results['curtailed_solar_mw'],
            'unmet_load_mw': sim_results['unmet_load_mw'],
            'load_served_mw': total_load_mw - (sim_results['unmet_load_mw'] / avg_bus_to_load_multiplier)
        })
        stress_features = extract_stress_features(hourly_data)
    
    logger.info(f"Results: {uptime_pct:.2f}% uptime, {energy_served_pct:.2f}% energy served, "
                f"{battery_cycles:.1f} cycles/year")
    
    return SimulationResult(
        uptime_pct=uptime_pct,
        energy_served_pct=energy_served_pct,
        solar_generation_mwh=total_solar_generation_mwh,
        solar_curtailed_mwh=total_solar_curtailed_mwh,
        load_served_mwh=total_served_mwh,
        battery_charged_mwh=total_battery_charged_mwh,
        battery_discharged_mwh=total_battery_discharged_mwh,
        battery_cycles_per_year=battery_cycles,
        unmet_load_mwh=total_unmet_mwh_at_load,
        stress_features=stress_features,
        hourly_data=hourly_data
    )


