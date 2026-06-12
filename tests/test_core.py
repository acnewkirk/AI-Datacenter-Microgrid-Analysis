"""
Offline regression tests for the core src/ modules.

Everything here runs without network access or cached weather data. Golden
values were generated from the code state of 2026-06-11 (post bug-fix pass:
grid single-derate, mid-year idling, EUE shortfall) and pin current behavior.
"""
import math
from pathlib import Path

import numpy as np
import pytest

from config import Config, load_config, save_config
from power_systems_estimator import PowerFlowAnalyzer

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config()


# ═══════════════════════════════════════════════════════════════════════════
# Construction schedules and operations start
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("years", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.33, 3.5, 4.0, 4.5, 5.25])
def test_construction_schedule_sums_to_one(years):
    from lcoe_calc import get_construction_schedule
    schedule = get_construction_schedule(years)
    assert sum(f for _, f in schedule) == pytest.approx(1.0)
    assert all(f > 0 for _, f in schedule)


def test_construction_schedule_regimes():
    from lcoe_calc import get_construction_schedule
    assert get_construction_schedule(1.0) == [(0.0, 1.0)]
    assert get_construction_schedule(1.5) == [(0.0, 0.4), (1.0, 0.6)]
    assert get_construction_schedule(2.5) == [(0.0, 0.3), (1.0, 0.4), (2.0, 0.3)]


def test_operations_start_info():
    from lcoe_calc import get_operations_start_info
    # Convention: ops start in calendar year ceil(construction_years); a
    # fractional construction year makes that first operational year partial.
    assert get_operations_start_info(1.0) == (1, 1.0)
    assert get_operations_start_info(2.0) == (2, 1.0)
    assert get_operations_start_info(1.5) == (2, 0.5)
    start, frac = get_operations_start_info(2.3)
    assert start == 3
    assert frac == pytest.approx(0.7)
    # NG production case: 40 months
    start, frac = get_operations_start_info(40 / 12)
    assert start == 4
    assert frac == pytest.approx(1 - (40 / 12 - 3))


# ═══════════════════════════════════════════════════════════════════════════
# Power-flow multipliers
# ═══════════════════════════════════════════════════════════════════════════

# Golden values computed from default config, 2026-06-11.
GOLDEN_MULTIPLIERS = {
    ("lv_direct", "ac_coupled"): {
        "solar_to_bus": 1.062816719382376,
        "battery_to_bus": 1.1203071895187777,
        "solar_to_battery": 1.1669866557487267,
        "bus_to_it": 1.0745755426606491,
        "bus_to_cooling": 1.0309278350515465,
    },
    ("lv_direct", "dc_coupled"): {
        "solar_to_bus": 1.0306101521283646,
        "battery_to_bus": 1.097443777487782,
        "solar_to_battery": 1.108529068169477,
        "bus_to_it": 1.0307153164296021,
        "bus_to_cooling": 1.0204081632653061,
    },
    ("mv_coupled", "ac_coupled"): {
        "solar_to_bus": 1.1178178277804873,
        "battery_to_bus": 1.190306748802346,
        "solar_to_battery": 1.0864693397129044,
        "bus_to_it": 1.0745755426606491,
        "bus_to_cooling": 1.0309278350515465,
    },
    ("mv_coupled", "dc_coupled"): {
        "solar_to_bus": 1.0731051146692676,
        "battery_to_bus": 1.1198405892732473,
        "solar_to_battery": 1.1086421833805147,
        "bus_to_it": 1.0307153164296021,
        "bus_to_cooling": 1.0204081632653061,
    },
    ("lv_direct", "natural_gas"): {
        "grid_to_bus": 1.1301851739601863,
        "bus_to_it": 1.0745755426606491,
        "bus_to_cooling": 1.0309278350515465,
    },
    ("mv_coupled", "natural_gas"): {
        "grid_to_bus": 1.1301851739601863,
        "bus_to_it": 1.0745755426606491,
        "bus_to_cooling": 1.0309278350515465,
    },
}


@pytest.mark.parametrize("topology,architecture", list(GOLDEN_MULTIPLIERS))
def test_multiplier_goldens(cfg, topology, architecture):
    pfa = PowerFlowAnalyzer(cfg, topology=topology)
    mult = pfa.get_bus_architecture_multipliers(architecture)
    golden = GOLDEN_MULTIPLIERS[(topology, architecture)]
    assert set(mult) == set(golden)
    for key, value in golden.items():
        assert mult[key] == pytest.approx(value, rel=1e-12), key


@pytest.mark.parametrize("topology,architecture", list(GOLDEN_MULTIPLIERS))
def test_multipliers_exceed_unity(cfg, topology, architecture):
    # Multipliers are 1/efficiency: every path has losses, so all > 1.
    pfa = PowerFlowAnalyzer(cfg, topology=topology)
    for key, value in pfa.get_bus_architecture_multipliers(architecture).items():
        assert value > 1.0, key


@pytest.mark.parametrize("architecture", ["ac_coupled", "dc_coupled"])
def test_lv_direct_charge_path_composition(cfg, architecture):
    # lv_direct charge path is defined as the composition of the two-step
    # path: solar->bus then bus->battery (the bus->battery helpers are
    # private composition stages, no longer exposed in the multiplier dict).
    pfa = PowerFlowAnalyzer(cfg, topology="lv_direct")
    mult = pfa.get_bus_architecture_multipliers(architecture)
    bus_to_battery_eff = (pfa._ac_bus_to_battery() if architecture == "ac_coupled"
                          else pfa._dc_bus_to_battery())
    assert mult["solar_to_battery"] == pytest.approx(
        mult["solar_to_bus"] / bus_to_battery_eff, rel=1e-12
    )


def test_unknown_topology_and_architecture_raise(cfg):
    with pytest.raises(ValueError):
        PowerFlowAnalyzer(cfg, topology="hv_direct")
    with pytest.raises(ValueError):
        PowerFlowAnalyzer(cfg).get_bus_architecture_multipliers("fusion")


# ═══════════════════════════════════════════════════════════════════════════
# Natural gas: EUE, derating, part load, combined cycle
# ═══════════════════════════════════════════════════════════════════════════

def _make_plant(n_units=2, unit_mw=200.0, model="GE 7F.04"):
    from natgas_system_tool import PlantConfiguration
    return PlantConfiguration(
        turbine_model=model, turbine_class="f_class", n_units=n_units,
        cycle_type="SC", unit_capacity_mw=unit_mw,
        total_capacity_mw=n_units * unit_mw, efficiency=0.375,
        availability=0.93, capex_per_kw=1319.0, fixed_om_per_kw_yr=25.7,
        var_om_per_mwh=6.94, nrel_reference="f_class_sc", scaling_factors={},
    )


def _forced_availability(cfg, turbine_class="f_class", availability=0.93):
    maint_hours = cfg.design.gas_maintenance_hours[turbine_class]
    pof = maint_hours / 8760.0
    efor = max(0.0, 1.0 - availability - pof)
    return 1.0 - efor


def test_eue_forced_two_unit_hand_calc(cfg):
    # 2x200 MW, demand 300 MW: k=0 short 300 MW, k=1 short 100 MW.
    from natgas_system_tool import calculate_eue_forced
    a = _forced_availability(cfg)
    expected = ((1 - a) ** 2 * 300 + 2 * a * (1 - a) * 100) * 8760
    assert calculate_eue_forced(_make_plant(), 300.0, cfg) == pytest.approx(expected)


def test_eue_forced_single_unit_needed(cfg):
    # Demand below one unit: only the all-units-down state is short.
    from natgas_system_tool import calculate_eue_forced
    a = _forced_availability(cfg)
    expected = (1 - a) ** 3 * 150 * 8760
    assert calculate_eue_forced(_make_plant(n_units=3), 150.0, cfg) == pytest.approx(expected)


def test_eue_planned_spare_capacity(cfg):
    from natgas_system_tool import calculate_eue_planned
    maint = cfg.design.gas_maintenance_hours["f_class"]
    # No spare unit: each staggered outage drops one unit's worth of supply.
    assert calculate_eue_planned(_make_plant(n_units=2), 300.0, cfg) == pytest.approx(200 * maint * 2)
    # Spare unit available: maintenance causes no shortfall.
    assert calculate_eue_planned(_make_plant(n_units=3), 300.0, cfg) == 0.0


def test_temperature_derating(cfg):
    from degradation_model import get_temperature_derating
    assert get_temperature_derating("f_class", 15.0, cfg) == 1.0
    assert get_temperature_derating("f_class", 10.0, cfg) == 1.0
    rate = cfg.gas_turbine.temp_derating_per_c_f_class
    assert get_temperature_derating("f_class", 35.0, cfg) == pytest.approx(1 - rate * 20)
    assert get_temperature_derating("f_class", 200.0, cfg) == 0.7  # floor


def test_part_load_efficiency(cfg):
    from natgas_system_tool import calculate_part_load_efficiency
    assert calculate_part_load_efficiency("f_class", 1.0, cfg) == pytest.approx(1.0)
    penalty = cfg.gas_turbine.part_load_penalty_f_class
    assert calculate_part_load_efficiency("f_class", 0.5, cfg) == pytest.approx(1 - penalty * 0.5)
    # Load factor clamps at 0.3
    assert calculate_part_load_efficiency("f_class", 0.1, cfg) == \
        calculate_part_load_efficiency("f_class", 0.3, cfg)


def test_calculate_cc_capacity_hand_calc(cfg):
    # GT 100 MW @ 40%: fuel 250 MW, exhaust 150 MW, recovered 120 MW (80%),
    # steam 40.8 MW (34%) -> 140.8 MW combined.
    from natgas_system_tool import calculate_cc_capacity
    assert calculate_cc_capacity(100.0, 0.40, cfg) == pytest.approx(140.8)


def test_engineering_filter_load_factor_bounds(cfg):
    from natgas_system_tool import passes_engineering_filter, FilterConfig
    fc = FilterConfig()
    plant = _make_plant(n_units=2)  # 400 MW total
    assert passes_engineering_filter(plant, 200.0, fc, 15.0)        # beta 0.50
    assert not passes_engineering_filter(plant, 50.0, fc, 15.0)     # beta 0.125 < 0.30
    assert not passes_engineering_filter(plant, 390.0, fc, 15.0)    # beta 0.975 > 0.90


# ═══════════════════════════════════════════════════════════════════════════
# Financial: NPV conventions, idling, grid baseline
# ═══════════════════════════════════════════════════════════════════════════

def test_calculate_npv_midyear():
    from lcoe_calc import calculate_npv
    r = 0.07
    assert calculate_npv([100.0], r) == pytest.approx(100 / 1.07 ** 0.5)
    assert calculate_npv([0.0, 100.0], r) == pytest.approx(100 / 1.07 ** 1.5)


def test_gpu_idling_midyear_discounting(cfg):
    from lcoe_calc import calculate_gpu_idling_costs
    r = cfg.financial.discount_rate
    annual = 1000 * cfg.costs.gpu_hour_spot_price * 8760
    expected = annual / (1 + r) ** 0.5 + annual / (1 + r) ** 1.5
    assert calculate_gpu_idling_costs(1000, 2.0, cfg) == pytest.approx(expected)
    # Fractional construction: 2.5 years adds half a year's cost at year 2.5's slot
    expected += 0.5 * annual / (1 + r) ** 2.5
    assert calculate_gpu_idling_costs(1000, 2.5, cfg) == pytest.approx(expected)


def test_grid_baseline_energy_not_double_derated(cfg):
    # annual_energy_mwh arrives already uptime-derated from it_facil; the grid
    # baseline must not apply the uptime fraction a second time.
    from lcoe_calc import calculate_grid_baseline_lcoe, calculate_npv, GRID_DEFAULT
    annual_energy_mwh = 100_000.0
    res = calculate_grid_baseline_lcoe(annual_energy_mwh, 99.0, cfg, location=None)
    _, interconnect_years, price_cents, _ = GRID_DEFAULT
    ops_start = int(np.ceil(interconnect_years))
    flows = [0.0] * cfg.financial.evaluation_years
    for y in range(ops_start, cfg.financial.evaluation_years):
        flows[y] = annual_energy_mwh
    assert res.energy_npv == pytest.approx(calculate_npv(flows, cfg.financial.discount_rate))
    assert res.lcoe == pytest.approx(price_cents / 100.0)
    # Cost and energy flows must correspond: opex / energy = price ($/MWh)
    assert res.opex_npv / res.energy_npv == pytest.approx(price_cents * 10.0)


# ═══════════════════════════════════════════════════════════════════════════
# Degradation: interpolation anchors, solar factor, fade surrogate
# ═══════════════════════════════════════════════════════════════════════════

def _make_sim(load_served_mwh):
    from pvstoragesim import SimulationResult
    return SimulationResult(
        uptime_pct=100.0, energy_served_pct=100.0, solar_generation_mwh=0.0,
        solar_curtailed_mwh=0.0, load_served_mwh=load_served_mwh,
        battery_charged_mwh=0.0, battery_discharged_mwh=0.0,
        battery_cycles_per_year=0.0, unmet_load_mwh=0.0,
    )


def test_interpolate_annual_energy_anchor_placement():
    from degradation_model import interpolate_annual_energy
    e0, e13, e14, e25 = 1000.0, 900.0, 950.0, 800.0
    thermal_kwh = 50_000.0  # 50 MWh, subtracted from every anchor
    energy = interpolate_annual_energy(
        _make_sim(e0), _make_sim(e13), _make_sim(e14), _make_sim(e25),
        construction_years=2.0,
        year_0_stats={"thermal_parasitic_kwh": thermal_kwh},
    )
    tp = thermal_kwh / 1000
    assert energy.shape == (27,)
    assert energy[0] == 0.0 and energy[1] == 0.0          # construction
    assert energy[2] == pytest.approx(e0 - tp)            # ops year 0
    assert energy[14] == pytest.approx(e13 - tp)          # ops year 12 (pre-replacement)
    assert energy[15] == pytest.approx(e14 - tp)          # ops year 13 (post-replacement)
    assert energy[26] == pytest.approx(e25 - tp)          # ops year 24
    # Linear interpolation between first two anchors
    frac = (8 - 2) / (14 - 2)
    assert energy[8] == pytest.approx((e0 - tp) * (1 - frac) + (e13 - tp) * frac)


def test_solar_capacity_factor(cfg):
    from degradation_model import get_solar_capacity_factor
    first = cfg.degradation.solar_first_year
    annual = cfg.degradation.solar_annual
    assert get_solar_capacity_factor(0, cfg) == 1.0
    assert get_solar_capacity_factor(1, cfg) == pytest.approx(1 - first)
    assert get_solar_capacity_factor(5, cfg) == pytest.approx((1 - first) * (1 - annual) ** 4)


def test_fade_surrogate_loads_and_capacity_decreases(cfg):
    # Also serves as the sklearn pickle-compatibility canary.
    from degradation_model import get_battery_capacity_at_year
    cfg.degradation.fade_model_path = str(REPO_ROOT / "output_tables" / "fade_surrogate.pkl")
    stats = {"mean_soc": 0.60, "mean_soc0_pct": 60.0, "efc0": 250.0, "mean_temperature": 25.0}
    caps = [get_battery_capacity_at_year(y, stats, cfg) for y in range(0, 14)]
    assert caps[0] == 1.0
    assert all(c2 < c1 for c1, c2 in zip(caps, caps[1:]))
    assert 0.5 < caps[13] < 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Battery dispatch
# ═══════════════════════════════════════════════════════════════════════════

DISPATCH_MULTS = dict(solar_to_bus_mult=1.25, solar_to_battery_mult=1.30,
                      battery_to_bus_mult=1.20)


def test_dispatch_power_balance_identities():
    from pvstoragesim import simulate_battery_operation
    solar_dc = np.array([0.0, 50.0, 100.0, 20.0, 0.0, 0.0])
    bus_load = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 5.0])
    out = simulate_battery_operation(
        solar_dc_mw=solar_dc, battery_power_mw=20.0, battery_energy_mwh=40.0,
        hourly_bus_load_mw=bus_load, initial_soc=50.0, **DISPATCH_MULTS,
    )
    soc = out["battery_soc"]
    charge, discharge = out["battery_charge_mw"], out["battery_discharge_mw"]
    curtailed, unmet = out["curtailed_solar_mw"], out["unmet_load_mw"]
    stb = DISPATCH_MULTS["solar_to_bus_mult"]
    stbat = DISPATCH_MULTS["solar_to_battery_mult"]
    btb = DISPATCH_MULTS["battery_to_bus_mult"]

    assert np.all(soc >= -1e-9) and np.all(soc <= 40.0 + 1e-9)
    for t in range(len(solar_dc)):
        surplus = solar_dc[t] / stb - bus_load[t] > 0
        if surplus:
            # PV splits exactly into load-serving DC, charging DC, curtailment
            assert unmet[t] == 0.0
            assert solar_dc[t] == pytest.approx(
                bus_load[t] * stb + charge[t] * stbat + curtailed[t])
        else:
            assert charge[t] == 0.0 and curtailed[t] == 0.0
            deficit = bus_load[t] - solar_dc[t] / stb
            assert unmet[t] == pytest.approx(deficit - discharge[t] / btb)
        if t + 1 < len(soc):
            assert soc[t + 1] - soc[t] == pytest.approx(charge[t] - discharge[t])


def test_dispatch_oversized_system_full_uptime():
    from pvstoragesim import simulate_battery_operation
    rng = np.random.default_rng(7)
    solar_dc = np.tile(np.array([0, 0, 0, 0, 0, 0, 10, 40, 80, 100, 110, 115,
                                 115, 110, 100, 80, 40, 10, 0, 0, 0, 0, 0, 0],
                                dtype=float), 10)
    bus_load = 10.0 + rng.random(len(solar_dc))
    out = simulate_battery_operation(
        solar_dc_mw=solar_dc, battery_power_mw=50.0, battery_energy_mwh=400.0,
        hourly_bus_load_mw=bus_load, initial_soc=75.0, **DISPATCH_MULTS,
    )
    assert np.all(out["unmet_load_mw"] < 1e-9)


def test_dispatch_inverter_cap_recaptures_clipped_pv():
    # mv_coupled AC: solar->load path capped at the inverter rating, but the
    # DC-coupled battery still charges from PV above the cap.
    from pvstoragesim import simulate_battery_operation
    solar_dc = np.array([100.0])
    bus_load = np.array([10.0])
    out = simulate_battery_operation(
        solar_dc_mw=solar_dc, battery_power_mw=20.0, battery_energy_mwh=80.0,
        hourly_bus_load_mw=bus_load, inverter_cap_mw_dc=60.0, initial_soc=0.0,
        **DISPATCH_MULTS,
    )
    stb = DISPATCH_MULTS["solar_to_bus_mult"]
    stbat = DISPATCH_MULTS["solar_to_battery_mult"]
    assert out["solar_at_load_mw"][0] == pytest.approx(60.0 / stb)   # capped
    assert out["battery_charge_mw"][0] == pytest.approx(20.0)        # full rate
    leftover_dc = 100.0 - bus_load[0] * stb                          # uncapped PV
    assert out["curtailed_solar_mw"][0] == pytest.approx(leftover_dc - 20.0 * stbat)


def test_dispatch_rejects_mismatched_lengths():
    from pvstoragesim import simulate_battery_operation
    with pytest.raises(ValueError):
        simulate_battery_operation(
            solar_dc_mw=np.zeros(10), battery_power_mw=1.0, battery_energy_mwh=4.0,
            hourly_bus_load_mw=np.zeros(9), **DISPATCH_MULTS,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Topology defaults and clipping consolidation (Phase 3, 2026-06-11)
# ═══════════════════════════════════════════════════════════════════════════

def test_mv_coupled_is_default_everywhere(cfg):
    import inspect
    from pvstoragesim import evaluate_system, get_solar_generation
    from microgrid_optimizer import MicrogridOptimizer, PowerSystemOptimizer
    from lcoe_calc import compare_datacenter_power_systems

    assert PowerFlowAnalyzer(cfg).topology == "mv_coupled"
    for func, name in [
        (evaluate_system, "evaluate_system"),
        (MicrogridOptimizer.__init__, "MicrogridOptimizer"),
        (PowerSystemOptimizer.optimize_solar_storage, "optimize_solar_storage"),
        (compare_datacenter_power_systems, "compare_datacenter_power_systems"),
    ]:
        assert inspect.signature(func).parameters["topology"].default == "mv_coupled", name

    # Threshold-based selection is gone: topology is an explicit flag.
    assert not hasattr(PowerFlowAnalyzer, "select_topology")
    assert not hasattr(cfg.design, "topology_threshold_mw")

    # get_solar_generation is architecture-agnostic; clipping lives in
    # evaluate_system.
    params = inspect.signature(get_solar_generation).parameters
    assert "architecture" not in params and "topology" not in params


def _synthetic_inputs(cfg, solar_capacity_mw=5.0):
    import pandas as pd
    from it_facil import calculate_facility_load
    facility = calculate_facility_load(total_gpus=80, config=cfg,
                                       use_hourly_load_csv=False)
    day = np.zeros(24)
    day[10:14] = 1.0   # above the inverter cap (1/ILR = 0.833 of nameplate)
    day[[8, 9, 14, 15]] = 0.6
    p_dc = np.tile(day, 365)
    profile = pd.DataFrame({"hour": range(8760), "p_dc": p_dc})
    return facility, profile, p_dc


@pytest.mark.parametrize("architecture,topology,clipped", [
    ("ac_coupled", "lv_direct", True),    # battery behind inverter: PV clipped
    ("ac_coupled", "mv_coupled", False),  # DC-coupled storage recaptures
    ("dc_coupled", "lv_direct", False),   # no inverter on the PV path
    ("dc_coupled", "mv_coupled", False),
])
def test_evaluate_system_inverter_clipping(cfg, architecture, topology, clipped):
    from pvstoragesim import evaluate_system
    solar_mw = 5.0
    facility, profile, p_dc = _synthetic_inputs(cfg, solar_mw)
    result = evaluate_system(
        0.0, 0.0, solar_mw, battery_power_mw=2.0, facility_load=facility,
        architecture=architecture, topology=topology, efficiency_params=cfg,
        solar_profile=profile,
    )
    cap_mw = solar_mw / cfg.efficiency.inverter_load_ratio
    expected_gen = np.minimum(p_dc * solar_mw, cap_mw).sum() if clipped \
        else (p_dc * solar_mw).sum()
    assert result.solar_generation_mwh == pytest.approx(expected_gen)
    assert result.uptime_pct == 100.0  # heavily overbuilt for the tiny load


# ═══════════════════════════════════════════════════════════════════════════
# Config round-trip
# ═══════════════════════════════════════════════════════════════════════════

def test_config_save_load_roundtrip(tmp_path):
    cfg = load_config()
    path = tmp_path / "config.json"
    save_config(cfg, str(path))
    cfg2 = load_config(str(path))

    for section in ("costs", "efficiency", "it_load", "design", "financial",
                    "gas_turbine", "wind_turbine"):
        assert getattr(cfg, section).__dict__ == getattr(cfg2, section).__dict__, section

    # Known JSON round-trip artifact (pinned): tuple values in
    # gas_degradation_rates come back as lists. Consumers unpack positionally,
    # so this is safe — but if it changes, downstream assumptions should be
    # rechecked.
    d1, d2 = cfg.degradation.__dict__, cfg2.degradation.__dict__
    assert set(d1) == set(d2)
    for key, value in d1.items():
        if key == "gas_degradation_rates":
            assert {k: list(v) for k, v in value.items()} == \
                   {k: list(v) for k, v in d2[key].items()}
            assert all(isinstance(v, list) for v in d2[key].values())
        else:
            assert value == d2[key], key
