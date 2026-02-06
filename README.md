# AI Datacenter Microgrids Analysis Tool

Comparing levelized cost of electricity (LCOE) for datacenter power supply options — solar+storage, natural gas, and grid — at any US location.

## Overview

AI Microgrids takes a GPU count, geographic coordinates, and required uptime percentage, then models four power supply architectures (AC-coupled solar+storage, DC-coupled solar+storage, natural gas with diesel backup, and utility grid) and returns an LCOE comparison over a 27-year project life.

The model captures location-specific cooling loads and PUE via PVGIS weather data, hour-by-hour solar+battery dispatch with rainflow cycle counting, grey-box battery degradation (Arrhenius scaffold + Gaussian process residuals), gas turbine reliability via binomial expected unserved energy (EUE), and bus-centric power flow accounting through each architecture's conversion stages. All costs are in 2022 USD; defaults are sourced from NREL ATB, EIA, and PNNL.

When you run an analysis, the tool follows this pipeline:

```
Location + GPUs + Required Uptime
  → Weather fetch + cooling system selection → Facility load profile
    → Solar+battery optimization (AC & DC coupled)   ┐
    → Natural gas plant sizing + diesel backup       ├→ 27-year LCOE comparison
    → Grid baseline (state-level price lookup)       ┘
```

## Paper

>  Newkirk, Alex, Daniel Gerber, Erica Fuchs, et al. 2025. "Technoeconomic Analysis of Microgrids for AI Data Centers in the Continental United States." Preprint, Research
         +Square, December 22. [https://doi.org/10.21203/rs.3.rs-8272920/v1](https://doi.org/10.21203/rs.3.rs-8272920/v1)

Methodological details, validation, and results are in the paper. When referencing this codebase please cite the associated manuscript. 

## Requirements & Setup

**Python >= 3.10**

Install dependencies:

```bash
pip install numpy pandas scipy pvlib rainflow tzfpy reverse_geocoder scikit-learn requests
```

Requires internet access at runtime for PVGIS TMY API calls.

## Quick Start

### Command line

```bash
cd src
python analysis_wrapper.py 10000 31.77 -106.46
```

Arguments: GPU count, latitude, longitude. Optional trailing arguments: uptime % (default 99), gas price in $/MMBtu (default: state-level lookup).

```bash
# With uptime and gas price overrides
python analysis_wrapper.py 10000 33.45 -112.07 99.5 4.00

# Using a preset location name
python analysis_wrapper.py 10000 phoenix
```

Available presets: `el_paso`, `phoenix`, `dallas`, `seattle`, `chicago`.

### Python API

```python
from lcoe_calc import compare_datacenter_power_systems

comparison = compare_datacenter_power_systems(
    total_gpus=10_000,
    required_uptime_pct=99.0,
    location=(33.45, -112.07)
)
```

The returned object contains per-system LCOE, capacities, and construction timelines. See `analysis_wrapper.py` for a formatted output example.

## Project Structure

**Configuration**

| File | Role |
|------|------|
| `config.py` | Centralized parameters as nested `@dataclass` hierarchy (costs, efficiencies, degradation rates, financials) |

**Demand side**

| File | Role |
|------|------|
| `it_facil.py` | Calculates hourly IT facility electrical loads from GPU specs and normalized load profiles |
| `pue_tool.py` | Selects optimal cooling system and generates hourly PUE profile from PVGIS weather data |
| `datacenter_analyzer.py` | Coordinates cooling selection with facility load modeling for a given location |

**Supply side — solar+storage**

| File | Role |
|------|------|
| `pvstoragesim.py` | Hour-by-hour solar+battery dispatch simulation with power flow and battery cycle tracking |
| `power_systems_estimator.py` | Bus-centric power flow analysis tracking conversion losses through each architecture |
| `degradation_model.py` | Grey-box battery fade model (Arrhenius + GP residuals), solar degradation, gas turbine derating |
| `microgrid_optimizer.py` | Two-stage solar+battery sizing optimizer (Latin hypercube screening → differential evolution) |

**Supply side — natural gas**

| File | Role |
|------|------|
| `natgas_system_tool.py` | Gas turbine plant configuration, reliability analysis (EUE), and diesel backup sizing |

**LCOE comparison**

| File | Role |
|------|------|
| `lcoe_calc.py` | NPV-based LCOE calculation, GPU idling costs, grid baseline, and cross-system comparison |

**CLI wrapper**

| File | Role |
|------|------|
| `analysis_wrapper.py` | Command-line interface with preset locations and formatted output |

## Data Files

### `output_tables/` — required at runtime

| File | Description |
|------|-------------|
| `lookup_PUE_case{1,2,14,15,16,17}.csv` | PUE lookup tables for 6 cooling system architectures (temperature × humidity → PUE) |
| `hourly_load_data.csv` | Normalized 8760-hour IT load shape (from Inference Demand Model) |
| `fade_surrogate.pkl` | Trained grey-box battery fade model (Arrhenius scaffold + Gaussian process residuals) |

### `proxy_training/` — training data for the fade surrogate (not needed at runtime)

| File | Description |
|------|-------------|
| `degradation_full.csv` / `.parquet` | Source degradation data |
| `training_X/y_cal/cyc.npy` | Preprocessed training arrays |

### Runtime data — fetched from PVGIS API

TMY weather data (temperature, humidity, irradiance) for the specified location, retrieved automatically during analysis.

## Configuration

All parameters are centralized in `config.py` as a nested `@dataclass` hierarchy. Defaults are documented with source citations in the code.

To override via JSON:

```python
from config import load_config, save_config

config = load_config()              # get defaults
save_config(config, "my_config.json")  # export to JSON, edit as needed
config = load_config("my_config.json") # reload with overrides
```

Key configurable categories: capital costs, O&M costs, power conversion efficiencies, IT load specs, battery/solar/turbine degradation rates, financial parameters (discount rate, project lifetime), and gas turbine performance curves.

## License

MIT License
