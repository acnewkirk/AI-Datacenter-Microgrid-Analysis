# AI Datacenter Microgrids Analysis Tool

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17704494.svg)](https://doi.org/10.5281/zenodo.17704494)

Comparing levelized cost of electricity (LCOE) for datacenter power supply options — solar+storage, natural gas, and grid — at any US location.

## Overview

AI Microgrids takes a GPU count, geographic coordinates, and required uptime percentage, then models four power supply architectures (AC-coupled solar+storage, DC-coupled solar+storage, natural gas with diesel backup, and utility grid) and returns an LCOE comparison over a 27-year project life.

The model captures location-specific cooling loads and PUE via NSRDB weather data, hour-by-hour solar+battery dispatch with rainflow cycle counting, grey-box battery degradation (Arrhenius scaffold + Gaussian process residuals), gas turbine reliability via binomial expected unserved energy (EUE), and bus-centric power flow accounting through each architecture's conversion stages. The model outputs both a standard LCOE and a speed-adjusted LCOE that adds the opportunity cost of GPU-hours to compare systems of different construction times. All costs are in 2022 USD; with parameter values in config and parameter sourcing available in the manuscript.

When you run an analysis, the tool follows this pipeline:

```
Location + GPUs + Required Uptime
  → Weather fetch + cooling system selection → Facility load profile
    → Solar+battery optimization (AC & DC coupled)   ┐
    → Natural gas plant sizing + diesel backup       ├→ Architecture LCOE comparison
    → Grid baseline (state-level price lookup)       ┘
```

## Paper

>  Newkirk, Alex C., Daniel L. Gerber, Erica R. H. Fuchs, et al. 2026. “Speed-to-Power Dominates Energy Costs for AI Data Centers: Technoeconomic Analysis of Powering AI in the Continental United States.” arXiv:2611.17098. Preprint, arXiv, November 12. https://doi.org/10.48550/arXiv.2611.17098.

Methodological details, validation, and results are all documented in that paper.

## Citing

Please cite the paper above for the methods and results. Each tagged release of this repository is also archived on Zenodo with a version-specific DOI; if you use or modify the code, cite the DOI of the version you used so the exact code can be recovered. The DOI [10.5281/zenodo.17704494](https://doi.org/10.5281/zenodo.17704494) always resolves to the latest version and lists all earlier ones. Citation metadata for both are in [`CITATION.cff`](CITATION.cff), and GitHub's "Cite this repository" button formats them.

## Requirements & Setup

**Python >= 3.11** (the pinned versions below are tested on Python 3.13)

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the versions the test suite passes under. `scikit-learn` in particular must stay on 1.8.x to match `fade_surrogate.pkl`, a fitted estimator built under 1.8.0 and loaded at runtime. The figure-reproduction notebook under `notebooks/` additionally needs the packages in `requirements-figures.txt` (`pip install -r requirements-figures.txt`); `cartopy` and `geopandas` install more reliably from conda-forge.

Run the tests from the repository root with `python -m pytest tests -q`.

Weather data comes from NSRDB (PSM4 TMY) via the NLR (formerly NREL) API, so
the model needs an API key. Request a free key at
https://developer.nlr.gov/signup/ and set two environment variables before your
first run:

```bash
export NLR_API_KEY=your_key
export NLR_EMAIL=you@example.com
```

Each location's weather is fetched once and written to
`output_tables/nsrdb_cache/` (git-ignored); later runs of the same location
read from there and need no key or network. Delete that folder anytime to force
a refetch.

## Quick Start

### Command line

```bash
# From the repository root (required: runtime data paths are root-relative)
python src/analysis_wrapper.py 10000 31.77 -106.46
```

Arguments: GPU count, latitude, longitude. Optional trailing arguments: uptime % (default 99), gas price in $/MMBtu (default: state-level lookup). The PV collection topology defaults to a centralized MV spine (`mv_coupled`); pass `--topology lv` to model the modular low-voltage pod design instead.

```bash
# With uptime and gas price overrides
python src/analysis_wrapper.py 10000 33.45 -112.07 99.5 4.00

# Using a preset location name
python src/analysis_wrapper.py 10000 phoenix

# Modular LV pod topology
python src/analysis_wrapper.py 10000 phoenix --topology lv
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

The returned object contains per-system LCOE and speed-premium LCOE, capacities, and other relevant summary results. See `analysis_wrapper.py` for a formatted output example.

## Project Structure

**Configuration**

| File | Role |
|------|------|
| `config.py` | Centralized parameters as nested `@dataclass` hierarchy (costs, efficiencies, degradation rates, financials) |

**Demand side**

| File | Role |
|------|------|
| `it_facil.py` | Calculates hourly IT facility electrical loads from GPU specs and normalized load profiles |
| `pue_tool.py` | Selects optimal cooling system and generates hourly PUE profile from NSRDB weather data |
| `nsrdb_loader.py` | Fetches and caches NSRDB PSM4 TMY weather data (NLR API) |
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

**Cooling System Cases** — PUE lookup tables are indexed by case number:

| Case | Cooling System |
|------|---|
| 1 | Large adiabatic water chiller |
| 2 | Large water-side economizer water chiller |
| 14 | Immersion/DTC liquid dry cooler air cooled chiller |
| 15 | Immersion/DTC liquid water-side economizer water cooled chiller |
| 16 | Immersion/DTC liquid dry/hybrid cooler air cooled chiller |
| 17 | Mid-size dry cooler air cooled chiller |

### `output_tables/lcoe_results.csv` — paper results (not needed at runtime)

CONUS-wide LCOE results as submitted with the manuscript, one row per H3 hexagon and facility size (24k, 50k, 100k, and 750k GPUs). Read by `notebooks/reproduce_paper_figures.ipynb` to draw the paper's figures.

### `proxy_training/` — training data for the fade surrogate (not needed at runtime)

| File | Description |
|------|-------------|
| `degradation_full.csv` / `.parquet` | Source degradation data |
| `training_X/y_cal/cyc.npy` | Preprocessed training arrays |

### Runtime weather data — fetched from NSRDB (PSM4 TMY) via the NLR API

TMY weather (temperature, humidity, irradiance) for the requested location is
fetched at runtime and cached as parquet under `output_tables/nsrdb_cache/`. See
[Requirements & Setup](#requirements--setup) for the API key and environment
variables.

## Configuration

All parameters are centralized in `config.py` as a nested `@dataclass` hierarchy. Defaults are documented with source citations in the code.

To override via JSON:

```python
from config import load_config, save_config

config = load_config()              # get defaults
save_config(config, "my_config.json")  # export to JSON, edit as needed
config = load_config("my_config.json") # reload with overrides
```

Key configurable categories: capital costs, O&M costs, power conversion efficiencies, IT load specs, solar/turbine degradation rates, financial parameters (discount rate, project lifetime), and gas turbine performance curves.

## License

MIT License
