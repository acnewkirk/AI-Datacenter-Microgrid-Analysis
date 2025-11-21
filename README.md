# Datacenter Power Systems Analysis

A Python toolkit for analyzing and optimizing power systems for large-scale AI datacenters. This codebase compares levelized cost of electricity (LCOE) across multiple power generation options: AC-coupled solar+storage, DC-coupled solar+storage, natural gas with diesel backup, and baseline industrial grid electricity.

## Overview

The analysis accounts for:
- Location-specific weather patterns and cooling efficiency (PUE)
- Component degradation over time (solar panels, batteries, gas turbines)
- Power conversion losses through different architectures
- Construction timelines and GPU idling costs
- Reliability requirements (uptime guarantees)


## Architecture

### Core Workflow

```
1. Location Input (lat/lon) + GPU Count
         ↓
2. Cooling Analysis (pue_tool.py)
   → Selects optimal cooling system
   → Generates hourly PUE profile
         ↓
3. Facility Load Calculation (datacenter_analyzer.py + it_facil.py)
   → Hourly IT load from empirical hardware benchmarking
   → Hourly Cooling load from PUE and IT load
   → Design loads with contingency
         ↓
4. Power System Optimization
   ├─ Solar+Storage (microgrid_optimizer.py)
   │  → AC-coupled: solar → inverter → AC bus → IT/cooling
   │  → DC-coupled: solar → DC bus → IT/cooling (no grid)
   │  → Degradation analysis (4 anchor points: years 0, 13, 14, 25)
   │
   └─ Natural Gas (natgas_system_tool.py)
      → Turbine selection (aeroderivative, F-class, H-class)
      → Reliability analysis (forced outages + planned maintenance)
      → Diesel backup sizing
         ↓
5. LCOE Comparison (lcoe_calc.py)
   → NPV of capital, operating costs, and energy production
   → Include GPU idling costs during construction
   → Grid baseline for comparison
```

### Module Responsibilities

#### Facility Modules

- **`config.py`**: Central configuration management (costs, efficiencies, design parameters)
- **`datacenter_analyzer.py`**: Top-level coordinator that integrates cooling with facility load
- **`it_facil.py`**: IT load calculations from GPU specs, handles hourly load profiles
- **`pue_tool.py`**: Location-specific cooling system selection using PVGIS weather data

#### Power Generation and Storage Modules

- **`pvstoragesim.py`**: Hour-by-hour solar+battery simulation with PVGIS solar profiles
- **`microgrid_optimizer.py`**: Optimization loop for evaluated architectures
- **`natgas_system_tool.py`**: Natural gas plant configuration, reliability analysis, diesel backup sizing
- **`power_systems_estimator.py`**: Bus-centric power flow analysis (tracks conversion losses)

#### System Evaluation Modules

- **`degradation_model.py`**: Component degradation over time
  - Grey-box battery fade model (physics + GP residuals)
  - Solar degradation (first year + annual)
  - Gas turbine degradation (capacity + efficiency)
  - Battery thermal modeling and parasitic thermal load derating
  
- **`lcoe_calc.py`**: Financial analysis and LCOE calculations
  - NPV calculations with proper discounting
  - Construction phasing
  - GPU idling costs
  - Grid baseline comparison

#### Utility Modules

- **`analysis_wrapper.py`**: Simple command-line interface for quick analyses

## Installation

### Prerequisites

```bash
# Python 3.9+ required
python --version

# Install dependencies
pip install numpy pandas scipy pvlib requests tzfpy rainflow reverse_geocoder
```

### Required Data Files

The codebase expects these directories and files:

```
project_root/
├── output_tables/
│   ├── lookup_PUE_case1.csv
│   ├── lookup_PUE_case2.csv
│   ├── lookup_PUE_case14.csv
│   ├── lookup_PUE_case15.csv
│   ├── lookup_PUE_case16.csv
│   ├── lookup_PUE_case17.csv
│   └── hourly_load_data.csv
├── models/
│   └── fade_surrogate.pkl (battery degradation model)
└── src/
    └── [all Python modules]
```

**PUE Lookup Tables**: CSV files with columns `T_oa`, `RH_oa`, `pue` representing cooling system efficiency at different outdoor air temperatures and humidity levels.

**Hourly Load Data**: CSV with columns `date`, `hour`, `it_load_avg`, `it_load_norm` representing normalized IT load shape over 8760 hours.

**Battery Fade Model**: Pickled sklearn model for battery degradation prediction.

## Usage

### Quick Start

```python
from lcoe_calc import compare_datacenter_power_systems

# Compare all power systems for a 10,000 GPU datacenter in Phoenix
comparison = compare_datacenter_power_systems(
    total_gpus=10_000,
    required_uptime_pct=99.0,
    location=(33.45, -112.07),  # Phoenix, AZ
    gas_price=3.50  # $/MMBtu (optional, uses location-specific if None)
)

# View results
from lcoe_calc import print_lcoe_comparison_table
print_lcoe_comparison_table(comparison)
```

### Command Line Interface

```bash
# Quick analysis for specific location
python analysis_wrapper.py 10000 phoenix

# Or use coordinates directly
python analysis_wrapper.py 10000 33.45 -112.07

# Specify uptime requirement and gas price
python analysis_wrapper.py 10000 33.45 -112.07 99.9 4.00
```

### Detailed Analysis

```python
from datacenter_analyzer import DatacenterAnalyzer
from microgrid_optimizer import MicrogridOptimizer, SystemCosts
from config import load_config

# 1. Load configuration
config = load_config()

# 2. Analyze datacenter requirements with location-specific cooling
analyzer = DatacenterAnalyzer(
    latitude=33.45,
    longitude=-112.07,
    total_gpus=10_000
)

# Get facility load with hourly PUE profile
facility_load = analyzer.calculate_facility_load(required_uptime_pct=99.0)

# 3. Optimize solar+storage
costs = SystemCosts(
    solar_cost_per_kw=config.costs.solar_cost_y0,
    battery_cost_per_kw=config.costs.bess_cost_y0,
    solar_bos_cost_per_kw=config.costs.solar_bos_cost_y0_ac,
    battery_bos_cost_per_kw=config.costs.battery_bos_cost_y0_ac
)

optimizer = MicrogridOptimizer(
    latitude=33.45,
    longitude=-112.07,
    facility_load=facility_load,
    required_uptime_pct=99.0,
    costs=costs,
    architecture="ac_coupled"
)

result = optimizer.optimize()

print(f"Optimal system: {result.solar_mw:.1f} MW solar, "
      f"{result.battery_mw:.1f} MW / {result.battery_mwh:.0f} MWh battery")
print(f"Uptime: {result.uptime_pct:.2f}%")
print(f"Cost: ${result.total_cost_million:.1f}M")
```

## Key Algorithms

### 1. Cooling System Selection (pue_tool.py)

For each location:
1. Fetch PVGIS TMY (Typical Meteorological Year) weather data
2. For each hour and each cooling system case:
   - Round temperature/humidity to lookup table resolution
   - Find PUE via nearest neighbor if exact match unavailable
3. Calculate annual average PUE
4. Select cooling system with lowest annual PUE

### 2. Solar+Storage Optimization (microgrid_optimizer.py)

**Two-stage approach**:

**Stage 1: Feasibility Screening**
- Generate ~100 Latin hypercube samples across search space
- Fast evaluation (Year 0 only, no hourly data)
- Filter to feasible designs meeting uptime requirement
- Identify promising region

**Stage 2: Differential Evolution**
- Full degradation analysis (4 anchor years)
- Optimize within feasible region using scipy.optimize.differential_evolution
- Minimize total system capital cost (proven LCOE proxy)
- Progressive bounds expansion if no solution found

**Degradation Analysis**:
- Year 0: Fresh system, extract battery statistics
- Year 13: Pre-battery-replacement (solar + battery degraded)
- Year 14: Post-replacement (solar degraded, fresh battery)
- Year 25: End of life (both degraded)
- Interpolate linearly between anchor points

### 3. Natural Gas System Design (natgas_system_tool.py)

**Configuration Generation**:
1. For each turbine type (aeroderivative, F-class, H-class):
   - Determine valid cycle types (SC/CC)
   - Calculate unit capacity (with CC heat recovery if applicable)
   - Generate configurations with 1 to N units
   
2. **Engineering Pre-filter**:
   - Load factor bounds 
   - Unit count limits by turbine class
   - Temperature derating check
   - Optional N-1 redundancy requirement
   
3. **Reliability Analysis**:
   - Calculate EUE from forced outages (probabilistic)
   - Calculate EUE from planned maintenance (deterministic)
   - Size diesel backup to cover total EUE
   
4. **LCOE Calculation**:
   - Account for degradation (capacity and efficiency)
   - Part-load efficiency curves
   - Diesel testing + outage coverage
   - Construction timeline by turbine class

### 4. Battery Degradation Model (degradation_model.py)

**Grey-box approach**:
- **Calendar fade**: `A × t^β × exp(p₂/T) × (1 + γ|SOC - 0.5|)`
- **Cycle fade**: `B × EFC^α × exp(p₇/T)`
- Plus Gaussian Process residuals for both

Inputs:
- Operating year
- Mean state of charge (%)
- Cumulative equivalent full cycles
- Mean temperature (°C, with thermal model)

**Thermal Model**:
- Tracks battery temperature hour-by-hour
- Heat generation from charge/discharge inefficiency
- Cooling when temp exceeds max threshold
- Heating when temp below min (only when battery idle)

## Configuration

All costs, efficiencies, and design parameters are centralized in `config.py` using dataclasses:

```python
@dataclass
class ITLoadParams:
    gpus_per_node: int = 8
    node_power_avg_kw: float = 10.0
    node_power_max_kw: float = 12.0
    default_pue: float = 1.3
    design_contingency_factor: float = 1.15

@dataclass
class CostParams:
    solar_cost_y0: float = 1000  # $/kW
    bess_cost_y0: float = 300    # $/kWh
    # ... many more
```

To modify costs or assumptions, edit `config.py` or override at runtime:

```python
config = load_config()
config.costs.solar_cost_y0 = 900  # Override solar cost
```

## Data Flow Example

Let's trace a 10,000 GPU datacenter in Phoenix:

1. **Location Input**: (33.45°N, 112.07°W)
   - 20,000 GPUs, 99% uptime

2. **Cooling Analysis**:
   - Fetch Phoenix TMY data from PVGIS
   - Test 6 cooling systems against 8760 hours
   - Select optimal case (PUE = 1.12 annual average, 1.23 design maximum)
   
3. **Facility Load**:
   - 20,000 GPUs ÷ 8 GPUs/node = 2500 nodes
   - 2500 × 7.3 kW average = 18.2 MW IT load on average
   - Add cooling load based on hourly PUE
   - Include Design contingency (5% contingency)
   - All architectures deliver equivalent energy services, but differ in required geenration due to differen power delivery path efficiecy

4. **Solar+Storage Optimization**:
   - Two-stage optimization: (1) feasibility screening, (2) cost minimization
   - Runs full degradation analysis at 4 anchor years (0, 13, 14, 25)
   - AC result: 158 MW solar / 174 MW battery, $441M, 2-year construction
   - DC result: 170 MW solar / 114 MW battery, $370M, 2-year construction

DC power path efficiency allows for a more solar favoring 

5. **Natural Gas**:
   - Generates turbine configurations, filters by engineering constraints (minimum power, load factor, etc)
   - Runs reliability analysis (forced outages + planned maintenance)
   - Selects lowest LCOE option among valid designs
   - Best: 1× GE LM6000PC (51 MW)
   - 
6. **Grid Baseline**:
   - Phoenix: $0.0684/kWh industrial rate (4-year avg)
   - 4-year interconnection timeline
   - LCOE: $0.0684/kWh (pass-through simplification

7. **Final Comparison** Baseline LCOE and Total LCOE:
   -Total LCOE includes the net present value of GPU idling costs during construction
   - Natural Gas: $0.2038/kWh (4 year construction), $1.173/kWh total
   - Grid: $0.0684/kWh (4 year interconnection), $1.0481/kWh total
   - AC Solar: $0.2237/kWh (2 year construction), $0.6616/kWh total
   - DC Solar: $0.1732/kWh (2 year construction),  $0.6118/kWh total

## Output Interpretation

### LCOE Comparison Table

```
LCOE COMPARISON RESULTS
================================================================================
Location: (33.45, -112.07)
State: Arizona
Natural Gas Price: $6.29/MMBtu
IT Facility Design Load: 27 MW
Required Uptime: 99.0%

SYSTEM CAPACITIES:
AC Solar+Storage: 158 MW solar / 174 MW (695 MWh) battery - 3.9sq km
DC Solar+Storage: 170 MW solar / 114 MW (456 MWh) battery - 3.9sq km
Natural Gas: 51 MW - 1× GE LM6000PC SC

LCOE COMPARISON TABLE:
---
System               Construction Base LCOE       Idling Cost     Total LCOE
Type                 Years        ($/kWh)         ($M NPV)        ($/kWh)
--------------------------------------------------------------------------------
AC Solar+Storage     2.00         $0.2237         $813.5          $0.6616
DC Solar+Storage     2.00         $0.1732         $813.5          $0.6118
Natural Gas          4.00         $0.2038         $1524.0         $1.1730
Grid                 4.00         $0.0684         $1524.0         $1.0481
```

**Base LCOE**: Capital + operating costs ÷ energy produced (NPV basis)

**Idling Cost**: Cost of GPUs sitting idle during construction/interconnection
- Calculated as: `NPV of ( GPUs × spot_price × hours )`


**Total LCOE**: Base LCOE + idling cost amortized over energy produced






