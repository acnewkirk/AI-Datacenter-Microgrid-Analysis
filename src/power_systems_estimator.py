"""
Power Systems Estimator

Two independent axes describe a PV+BESS design:

  architecture - data-hall distribution bus type: 'ac_coupled' (480 V AC to
                 the IT load) or 'dc_coupled' (DC distribution).
  topology     - PV collection network: 'mv_coupled' (default) is a
                 centralized PV field with an MV gathering spine to the data
                 hall; 'lv_direct' is one modular pod with solar, battery,
                 and IT on a shared LV bus. A single LV bus is practical to
                 roughly ~2 MW; modular facilities tile pods, so lv_direct
                 per-MWh results scale linearly with facility size.

Where the battery couples (note the trap in the ac_coupled/mv_coupled cell):

    architecture   topology     battery position
    ac_coupled     lv_direct    AC-coupled at the data-hall LV bus
    ac_coupled     mv_coupled   DC-coupled at the PV-field DC link (pre-inverter)
    dc_coupled     lv_direct    on the data-hall LV DC bus (own DC-DC)
    dc_coupled     mv_coupled   on the MV-DC spine

Grid and NG always use the MV-coupled in-facility chain (source -> MV
backbone -> data hall LV); their sources are MV at origin and cannot
modularize. Full network diagrams: Newkirk et al., manuscript Methods.
"""
from typing import Dict, Optional
from config import Config, load_config

class PowerFlowAnalyzer:
    """Analyzes power flow with bus-centric architecture.

    `topology='mv_coupled'` (default) uses the monolithic MV gathering chain
    for PV+BESS solar coupling; `'lv_direct'` uses the modular per-pod chain.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        topology: str = "mv_coupled",
    ):
        if topology not in ("lv_direct", "mv_coupled"):
            raise ValueError(
                f"Unknown topology '{topology}'. Use 'lv_direct' or 'mv_coupled'."
            )
        self.config = config or load_config()
        self.params = self.config.efficiency
        self.topology = topology

    def _calculate_path_efficiency(self, stages: list[float]) -> float:
        """Calculate total efficiency through a series of conversion stages"""
        total_efficiency = 1.0
        for efficiency in stages:
            total_efficiency *= efficiency
        return total_efficiency
    
    # ============================================================
    # SOURCE TO BUS PATHS
    # ============================================================
    
    def _solar_to_ac_bus_lv_direct(self) -> float:
        """Solar -> AC bus, LV-direct (modular pod) topology.

        Path: Solar -> Inverter -> Switchgear -> Cabling -> AC Bus
        """
        stages = [
            self.params.inverter_efficiency,
            self.params.switchgear,
            self.params.cable_efficiency,
        ]
        return self._calculate_path_efficiency(stages)

    def _solar_to_ac_bus_mv_coupled(self) -> float:
        """Solar -> AC bus, MV-coupled (monolithic) topology.

        DC-coupled storage: the PV array feeds a regulated DC link through its
        own MPPT DC-DC (the battery taps the same link pre-inverter), so the
        MPPT is a discrete stage upstream of the inverter — it must appear on
        every PV-origin path, load and charge alike.

        Path: Solar -> MPPT DC-DC -> Inverter -> step-up xfmr -> MV cable
              -> step-down xfmr -> Switchgear -> AC Bus
        """
        stages = [
            self.params.mppt_efficiency,    # PV -> regulated DC link (MPPT)
            self.params.inverter_efficiency,
            self.params.transformer,        # LV -> MV backbone (~34.5 kV)
            self.params.transformer,        # MV backbone -> data hall LV
            self.params.switchgear,
            self.params.cable_efficiency,
        ]
        return self._calculate_path_efficiency(stages)

    def _solar_to_ac_bus(self) -> float:
        """Dispatch solar->AC path by topology."""
        if self.topology == "mv_coupled":
            return self._solar_to_ac_bus_mv_coupled()
        return self._solar_to_ac_bus_lv_direct()

    def _solar_to_dc_bus_lv_direct(self) -> float:
        """Solar -> DC bus, LV-direct (modular pod) topology.

        Path: Solar -> DC-DC (MPPT) -> Switchgear -> Cabling -> DC Bus
        """
        stages = [
            self.params.mppt_efficiency,
            self.params.switchgear,
            self.params.cable_efficiency,
        ]
        return self._calculate_path_efficiency(stages)

    def _solar_to_dc_bus_mv_coupled(self) -> float:
        """Solar -> DC bus, MV-coupled (monolithic) topology.

        Path: Solar -> MPPT -> MVDC step-up -> MVDC cable
              -> MVDC step-down -> Switchgear -> DC Bus
        """
        stages = [
            self.params.mppt_efficiency,
            self.params.mvdc_converter,     # LV DC -> MVDC backbone
            self.params.mvdc_converter,     # MVDC backbone -> data hall LV DC
            self.params.switchgear,
            self.params.cable_efficiency,
        ]
        return self._calculate_path_efficiency(stages)

    def _solar_to_dc_bus(self) -> float:
        """Dispatch solar->DC path by topology."""
        if self.topology == "mv_coupled":
            return self._solar_to_dc_bus_mv_coupled()
        return self._solar_to_dc_bus_lv_direct()
    
    def _battery_to_ac_bus_lv_direct(self) -> float:
        """Battery → AC bus discharge, LV-direct (battery at the data-hall LV bus).

        Path: Battery → PCS (DC-AC) → Switchgear → Cabling → AC bus
        """
        stages = [
            self.params.battery_rte**0.5,
            self.params.dc_ac,
            self.params.switchgear,
            self.params.cable_efficiency,
        ]
        return self._calculate_path_efficiency(stages)

    def _battery_to_ac_bus_mv_coupled(self) -> float:
        """Battery → AC bus discharge, MV-coupled. DC-coupled storage: the
        battery sits on the PV-field DC link via its own DC-DC converter and
        shares the PV inverter, then discharge crosses the full MV spine to the
        data-hall LV bus. The battery DC-DC is the same converter charge pays,
        so it appears on discharge too (symmetric with `_solar_to_battery_ac`).

        Path: Battery → DC-DC → shared inverter (DC-AC) → step-up xfmr
              → step-down xfmr → sw → cab
        """
        stages = [
            self.params.battery_rte**0.5,
            self.params.battery_converter,  # battery → DC link (DC-DC)
            self.params.dc_ac,              # shared inverter (DC link → AC)
            self.params.transformer,        # LV → MV backbone
            self.params.transformer,        # MV backbone → data hall LV
            self.params.switchgear,
            self.params.cable_efficiency,
        ]
        return self._calculate_path_efficiency(stages)

    def _battery_to_ac_bus(self) -> float:
        """Dispatch battery→AC discharge path by topology."""
        if self.topology == "mv_coupled":
            return self._battery_to_ac_bus_mv_coupled()
        return self._battery_to_ac_bus_lv_direct()
    
    def _battery_to_dc_bus_lv_direct(self) -> float:
        """Battery → DC bus discharge, LV-direct (battery at the data-hall LV DC bus).

        Path: Battery → DC-DC → Switchgear → Cabling → DC bus
        """
        stages = [
            self.params.battery_rte**0.5,
            self.params.battery_converter,
            self.params.switchgear,
            self.params.cable_efficiency,
        ]
        return self._calculate_path_efficiency(stages)

    def _battery_to_dc_bus_mv_coupled(self) -> float:
        """Battery → DC bus discharge, MV-coupled. Battery on the MV-DC spine;
        discharge steps down to the data-hall LV DC bus.

        Path: Battery → DC-DC → MVDC step-down → Switchgear → Cabling → DC bus
        """
        stages = [
            self.params.battery_rte**0.5,
            self.params.battery_converter,
            self.params.mvdc_converter,     # MV-DC spine → data hall LV DC
            self.params.switchgear,
            self.params.cable_efficiency,
        ]
        return self._calculate_path_efficiency(stages)

    def _battery_to_dc_bus(self) -> float:
        """Dispatch battery→DC discharge path by topology."""
        if self.topology == "mv_coupled":
            return self._battery_to_dc_bus_mv_coupled()
        return self._battery_to_dc_bus_lv_direct()

    # ------------------------------------------------------------
    # SOLAR → BATTERY CHARGE PATHS (charge drawn upstream of the load bus)
    # ------------------------------------------------------------
    # With the battery given its own network position, charging energy is
    # drawn upstream of the data-hall load bus, not from post-distribution
    # surplus. Dispatch (pvstoragesim) consumes 'solar_to_battery'. The
    # bus→battery methods below are composition helpers for the lv_direct
    # charge paths (lv_direct: solar_to_battery = solar_to_bus ∘ bus_to_battery).

    def _solar_to_battery_dc(self) -> float:
        """Solar → battery charge path, DC architecture.

        MV-coupled: PV charges off the MV-DC spine (one spine crossing).
        LV-direct: composes the legacy two-step path (solar→bus, bus→battery)
        so the restructured dispatch reproduces prior LV charging exactly.
        """
        if self.topology == "mv_coupled":
            # PV → MPPT → MVDC step-up → battery DC-DC → Battery
            stages = [
                self.params.mppt_efficiency,
                self.params.mvdc_converter,
                self.params.battery_converter,
                self.params.battery_rte**0.5,
            ]
            return self._calculate_path_efficiency(stages)
        return self._solar_to_dc_bus_lv_direct() * self._dc_bus_to_battery()

    def _solar_to_battery_ac(self) -> float:
        """Solar → battery charge path, AC architecture.

        MV-coupled: DC-coupled storage on the PV DC bus — charges pre-inverter
        (no inverter, no transformer), so inverter clipping is recoverable in
        dispatch. LV-direct: composes the legacy two-step path.
        """
        if self.topology == "mv_coupled":
            # PV DC bus → MPPT → battery DC-DC → Battery
            stages = [
                self.params.mppt_efficiency,
                self.params.battery_converter,
                self.params.battery_rte**0.5,
            ]
            return self._calculate_path_efficiency(stages)
        return self._solar_to_ac_bus_lv_direct() * self._ac_bus_to_battery()
    
    def _grid_to_ac_bus(self) -> float:
        """Grid or NG source to AC bus efficiency.

        Two transformer stages: source → facility MV backbone (~34.5 kV),
        then MV backbone → data hall LV (~480 V). See module docstring.

        Includes a double-conversion UPS stage. This is deliberately absent
        from the PV+BESS paths: in the islanded architectures the
        battery/inverter system provides ride-through and power conditioning
        inherently, so a separate UPS would be redundant. Grid/NG sources
        need one. (See manuscript Methods.)
        """
        stages = [
            self.params.transformer,        # source → facility MV backbone
            self.params.transformer,        # MV backbone → data hall LV
            self.params.switchgear,
            self.params.cable_efficiency,   # combined MV + LV distribution
            self.params.ups
        ]
        return self._calculate_path_efficiency(stages)
    

    
    # ============================================================
    # BUS TO LOAD PATHS
    # ============================================================
    
    def _ac_bus_to_it(self) -> float:
        """AC bus to IT load efficiency"""
        # Path: AC Bus → PDU → Server PSU (AC-DC) → IT Load
        stages = [
            
            self.params.pdu_efficiency,
            self.params.ac_psu_efficiency
        ]
        return self._calculate_path_efficiency(stages)
    
    def _ac_bus_to_cooling(self) -> float:
        """AC bus to cooling load efficiency"""
        # Path: AC Bus  → VFD → Cooling Motors
        stages = [
            
            self.params.vfd_efficiency
        ]
        return self._calculate_path_efficiency(stages)
    
    def _dc_bus_to_it(self) -> float:
        """DC bus to IT load efficiency"""
        # Path: DC Bus → PDU → Server PSU (DC) → IT Load
        stages = [
            self.params.pdu_efficiency,
            self.params.dc_psu_efficiency
        ]
        return self._calculate_path_efficiency(stages)
    
    def _dc_bus_to_cooling(self) -> float:
        """DC bus to cooling load efficiency"""
        # Path: DC Bus → DC Motor Controller → Cooling Motors
        stages = [
            self.params.dc_motor_controller_efficiency
        ]
        return self._calculate_path_efficiency(stages)
    
    # ============================================================
    # BATTERY CHARGING PATHS (FROM BUS)
    # ============================================================
    
    def _ac_bus_to_battery(self) -> float:
        """AC bus to battery charging efficiency"""
        # Path: AC Bus → Rectifier (AC-DC) → Battery
        stages = [
            self.params.ac_dc,
            self.params.battery_rte**0.5
        ]
        return self._calculate_path_efficiency(stages)
    
    def _dc_bus_to_battery(self) -> float:
        """DC bus to battery charging efficiency"""
        # Path: DC Bus → DC-DC Converter → Battery
        stages = [
            self.params.battery_converter,
            self.params.battery_rte**0.5
        ]
        return self._calculate_path_efficiency(stages)
    
    # ============================================================
    # PUBLIC INTERFACE
    # ============================================================
    
    def get_bus_architecture_multipliers(self, architecture: str) -> Dict[str, float]:
        """
        Get all bus-centric multipliers for a given architecture as a flat dictionary.
        A 'multiplier' is the factor by which a load must be increased to account for
        losses (i.e., multiplier = 1 / efficiency).

        Keys name directed edges in the power-flow graph: '{source}_to_bus',
        'bus_to_{load}', and 'solar_to_battery' (the charge path, which runs
        upstream of the load bus rather than through it).
        """
        if architecture == "ac_coupled":
            # Effectively the inverter to bus 
            return {
                'solar_to_bus': 1 / self._solar_to_ac_bus(),
                'battery_to_bus': 1 / self._battery_to_ac_bus(),  # Discharge path
                'solar_to_battery': 1 / self._solar_to_battery_ac(),  # Charge (upstream of load bus)
                'bus_to_it': 1 / self._ac_bus_to_it(),
                'bus_to_cooling': 1 / self._ac_bus_to_cooling(),
            }
        elif architecture == "dc_coupled":
            # This architecture is islanded and has no grid connection.
            return {
                'solar_to_bus': 1 / self._solar_to_dc_bus(),
                'battery_to_bus': 1 / self._battery_to_dc_bus(),  # Discharge path
                'solar_to_battery': 1 / self._solar_to_battery_dc(),  # Charge (upstream of load bus)
                'bus_to_it': 1 / self._dc_bus_to_it(),
                'bus_to_cooling': 1 / self._dc_bus_to_cooling(),
            }
        elif architecture in ["grid", "natural_gas"]:  # These architectures rely on an external source.
            return {
                'grid_to_bus': 1 / self._grid_to_ac_bus(), # NG plant or Grid connects here
                'bus_to_it': 1 / self._ac_bus_to_it(),
                'bus_to_cooling': 1 / self._ac_bus_to_cooling()
            }
        else:
            raise ValueError(f"Unknown architecture: {architecture}")


