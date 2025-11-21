"""
Power Systems Estimator - Refactored to use config.py
"""
from typing import Dict, Optional
from config import Config, load_config

class PowerFlowAnalyzer:
    """Analyzes power flow with bus-centric architecture"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize with config object"""
        self.config = config or load_config()
        self.params = self.config.efficiency
    
    def _calculate_path_efficiency(self, stages: list[float]) -> float:
        """Calculate total efficiency through a series of conversion stages"""
        total_efficiency = 1.0
        for efficiency in stages:
            total_efficiency *= efficiency
        return total_efficiency
    
    # ============================================================
    # SOURCE TO BUS PATHS
    # ============================================================
    
    def _solar_to_ac_bus(self) -> float:
        """Solar panels to AC bus efficiency"""
        # Path: Solar Panels → Inverter → Switchgear → Cabling → AC Bus
        stages = [         
            self.params.inverter_efficiency,
            self.params.switchgear,
            self.params.cable_efficiency
        ]
        return self._calculate_path_efficiency(stages)
    
    def _solar_to_dc_bus(self) -> float:
        """Solar panels to DC bus efficiency"""
        # Path: Solar Panels → DC-DC Converter (MPPT) → Switchgear → Cabling → DC Bus
        stages = [
            self.params.mppt_efficiency,
            self.params.switchgear,
            self.params.cable_efficiency
        ]
        return self._calculate_path_efficiency(stages)
    
    def _battery_to_ac_bus(self) -> float:
        """Battery to AC bus efficiency (discharge only)"""
        # Path: Battery → Inverter (DC-AC) → Switchgear → Cabling → AC Bus
        stages = [
            self.params.battery_rte**0.5,
            self.params.dc_ac,
            self.params.switchgear,
            self.params.cable_efficiency
        ]
        return self._calculate_path_efficiency(stages)
    
    def _battery_to_dc_bus(self) -> float:
        """Battery to DC bus efficiency (discharge only)"""
        # Path: Battery → DC-DC Converter → Switchgear → Cabling → DC Bus
        stages = [
            self.params.battery_rte**0.5,
            self.params.dc_dc,
            self.params.switchgear,
            self.params.cable_efficiency
        ]
        return self._calculate_path_efficiency(stages)
    
    def _grid_to_ac_bus(self) -> float:
        """Grid to AC bus efficiency"""
        # Path: Grid → Transformer → Switchgear → Cabling → AC Bus -> UPS (modelling cooling loads as critical)
        stages = [
            self.params.transformer,
            self.params.switchgear,
            self.params.cable_efficiency,
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
        
        Keys are structured as '{source}_to_bus' or 'bus_to_{load}'.
        """
        if architecture == "ac_coupled":
            # Effectively the inverter to bus 
            return {
                'solar_to_bus': (1 / self._solar_to_ac_bus()) ,
                'battery_to_bus': 1 / self._battery_to_ac_bus(),  # Discharge path
                'bus_to_it': 1 / self._ac_bus_to_it(),
                'bus_to_cooling': 1 / self._ac_bus_to_cooling(),
                'bus_to_battery': 1 / self._ac_bus_to_battery()  # Charge path
            }
        elif architecture == "dc_coupled":
            # This architecture is islanded and has no grid connection.
            return {
                'solar_to_bus': 1 / self._solar_to_dc_bus(),
                'battery_to_bus': 1 / self._battery_to_dc_bus(),  # Discharge path
                'bus_to_it': 1 / self._dc_bus_to_it(),
                'bus_to_cooling': 1 / self._dc_bus_to_cooling(),
                'bus_to_battery': 1 / self._dc_bus_to_battery()  # Charge path
            }
        elif architecture in ["grid", "natural_gas"]:  # These architectures rely on an external source.
            return {
                'grid_to_bus': 1 / self._grid_to_ac_bus(), # NG plant or Grid connects here
                'bus_to_it': 1 / self._ac_bus_to_it(),
                'bus_to_cooling': 1 / self._ac_bus_to_cooling()
            }
        else:
            raise ValueError(f"Unknown architecture: {architecture}")


