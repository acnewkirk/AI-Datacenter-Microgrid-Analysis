"""
NSRDB weather data loader using pvlib PSM4 API.
Drop-in replacement for get_pvgis_tmy()[0].
"""

import os
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
from pvlib import iotools
from pvlib.iotools import psm4 as _pvlib_psm4

# NREL renamed to National Lab of the Rockies (NLR) in May 2026 and migrated
# the API from developer.nrel.gov to developer.nlr.gov. pvlib hard-codes the
# old domain as module-level constants; patch them at import time so the
# rest of the codebase (and pvlib itself) hits the working endpoint.
_NLR_API_BASE = "https://developer.nlr.gov/api/nsrdb/v2/solar/"
_pvlib_psm4.NSRDB_API_BASE = _NLR_API_BASE
_pvlib_psm4.PSM4_AGG_URL = urljoin(_NLR_API_BASE, _pvlib_psm4.PSM4_AGG_ENDPOINT)
_pvlib_psm4.PSM4_TMY_URL = urljoin(_NLR_API_BASE, _pvlib_psm4.PSM4_TMY_ENDPOINT)
_pvlib_psm4.PSM4_CON_URL = urljoin(_NLR_API_BASE, _pvlib_psm4.PSM4_CON_ENDPOINT)
_pvlib_psm4.PSM4_FUL_URL = urljoin(_NLR_API_BASE, _pvlib_psm4.PSM4_FUL_ENDPOINT)

# API credentials come from the environment. Request a free key at
# https://developer.nlr.gov/signup/ . Cached locations (output_tables/
# nsrdb_cache) load without a key.
NLR_API_KEY = os.environ.get("NLR_API_KEY", "")
NLR_EMAIL = os.environ.get("NLR_EMAIL", "")

_CACHE_DIR = Path(__file__).resolve().parent.parent / "output_tables" / "nsrdb_cache"


def get_nsrdb_tmy(latitude: float, longitude: float) -> pd.DataFrame:
    """
    Fetch TMY weather data from NSRDB PSM4 via pvlib.

    Returns a DataFrame with columns compatible with the rest of the codebase:
    temp_air, relative_humidity, ghi, dni, dhi, wind_speed.

    PSM4 column mapping (map_variables=True is pvlib default):
        temp_air      <- already named temp_air by pvlib
        temp_dew      -> relative_humidity derived via Magnus formula
        ghi           <- already named ghi
        dni           <- already named dni
        dhi           <- already named dhi
        wind_speed    <- already named wind_speed
    """
    cache_path = _CACHE_DIR / f"nsrdb_tmy_{latitude:.4f}_{longitude:.4f}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    if not NLR_API_KEY or not NLR_EMAIL:
        raise RuntimeError(
            f"No cached NSRDB data for ({latitude:.4f}, {longitude:.4f}) and "
            "NLR_API_KEY / NLR_EMAIL environment variables are not set. "
            "Request a free key at https://developer.nlr.gov/signup/ and set "
            "both variables to fetch new locations."
        )

    df, _ = iotools.get_nsrdb_psm4_tmy(
        latitude=latitude,
        longitude=longitude,
        api_key=NLR_API_KEY,
        email=NLR_EMAIL,
    )

    # Derive relative_humidity from dew point using Magnus formula.
    # RH = 100 * exp(17.625*Td / (243.04+Td)) / exp(17.625*T / (243.04+T))
    T = df["temp_air"].values
    Td = df["temp_dew"].values
    df["relative_humidity"] = 100.0 * (
        np.exp((17.625 * Td) / (243.04 + Td))
        / np.exp((17.625 * T) / (243.04 + T))
    )

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df
