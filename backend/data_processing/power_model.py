"""
Physical models that turn hourly weather into plant output (MW).

These give the modelled "actual" output when fed reanalysis weather, and a
no-ML reference forecast when fed day-ahead forecast weather. No real plant
data is used anywhere in this project.

PV: fixed-tilt array facing north, tilted at the site latitude.
    Solar position (NREL SPA via pvlib) -> isotropic plane-of-array
    irradiance -> Faiman cell temperature -> PVWatts DC and inverter models.
Wind: generic turbine power curve with a cubic ramp between cut-in and
    rated speed, applied to 100 m wind speed, less farm losses.
"""
import numpy as np
import pandas as pd
import pvlib

from utils.config import PLANT, TIMEZONE


def solar_position(times: pd.Series, lat: float, lon: float) -> pd.DataFrame:
    """Solar position at the middle of each hour.

    Open-Meteo radiation values are means over the preceding hour, so the
    hour ending at `time` is centred 30 minutes earlier.
    """
    idx = pd.DatetimeIndex(times).tz_localize(TIMEZONE) - pd.Timedelta(minutes=30)
    return pvlib.solarposition.get_solarposition(idx, lat, lon)


def pv_output_mw(weather: pd.DataFrame, lat: float, lon: float) -> np.ndarray:
    """PV plant AC output in MW from GHI, DNI, DHI, air temperature and 10 m wind"""
    sp = solar_position(weather["time"], lat, lon)
    ghi = weather["shortwave_radiation"].to_numpy(dtype=float)
    dni = weather["direct_normal_irradiance"].to_numpy(dtype=float)
    dhi = weather["diffuse_radiation"].to_numpy(dtype=float)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=abs(lat), surface_azimuth=0.0,  # facing north
        solar_zenith=sp["apparent_zenith"].to_numpy(),
        solar_azimuth=sp["azimuth"].to_numpy(),
        dni=dni, ghi=ghi, dhi=dhi, model="isotropic",
    )["poa_global"]
    poa = np.nan_to_num(np.asarray(poa, dtype=float), nan=0.0).clip(min=0)

    temp_cell = pvlib.temperature.faiman(
        poa, weather["temperature_2m"].to_numpy(dtype=float),
        weather["wind_speed_10m"].to_numpy(dtype=float),
    )
    ac_capacity = PLANT["pv_ac_capacity_mw"]
    pdc = pvlib.pvsystem.pvwatts_dc(
        poa, temp_cell, ac_capacity * PLANT["pv_dc_ac_ratio"], PLANT["pv_gamma_pdc"]
    )
    pdc = np.asarray(pdc, dtype=float) * (1 - PLANT["pv_system_losses"])

    eta = PLANT["pv_inverter_efficiency"]
    pac = pvlib.inverter.pvwatts(pdc, ac_capacity / eta, eta_inv_nom=eta)
    pac = np.nan_to_num(np.asarray(pac, dtype=float), nan=0.0)
    return pac.clip(0, ac_capacity)


def wind_output_mw(wind_speed_ms) -> np.ndarray:
    """Wind farm output in MW from hub-height (100 m) wind speed in m/s"""
    v = np.nan_to_num(np.asarray(wind_speed_ms, dtype=float), nan=0.0)
    v_in, v_rated, v_out = (PLANT["wind_cut_in_ms"], PLANT["wind_rated_ms"],
                            PLANT["wind_cut_out_ms"])
    frac = (v ** 3 - v_in ** 3) / (v_rated ** 3 - v_in ** 3)
    frac = np.where(v < v_in, 0.0, np.where(v >= v_rated, 1.0, frac))
    frac = np.where(v >= v_out, 0.0, frac)
    return PLANT["wind_capacity_mw"] * (1 - PLANT["wind_farm_losses"]) * frac
