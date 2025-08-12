import os
import glob
import xarray as xr
import numpy as np
import logging
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from itertools import product
from tqdm.auto import tqdm

def merge_ISIMIP_datasets(src_isimip, models, scenarios, variables, overwrite=False, verbose=False, log_file='merge_isimip.log'):
    """
    Merges ISIMIP datasets for specified models, scenarios, and climate variables.

    Parameters:
        src_isimip (str): Source directory containing ISIMIP data to be merged.
        models (list): List of model names.
        scenarios (list): List of scenario names.
        variables (list): List of variable names to merge.
        overwrite (bool): Whether to overwrite existing merged files.
        verbose (bool): Whether to print progress messages to console.
        log_file (str): Path to log file.

    Returns:
        dict: Summary of merged files.
    """

    # Set up logging
    logger = logging.getLogger('ISIMIP_Merger')
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers on repeated calls
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO if verbose else logging.WARNING)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers
        logger.addHandler(fh)
        logger.addHandler(ch)

    summary = {}

    for model in models:
        for scenario in scenarios:
            for variable in variables:
                key = (model, scenario, variable)
                files = sorted(glob.glob(f'{src_isimip}/{model}/{scenario}/{model}*_{variable}_*.nc'))
                summary[key] = {'files_found': len(files), 'merged': False}

                if not files:
                    logger.warning(f'No files found for {key}')
                    continue

                output_file = f'{src_isimip}/{model}/{scenario}/{model}_{variable}_merged.nc'

                if os.path.exists(output_file) and not overwrite:
                    logger.info(f'Skipping existing file: {output_file}')
                    continue

                try:
                    logger.info(f'Merging {len(files)} files for {key}')
                    with xr.open_mfdataset(files, combine='by_coords', parallel=True) as ds:
                        ds.to_netcdf(output_file)

                    summary[key]['merged'] = True
                    logger.info(f'Successfully saved: {output_file}')

                except Exception as e:
                    summary[key]['error'] = str(e)
                    logger.error(f'Error processing {key}: {e}')

    return summary


#=========================================================================================================================
"""
BIAS-CORRECTION_SCRIPTS
"""
#=========================================================================================================================
#EMPIRICAL QUANTILE MAPPING (EQM) for bias correction of precipitation-like variables

#=========================================================================================================================
#This script implements a two-pass EQM bias correction method with zero-jittering for precipitation-like variables.
#It is designed to work with xarray DataArrays and can be applied to historical and future climate model outputs.
#The EQM method adjusts the distribution of model outputs to match observed data, improving the realism of climate projections.
#=========================================================================================================================

# ── Helpers (zero-only jitter + EQM) ─────────────────────────────────────────

def jitter_zeros(arr, jitter=1e-3):
    """
    Add small random noise to zero values only, then clip negatives to 0.

    Why: Precipitation series often have many zeros. Without jitter, the model CDF
    has flat steps at 0, which can cause unstable/degenerate quantile mapping.
    Jitter keeps zeros near zero while making the CDF strictly increasing.

    Parameters
    ----------
    arr : np.ndarray
        Input array (NaNs allowed).
    jitter : float
        Max absolute noise added to zeros (uniform in [-jitter, +jitter]).

    Returns
    -------
    np.ndarray (float64)
        Array with zero entries slightly perturbed, clipped to non-negative.
    """
    out = arr.astype(np.float64, copy=True)   # ensure float for NaNs + noise
    zeros = (out == 0)
    if np.any(zeros):
        noise = np.random.uniform(-jitter, jitter, size=out.shape)
        out[zeros] += noise[zeros]
        out[out < 0] = 0.0  # ensure non-negative (precip)
    return out


def compute_eqm_mapping(o, h, n_q=51, min_samples=10):
    """
    Build the empirical quantile mapping for one cell-month.

    We compute quantiles for the observed series (qo) and for the historical
    model series (qh, after jittering zeros). Later, for any x we find an
    adjustment factor af by interpolating the ratio qo/qh at the x value.

    Parameters
    ----------
    o : np.ndarray
        Observed values for a single month & cell (1-D: time).
    h : np.ndarray
        Historical modeled values for same month & cell (1-D: time).
    n_q : int
        Number of quantiles (e.g., 51 → 0, 0.02, …, 1.0).
    min_samples : int
        Minimum #valid (non-NaN) pairs to calibrate; otherwise return None.

    Returns
    -------
    tuple | None
        (qh, qo) quantile arrays (same shape), or None if insufficient data.
    """
    valid = (~np.isnan(o)) & (~np.isnan(h))
    if np.count_nonzero(valid) < min_samples:
        return None

    probs = np.linspace(0, 1, n_q)           # common probs grid
    h_j = jitter_zeros(h)                     # jitter only the model side
    qh = np.quantile(h_j[valid], probs)       # model quantiles
    qo = np.quantile(o [valid], probs)        # observed quantiles
    return qh, qo


def apply_eqm_vec(x, qh, qo):
    """
    Apply ratio EQM to a 1-D time series x using precomputed (qh, qo).

    Steps:
      1) Build multiplicative factors af = qo/qh (where qh>0; else use 1).
      2) Interpolate af as a function of x over the support defined by qh.
      3) Scale x by the interpolated factor; keep structural zeros as zeros.

    Parameters
    ----------
    x : np.ndarray
        Series to correct (1-D).
    qh : np.ndarray
        Historical model quantiles (from calibration).
    qo : np.ndarray
        Observed quantiles (from calibration).

    Returns
    -------
    np.ndarray
        Corrected series, with zeros preserved.
    """
    # Initialize multiplicative factors as ones; fill positive qh with ratios
    af = np.ones_like(qh, dtype=np.float64)
    nz = (qh > 0)
    af[nz] = qo[nz] / qh[nz]

    # Interpolate factor in x-space; clamp outside to edge factors
    corr = np.interp(x, qh, af, left=af[0], right=af[-1])

    # Apply correction; preserve structural zeros explicitly
    y = (x * corr).astype(np.float64, copy=False)
    y[x == 0] = 0.0
    return y


# ── Main two-pass function ───────────────────────────────────────────────────

def empirical_quantile_mapping(obs, hist, fut, n_q=51, min_samples=10):
    """
    Bias-correct historical and future modeled data using ratio-based EQM.

    Phase A: TRAIN
      - Align obs & hist on time (inner join).
      - For each calendar month and each (lat, lon) cell:
          * compute (qh, qo) quantiles and store mapping.

    Phase B: APPLY
      - For each month:
          * apply the corresponding cell-level mapping to both hist and fut.

    Parameters
    ----------
    obs : xr.DataArray   [time, lat, lon]
        Observed reference data.
    hist : xr.DataArray  [time, lat, lon]
        Historical model data (same grid as obs).
    fut : xr.DataArray   [time, lat, lon]
        Future model data (same grid as obs/hist) — *must be regridded beforehand*.
    n_q : int
        Number of quantiles used to build the mapping.
    min_samples : int
        Minimum #valid samples (per cell-month) to fit a mapping.

    Returns
    -------
    hist_bc : xr.DataArray (float64)
        Bias-corrected historical series on the original hist grid/time.
    fut_bc  : xr.DataArray (float64)
        Bias-corrected future series on the original fut grid/time.

    Raises
    ------
    ValueError
        If obs/hist/fut are not on the same spatial grid (lat/lon).
    """

    # A1) Align obs & hist on time to ensure paired samples for training
    obs_a, hist_a = xr.align(obs, hist, join="inner")

    # A2) Sanity check: spatial grids must match across all inputs
    if not (np.array_equal(obs.lat, hist.lat) and np.array_equal(obs.lon, hist.lon)):
        raise ValueError("obs and hist must share identical lat/lon grids.")
    if not (np.array_equal(obs.lat, fut.lat) and np.array_equal(obs.lon, fut.lon)):
        raise ValueError("fut must be on the same lat/lon grid as obs/hist (regrid first).")

    # B) Pre-allocate float outputs so NaNs are supported even if inputs were int
    hist_bc = xr.full_like(hist, np.nan, dtype=np.float64)
    fut_bc  = xr.full_like(fut,  np.nan, dtype=np.float64)

    lats, lons = obs.lat.values, obs.lon.values

    # A3) TRAIN: precompute month×cell mappings
    #     Key: (month, i, j) → Value: (qh, qo)
    mappings = {}
    for month in range(1, 13):
        # mask the aligned training time axis by calendar month
        msk = obs_a.time.dt.month == month
        if not msk.any():
            continue  # this dataset has no timestamps for this month

        # Extract month slices as NumPy (eager; fine for medium datasets)
        o_m = obs_a.sel(time=msk).values  # shape: (t_m, nlat, nlon)
        h_m = hist_a.sel(time=msk).values

        # Loop cells; micro-opt: slice rows once to reduce Python overhead
        for i in range(len(lats)):
            o_row = o_m[:, i, :]
            h_row = h_m[:, i, :]
            for j in range(len(lons)):
                res = compute_eqm_mapping(o_row[:, j], h_row[:, j],
                                          n_q=n_q, min_samples=min_samples)
                if res is not None:
                    mappings[(month, i, j)] = res  # (qh, qo)

    # B1) APPLY: use trained mappings month-wise to correct hist and fut
    for month in range(1, 13):
        msk_hist = (hist.time.dt.month == month).values
        msk_fut  = (fut .time.dt.month == month).values
        if not (msk_hist.any() or msk_fut.any()):
            continue

        # Pull raw arrays for speed (note: eager; consider dask-vectorization for big data)
        if msk_hist.any():
            h_all = hist.values[msk_hist, ...]  # (t_hm, nlat, nlon)
        if msk_fut.any():
            f_all = fut.values[msk_fut,  ...]   # (t_fm, nlat, nlon)

        for i in range(len(lats)):
            for j in range(len(lons)):
                key = (month, i, j)
                if key not in mappings:
                    # Insufficient training samples for this cell-month → leave NaNs
                    continue

                qh, qo = mappings[key]

                if msk_hist.any():
                    h_bc = apply_eqm_vec(h_all[:, i, j], qh, qo)
                    hist_bc.values[msk_hist, i, j] = h_bc

                if msk_fut.any():
                    f_bc = apply_eqm_vec(f_all[:, i, j], qh, qo)
                    fut_bc.values[msk_fut,  i, j] = f_bc

    # Nice-to-have: carry over metadata if present
    hist_bc.name = getattr(hist, "name", "hist_bc")
    fut_bc.name  = getattr(fut,  "name",  "fut_bc")
    hist_bc.attrs.update(getattr(hist, "attrs", {}))
    fut_bc.attrs.update(getattr(fut,  "attrs", {}))

    return hist_bc, fut_bc

#==========================================================================================================================
"""
Quantile Delta Mapping (QDM) over a lat×lon grid (xarray) using Joblib parallelism.
Includes:
  • DataFrame‑based QDM helpers (compute CDFs, apply to FUT/HIST)
  • A per‑cell worker that runs QDM for one (lat_j, lon_i)
  • A grid driver that stitches results back into xarray DataArrays

Choose kind="*" for precipitation‑like variables (multiplicative),
and kind="+" for temperature‑like variables (additive).

──────────────────────────────────────────────────────────────────────────────
                             HIGH‑LEVEL WORKFLOW
──────────────────────────────────────────────────────────────────────────────
Inputs
  obs(time,lat,lon), sim_hist(time,lat,lon), sim_future(time,lat,lon)

Train (implicit inside helpers via time intersection)
  For each grid cell (j,i):
    1) Align obs & hist on overlapping time
    2) Build probs ps and quantiles q_obs, q_simh
    3) QDM FUT at ps → corrected FUT series
    4) QM/QDM HIST at ps → corrected HIST series (diagnostics)

Apply
  Paste corrected 1‑D arrays into (lat,lon,time) numpy cubes
  Convert to xr.DataArray → hist_bc_da, fut_bc_da

Notes
  • This version uses pandas per‑cell; for very large grids consider a vectorized
    xarray/dask path to avoid Python overhead.
  • FUT must already be on the same lat/lon grid as OBS/HIST.
  • Extrapolation at the tails uses edge values of ps (0 or 1).
──────────────────────────────────────────────────────────────────────────────
"""
#==========================================================================================================================

def quantile_delta_mapping(
    obs: xr.DataArray,
    sim_hist: xr.DataArray,
    sim_future: xr.DataArray,
    *,
    n_quantiles: int = 251,
    min_valid: int = 10,
    kind: str = "+",  # "+" (additive, e.g., T) or "*" (multiplicative, e.g., P)
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Dask/xarray‑native QDM (no pandas, no per‑cell Python loops).

    Strategy (vectorized):
      1) Align OBS & HIST on overlapping time for training.
      2) Stack space = lat*lon so we operate on (time, space) matrices.
      3) Compute per‑space quantiles:
           q_obs(q,space), q_simh(q,space) on overlap; q_simf(q,space) on FUT.
      4) HIST (QM): p = interp(hist, q_simh→ps); hist_bc = interp(p, ps→q_obs).
      5) FUT (QDM): corr_q = obs_q + (simf_q - simh_q)   [additive]
                          or obs_q * (simf_q / simh_q)   [multiplicative]
                     p = interp(fut, simf_q→ps); fut_bc = interp(p, ps→corr_q).
      6) Unstack back to (lat, lon, time). Works eager or with dask chunks.

    Notes
    -----
    - If a pixel has < min_valid non‑NaN values in the training overlap,
      its outputs are filled with NaNs.
    - FUT must already be on the same (lat, lon) grid.
    - Use `.chunk()` on inputs before calling to control dask performance.
    """
    if kind not in {"+", "*"}:
        raise ValueError("kind must be '+' (additive) or '*' (multiplicative).")

    # 1) Align training pair on overlap (inner join on time)
    obs_aln, hist_aln = xr.align(obs, sim_hist, join="inner")
    # Basic grid sanity
    if not (np.array_equal(obs.lat, sim_hist.lat) and np.array_equal(obs.lon, sim_hist.lon)):
        raise ValueError("obs and sim_hist must share the same lat/lon grid.")
    if not (np.array_equal(obs.lat, sim_future.lat) and np.array_equal(obs.lon, sim_future.lon)):
        raise ValueError("sim_future must share the same lat/lon grid as obs/sim_hist.")

    # 2) Stack space (keeps chunking if dask)
    def _stack_space(da):
        return da.transpose("time", "lat", "lon").stack(space=("lat", "lon"))

    O  = _stack_space(obs_aln)       # (t_train, space)
    Ht = _stack_space(hist_aln)      # (t_train, space)
    F  = _stack_space(sim_future)    # (t_fut,   space)

    # Probability grid (shared across pixels)
    ps = xr.DataArray(np.linspace(0.0, 1.0, n_quantiles), dims=["q"], name="ps")

    # 3) Quantiles per space
    # -----------------------------------------------------------------
    # helper: nan‑aware quantile for 1D arrays
    def _nanquantile_1d(x, q):
        x = x.astype(np.float64, copy=False)
        if np.count_nonzero(~np.isnan(x)) < min_valid:
            return np.full_like(q, np.nan, dtype=np.float64)
        return np.nanquantile(x, q)

    # vectorized over 'space'; core dim is 'time' → output core dim 'q'
    q_obs  = xr.apply_ufunc(
        _nanquantile_1d, O, ps,
        input_core_dims=[["time"], ["q"]],
        output_core_dims=[["q"]],
        vectorize=True, dask="parallelized", output_dtypes=[np.float64]
    )
    q_simh = xr.apply_ufunc(
        _nanquantile_1d, Ht, ps,
        input_core_dims=[["time"], ["q"]],
        output_core_dims=[["q"]],
        vectorize=True, dask="parallelized", output_dtypes=[np.float64]
    )
    q_simf = xr.apply_ufunc(
        _nanquantile_1d, F, ps,
        input_core_dims=[["time"], ["q"]],
        output_core_dims=[["q"]],
        vectorize=True, dask="parallelized", output_dtypes=[np.float64]
    )

    # 4) HIST correction (standard QM)
    # -----------------------------------------------------------------
    # p = interp(hist_val, q_simh → ps)
    def _interp_prob_from_quantiles(x_t, qx_q, ps_q):
        # returns p(t); if qx_q has NaNs (insufficient training), return NaNs
        if np.any(np.isnan(qx_q)):
            return np.full_like(x_t, np.nan, dtype=np.float64)
        return np.interp(x_t, qx_q, ps_q, left=ps_q[0], right=ps_q[-1])

    p_hist = xr.apply_ufunc(
        _interp_prob_from_quantiles, Ht, q_simh, ps,
        input_core_dims=[["time"], ["q"], ["q"]],
        output_core_dims=[["time"]],
        vectorize=True, dask="parallelized", output_dtypes=[np.float64]
    )

    # hist_bc = interp(p, ps → q_obs)
    def _interp_value_from_prob(p_t, ps_q, qo_q):
        if np.any(np.isnan(qo_q)):
            return np.full_like(p_t, np.nan, dtype=np.float64)
        return np.interp(p_t, ps_q, qo_q)

    H_bc = xr.apply_ufunc(
        _interp_value_from_prob, p_hist, ps, q_obs,
        input_core_dims=[["time"], ["q"], ["q"]],
        output_core_dims=[["time"]],
        vectorize=True, dask="parallelized", output_dtypes=[np.float64]
    )

    # 5) FUT correction (QDM)
    # -----------------------------------------------------------------
    # corr_q = obs_q + (simf_q - simh_q)  [additive]
    #        = obs_q * (simf_q / simh_q)  [multiplicative]
    def _corr_quantiles(obs_q, simh_q, simf_q, mode, eps=0.1, rmin=0.2, rmax=10.0): #apply max sCALING factor
        """
        Apply QDM correction to quantiles with a max scaling factor.
        """
        if np.any(np.isnan(obs_q)) or np.any(np.isnan(simh_q)) or np.any(np.isnan(simf_q)):
            return np.full_like(obs_q, np.nan, dtype=np.float64)
        if mode == "+":
            return obs_q + (simf_q - simh_q)
        # multiplicative with floor + caps
        simh_q_safe = np.maximum(simh_q, eps)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = simf_q / simh_q_safe
        ratio = np.clip(ratio, rmin, rmax)
        return obs_q * ratio


    corr_q = xr.apply_ufunc(
        _corr_quantiles, q_obs, q_simh, q_simf, xr.DataArray(kind),
        input_core_dims=[["q"], ["q"], ["q"], []],
        output_core_dims=[["q"]],
        vectorize=True, dask="parallelized", output_dtypes=[np.float64]
    )

    # p for FUT with its own CDF: p = interp(fut_val, q_simf → ps)
    p_fut = xr.apply_ufunc(
        _interp_prob_from_quantiles, F, q_simf, ps,
        input_core_dims=[["time"], ["q"], ["q"]],
        output_core_dims=[["time"]],
        vectorize=True, dask="parallelized", output_dtypes=[np.float64]
    )

    # fut_bc = interp(p_fut, ps → corr_q)
    F_bc = xr.apply_ufunc(
        _interp_value_from_prob, p_fut, ps, corr_q,
        input_core_dims=[["time"], ["q"], ["q"]],
        output_core_dims=[["time"]],
        vectorize=True, dask="parallelized", output_dtypes=[np.float64]
    )

    # 6) Unstack back to (lat, lon, time)
    hist_bc = H_bc.unstack("space").transpose("time", "lat", "lon")
    fut_bc  = F_bc.unstack("space").transpose("time", "lat", "lon")

    # Name/attrs
    hist_bc = hist_bc.assign_coords(time=sim_hist.time).rename("hist_qdm")
    fut_bc  = fut_bc.assign_coords(time=sim_future.time).rename("future_qdm")
    # Optional: carry units
    if "units" in getattr(obs, "attrs", {}):
        hist_bc.attrs.setdefault("units", obs.attrs["units"])
        fut_bc .attrs.setdefault("units",  obs.attrs["units"])

    return hist_bc, fut_bc
#==========================================================================================================================