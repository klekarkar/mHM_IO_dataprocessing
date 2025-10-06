#THese scripts are for validating mHM flow and baseflow for seasonal recharge analysis
"""Python py_geospatial environment"""
import sys
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
import process_observed_discharge as mQ
import glob
from sklearn.metrics import r2_score
import baseflow #this is the baseflow package
import sys
from pathlib import Path

#segoe UI
plt.rcParams['font.family'] = 'Segoe UI'



##function to match frequency and resample to desired frequency
def match_frequency_and_resample(q_model: pd.Series, q_obs: pd.Series, obs_freq: str, resample_freq: str, station_id: str):
    """
    Match frequency of model and observed discharge time series.
    Resample both to the specified frequency if they differ.
    Parameters
    ----------
    q_model : pd.Series  
        Modeled discharge time series (indexed by date).  
    q_obs : pd.Series  
        Observed discharge time series (indexed by date).  
    obs_freq : str  
        The frequency of observed data (e.g., 'D' for daily).  
    resample_freq : str  
        Resampling frequency (e.g., 'ME' for month-end).  
    station_id : str  
        Identifier for the station (for logging purposes).  
    Returns
    -------
    pd.Series, pd.Series  
        Resampled model and observed discharge time series.  
    """
    #Check if both series have the same frequency and resample if not
    #Check if both series have the same frequency and resample if not
    #set indices to datetime if not already
    if not isinstance(q_model.index, pd.DatetimeIndex):
        q_model.index = pd.to_datetime(q_model.index)
    if not isinstance(q_obs.index, pd.DatetimeIndex):
        q_obs.index = pd.to_datetime(q_obs.index)

    #Enforce daily frequency for Observed data if not already
    q_obs = q_obs.asfreq(obs_freq)   # enforce daily frequency


    if not (pd.infer_freq(q_model.index) == pd.infer_freq(q_obs.index)):
        print(f"Station {station_id}: Data will be resampled to {resample_freq}.", end="\r")

        q_model_f = q_model.resample(resample_freq).mean()

        #resample observed data only if each month has at least 15 days of data
        q_obs_f = (
            q_obs
            .resample(resample_freq)
            .apply(lambda x: x.mean() if x.count() >= 20 else np.nan)
        )

    else:
        q_model_f = q_model
        q_obs_f = q_obs

    # Align and drop NaNs
    merged_Q = pd.concat([q_model_f, q_obs_f], axis=1)
    merged_Q.columns = ["q_model", "q_obs"]
    merged_Q = merged_Q.dropna()

    return merged_Q

#==================================================

#Function to extract per station flow quantiles

def extract_flow_quantiles(merged_Q: pd.Series, station_id: str):
    """
    Extract Q50 and Q90 for model vs. observed discharge at a given station.
    Returns a single-row DataFrame or None if too short.
    Parameters
    ----------
    merged_Q : pd.Series
        DataFrame with two columns: 'q_model' and 'q_obs', indexed by date.
    station_id : str
        Identifier for the station (for logging purposes).
    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ['station', 'q50_model', 'q90_model', 'q50_obs', 'q90_obs']
        or None if insufficient overlapping data.
    """
   
    # If DataFrame, reduce to first column
    if isinstance(merged_Q, pd.DataFrame):
        q_model = merged_Q.iloc[:, 0]
        q_obs = merged_Q.iloc[:, 1]

    if len(merged_Q) > 12:  # at least 2 years of overlapping data
        print(f"Station {station_id}: {len(merged_Q)} overlapping data points found.", end="\r")
        q50_model = np.nanpercentile(q_model.values, 50)
        q90_model = np.nanpercentile(q_model.values, 10)
        q50_obs   = np.nanpercentile(q_obs.values, 50)
        q90_obs   = np.nanpercentile(q_obs.values, 10)

        return pd.DataFrame(
            {
                "station": [station_id],
                "q50_model": [q50_model],
                "q90_model": [q90_model],
                "q50_obs": [q50_obs],
                "q90_obs": [q90_obs],
            }
        )
    else:
        print(f"Station {station_id}: skipped ({len(q_model)} points only).", end="\r")
        return None

#==================================================
#change dictionary keys to upper case
def keys_upper(test_dict):
    res = dict()
    for key in test_dict.keys():
        if isinstance(test_dict[key], dict):
            res[key.upper()] = keys_upper(test_dict[key])
        else:
            res[key.upper()] = test_dict[key]
    return res

#==================================================
#Function to extract flow quantiles for all stations and models

def extract_station_quantiles(base_sim_dir, sim_subfolder, models, eval_Obs):
    """  
    Extract flow quantiles (Q50 and Q90) for each station and model.  
    Parameters  
    ----------
    base_sim_dir : str  
        Base directory containing model subdirectories.  
    sim_subfolder : str  
        Subfolder within each model directory containing simulation CSV files.  
    models : list of str  
        List of model names corresponding to subdirectory names.  
    eval_Obs : dict  
        Dictionary of observed discharge time series (keyed by station name).    
    Returns  
    -------
    pd.DataFrame  
        DataFrame with columns ['station', 'q50_model', 'q90_model', 'q50_obs', 'q90_obs', 'model']  
    """

    all_quantiles = {}

    for model in models:
        sim_files = glob.glob(f"{base_sim_dir}/{model}/{sim_subfolder}/*.csv")
        quantiles = []  # <-- collect per model

        for fpath in sim_files:
            station_name = os.path.splitext(os.path.basename(fpath))[0]

            #match cases
            station_name = station_name.upper()

            #change dict keys to upper case
            eval_Obs_upper = keys_upper(eval_Obs)
            if station_name in eval_Obs_upper.keys():

                obs_Q = eval_Obs_upper[station_name]
                #set index to datetime if not already
                if not isinstance(obs_Q.index, pd.DatetimeIndex):
                    obs_Q.index = pd.to_datetime(obs_Q.index)
                sim_Q = pd.read_csv(fpath, index_col=0, parse_dates=True)

                # Match frequency and resample to monthly
                q_merged = match_frequency_and_resample(sim_Q, obs_Q, 'D','ME', station_name)

                qn = extract_flow_quantiles(q_merged, station_name)
                if qn is not None:
                    qn["model"] = model
                    quantiles.append(qn)

            if quantiles:  # <-- use the correct list
                quantiles_df = pd.concat(quantiles, ignore_index=True)
            else:
                quantiles_df = pd.DataFrame()

            all_quantiles[model] = quantiles_df
        
    #save to df
    all_quantiles_df = pd.concat(all_quantiles.values(), ignore_index=True)

    return all_quantiles_df

#==================================================

def seasonal_Q_comparison(base_sim_dir, sim_subfolder, models, eval_Obs, season_map):
    """  
    Extract seasonal (DJF, MAM, JJA, SON) mean discharge for each station and model.  
    Parameters  
    ----------
    models : list of str  
        List of model names corresponding to subdirectory names.  
    eval_Obs : dict  
        Dictionary of observed discharge time series (keyed by station name).    

    season_map : dict
        Mapping of month numbers to season labels.  

    Returns  
    -------
    dict of pd.DataFrame  
        Dictionary with model names as keys and DataFrames with seasonal means as values.  
    """

    all_seasonal = {}

    for model in models:
        sim_files = glob.glob(f"{base_sim_dir}/{model}/{sim_subfolder}/*.csv")
        model_season = []

        for fpath in sim_files:
            station_name = os.path.splitext(os.path.basename(fpath))[0]

            #match cases
            station_name = station_name.upper()

            #change dict keys to upper case
            eval_Obs_upper = keys_upper(eval_Obs)

            if station_name in eval_Obs_upper.keys():

                obs_Q = eval_Obs_upper[station_name]
                sim_Q = pd.read_csv(fpath, index_col=0, parse_dates=True)

                # merge by date
                merged_df = pd.concat([sim_Q, obs_Q], axis=1, join="inner")
                merged_df.columns = ["q_model", "q_obs"]
                merged_df = merged_df.dropna()

                # add season label
                merged_df["season"] = merged_df.index.month.map(season_map)

                # seasonal climatology
                season_means = merged_df.groupby("season")[["q_model","q_obs"]].mean().reset_index()
                season_means["name"] = station_name
                season_means["model"] = model

                model_season.append(season_means)

            if model_season:
                all_seasonal[model] = pd.concat(model_season, ignore_index=True)
    
    model_seasons_df = pd.concat(all_seasonal.values(), ignore_index=True)
    model_seasons_df = model_seasons_df[["name", "model", "season", "q_obs", "q_model"]]

    return model_seasons_df

#==================================================

from pathlib import Path
import sys

def extract_multistation_baseflow(base_sim_dir, sim_subfolder, models, eval_Obs):
    """  
    Extract baseflow time series for each station and model using the best method based on observed data.  
    Parameters  
    ----------
    base_sim_dir : str
        Base directory containing model subdirectories.
    sim_subfolder : str
        Subfolder within each model directory containing simulation CSV files.
    models : list of str
        List of model names corresponding to subdirectory names.
    eval_Obs : dict  
        Dictionary of observed discharge time series (keyed by station name).    
    Returns  
    -------
    dict of pd.DataFrame  
        Dictionary with model names as keys and DataFrames with baseflow time series as values.
    """
    # Ensure station names are uppercase for consistency
    eval_Obs_upper = keys_upper(eval_Obs)

    all_models_Qb = {}
    all_models_BFI = {}

    for model in models:
        sim_dir = Path(base_sim_dir) / model / sim_subfolder
        stations_Qb = []
        stations_BFI = []

        for fpath in sorted(sim_dir.glob("*.csv")):
            station_name = fpath.stem.upper()

            if station_name not in eval_Obs_upper:
                #extract baseflow
                msg = f"Extracting baseflow for {station_name}"
                sys.stdout.write("\r" + msg + "  " * 20)
                sys.stdout.flush()
                continue

            obs_Q = eval_Obs_upper[station_name]
            #set index to datetime if not already
            if not isinstance(obs_Q.index, pd.DatetimeIndex):
                obs_Q.index = pd.to_datetime(obs_Q.index)
            sim_Q = pd.read_csv(fpath, index_col=0, parse_dates=True)

            #merge on index
            q_merged = match_frequency_and_resample(sim_Q, obs_Q, 'D','D', station_name)

            if q_merged.empty:
                continue
            
            #extract baseflow
            msg = f"Extracting baseflow for {station_name}"
            sys.stdout.write("\r" + msg + "  " * 20)
            sys.stdout.flush()

            #extract baseflow using multiple methods and select best based on KGE
            obs_bf_dict, obs_bfi, obs_kge = baseflow.separation(q_merged[["q_obs"]], return_bfi=True, return_kge=True)

            #select the best method based on KGE
            best_method = obs_kge.idxmax(axis=1).iloc[0]  #iloc[0] grabs the station name from the index

            #extract the baseflow timeseries for the best method
            obs_Qb = obs_bf_dict[best_method]
            obs_bfi = obs_bfi[best_method]

            #use the same best method to extract the baseflow from the simulated Q
            sim_bf_dict, sim_bfi = baseflow.separation(q_merged[['q_model']], return_bfi=True, return_kge=False, method=best_method)
            sim_Qb= sim_bf_dict[best_method]
            sim_bfi = sim_bfi[best_method]

            #combine into a dataframe
            bf_df = pd.concat([obs_Qb, sim_Qb], axis=1)
            bf_df.columns = ['obs_Qb', 'sim_Qb']
            bf_df['station'] = station_name
            
            #rearrange columns
            bf_df=bf_df[["station", 'obs_Qb', 'sim_Qb']]
            stations_Qb.append(bf_df)

            #bfi data
            bfi_df=pd.DataFrame({"obs_bfi":obs_bfi.values, "sim_bfi":sim_bfi.values})
            bfi_df['name'] = station_name
            bfi_df['model'] = model

            bfi_df=bfi_df[["name","obs_bfi","sim_bfi", "model"]]
            stations_BFI.append(bfi_df)


        if stations_Qb:
            model_bf_df = pd.concat(stations_Qb)
            model_bf_df['model'] = model
            all_models_Qb[model] = model_bf_df
        
        if stations_BFI:
            model_bfi_df = pd.concat(stations_BFI)
            model_bfi_df['model'] = model
            # Optionally store BFI data if needed
            all_models_BFI[model] = model_bfi_df

    return all_models_Qb, all_models_BFI


#==================================================
def seasonal_baseflow_analysis(all_models_Qb, models, season_map):
    """  
    Analyze seasonal mean baseflow for each station and model.  
    Parameters  
    ----------
    all_models_Qb : dict of pd.DataFrame  of models and station baseflow
        Dictionary with model names as keys and DataFrames with baseflow time series as values.  
    models : list of str  
        List of model names corresponding to keys in all_models_Qb.  
    season_map : dict
        Mapping of month numbers to season labels.  

    Returns  
    -------
    pd.DataFrame  
        DataFrame with columns ['name', 'model', 'season', 'obs_Qb', 'sim_Qb'] containing seasonal mean baseflow.  
    """

    seasonal_baseflow = []

    # Initialize a dictionary to hold seasonal data for the station
    for model in models:
        model_df = all_models_Qb[model]

        stations_Qb = []  # <-- collect per model

        for nameStation in model_df['station'].unique():

            station_df = model_df[model_df['station'] == nameStation]
            
            station_df = station_df.copy()
            station_df["season"] = station_df.index.month.map(season_map)

            # seasonal climatology
            season_Qb = station_df.groupby("season")[["obs_Qb","sim_Qb"]].mean().reset_index()
            season_Qb["name"] = nameStation
            season_Qb["model"] = model
            stations_Qb.append(season_Qb)
        
        if stations_Qb:  # <-- use the correct list
            model_Qb_df = pd.concat(stations_Qb)
            seasonal_baseflow.append(model_Qb_df)
        
        #convert to dataframe
    seasonal_baseflow_df = pd.concat(seasonal_baseflow)
    #rearrange columns
    seasonal_baseflow_df = seasonal_baseflow_df[["name", "model", "season", "obs_Qb", "sim_Qb"]]

    return seasonal_baseflow_df
#==================================================

def plot_multimodel_spread(flow_df: dict, seasons: dict,
                           obs_name: str, sim_name: str,
                           xlabel: str ,
                           ylabel: str ):
    
    """ 
    Plot the spread of simulated vs observed baseflow for multiple models with 95PPU
    and compute the P-factor.
    Parameters
    ----------
    flow_df : dict
        dataframe with flow for different models with columns: 'name', 'season', obs_name, sim_name, model for each model
    seasons : list
        List of seasons to consider (e.g., ["DJF", "MAM", "JJA", "SON"]) mapped to month numbers
    obs_name : str
        Name of the observed flow column
    sim_name : str
        Name of the simulated flow column

    Returns
    -------
    plots the scatter plot with error bars and 95PPU band
    and prints the P-factor
    """
    plt.figure(figsize=(8,6), dpi=120)

    all_obs = []
    all_mean_sim = []
    for season in seasons:
        for station in flow_df['name'].unique():
            q_sim_season = flow_df.loc[
                (flow_df['season'] == season) &
                (flow_df['name'] == station), sim_name]
            
            eval_Obs_season = flow_df.loc[
                (flow_df['season'] == season) &
                (flow_df['name'] == station), obs_name]

            mean_sim = q_sim_season.mean()
            min_sim = q_sim_season.min()
            max_sim = q_sim_season.max()
            x_obs = eval_Obs_season.mean()

            all_obs.append(x_obs)
            all_mean_sim.append(mean_sim)

            plt.errorbar(
                x_obs, mean_sim,
                yerr=[[mean_sim - min_sim], [max_sim - mean_sim]],
                fmt='o', label=season,color='dodgerblue', markersize=5,
                alpha=0.6, capsize=2, ecolor='gray', elinewidth=0.7, capthick=0.6
            )
    # arrays
    all_obs = np.array(all_obs)
    all_mean_sim = np.array(all_mean_sim)

    # residuals
    residuals = all_mean_sim - all_obs
    lower = np.percentile(residuals, 2.5)
    upper = np.percentile(residuals, 97.5)

    # 1:1 line
    lims = [0, 60]
    plt.plot(lims, lims, 'r-', lw=1)

    # 95PPU band
    x_line = np.linspace(lims[0], lims[1], 200)
    plt.fill_between(x_line, x_line + lower, x_line + upper,
                    color='gray', alpha=0.2, label='95PPU')

    # ---- Compute P-factor ----
    inside = ((all_mean_sim >= all_obs + lower) & (all_mean_sim <= all_obs + upper))
    p_factor = inside.mean() * 100
    print(f"P-factor (percentage of points within 95PPU): {p_factor:.1f}%")

    # axes, legend
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.grid(alpha=0.4)

    #handles, labels = plt.gca().get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    #plt.legend(by_label.values(), by_label.keys())

    plt.show() 