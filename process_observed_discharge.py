
import pandas as pd
import os
import numpy as np
import shutil
import openpyxl
import xarray as xr
import glob
import pickle
import matplotlib.pyplot as plt
import zipfile
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
#//////////////////////////////////////

def unzip_files(zipped_folder, unzipped_folder):
    os.makedirs(unzipped_folder, exist_ok=True)

    zip_files = [f for f in os.listdir(zipped_folder) if f.lower().endswith('.zip')]

    for i, zipfile_name in enumerate(zip_files, 1):
        zip_path = os.path.join(zipped_folder, zipfile_name)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                print(f"[{i}/{len(zip_files)}] Extracting {zipfile_name}...", end ='\r')
                zf.extractall(path=unzipped_folder)
        except zipfile.BadZipFile:
            print(f"Skipping corrupt file: {zipfile_name}")

#============================================================================================================================
def copy_to_common_folder(src_folder, common_folder):
    """
    Copies all files from subdirectories within the source folder to the common folder.

    Parameters:
    src_folder (str): Path to the folder containing unzipped subfolders.
    common_folder (str): Path to the common folder where files will be copied.
    """
    # Ensure the common folder exists
    os.makedirs(common_folder, exist_ok=True)

    # Iterate through each subfolder in the source folder
    for subfolder in os.listdir(src_folder):
        subfolder_path = os.path.join(src_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.xlsx') or file_name.endswith('.csv'):
                    src_file = os.path.join(subfolder_path, file_name)
                    dst_file = os.path.join(common_folder, file_name)

                    shutil.copy(src_file, dst_file)
                    print(f"Copying {file_name} from {subfolder} to {common_folder}...", end='\r')

#============================================================================================================================

def extract_timeseries_wallonie(source_folder):
    """
    Extracts time series data and metadata from Hydrométrie Wallonie Excel files.

    Parameters:
    source_folder (str): Folder containing Wallonie .xlsx files.

    Returns:
    station_Q (dict): Dictionary of station names → time series DataFrames.
    info_df (pd.DataFrame): DataFrame with station names as index and lat/lon as columns.
    """
    station_info = {}  # Metadata per station
    station_Q = {}     # Time series per station

    files = glob.glob(os.path.join(source_folder, '*.xlsx'))

    for f in files:
        try:
            df = pd.read_excel(f, engine='openpyxl')
            filename = os.path.basename(f)
            
            # Extract header info
            df_header = df.head(8)
            station_name = str(df_header.iloc[0, 1]).strip()
            lat = float(df_header.iloc[1, 1])
            lon = float(df_header.iloc[2, 1])
            station_info[station_name] = {'lat': lat, 'lon': lon}

            # Extract time series
            data = df.iloc[9:, [0, 1]].copy()
            data.columns = ['Date', 'Q']
            data.dropna(subset=['Date'], inplace=True)
            data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
            data.set_index('Date', inplace=True)
            data = data.resample('D').mean()

            station_Q[station_name] = data
        except Exception as e:
            print(f"Failed to process {f}: {e}")

    info_df = pd.DataFrame.from_dict(station_info, orient='index')

    info_df = info_df.reset_index()
    #rename the columns
    info_df.columns = ['station_name', 'station_latitude', 'station_longitude']

    return station_Q, info_df

#============================================================================================================================

def load_or_extract_wallonie_data(dict_path, df_path, compute_func, *args, **kwargs):
    """
    Load Q_dict and station_coords from pickle if available,
    otherwise run compute_func and cache the results.
    Args:
        dict_path (str): Path to the pickle file for Q_dict.
        df_path (str): Path to the pickle file for station_coords.
        compute_func (callable): Function to compute Q_dict and station_coords.
        *args: Positional arguments for compute_func.
        **kwargs: Keyword arguments for compute_func.
    Returns:
        tuple: Q_dict and station_coords.

    """
    if os.path.exists(dict_path) and os.path.exists(df_path):
        print("Loading cached Wallonie discharge data...")
        with open(dict_path, 'rb') as f1, open(df_path, 'rb') as f2:
            return pickle.load(f1), pickle.load(f2)
    else:
        print("Processing Wallonie discharge data (first time)...")
        Q_dict, station_coords = compute_func(*args, **kwargs)
        with open(dict_path, 'wb') as f1:
            pickle.dump(Q_dict, f1)
        with open(df_path, 'wb') as f2:
            pickle.dump(station_coords, f2)
        return Q_dict, station_coords

#============================================================================================================================
#Extract subset of stations for model evaluation based on peak discharge
def extract_eval_stations(Q_dict, threshold_max, min_length_days):
    """
    Extracts stations for model evaluation based on peak discharge and non-NaN data length.
    
    Parameters:
    Q_dict (dict): Dictionary of station names → time series DataFrames with a 'Q' column.
    threshold_max (float): Minimum peak discharge (Q) required for inclusion.
    min_length_days (int): Minimum number of valid (non-NaN) daily discharge values required.
    
    Returns:
    eval_stations (dict): Dictionary of stations that meet the criteria.
    """
    eval_stations = {}

    for station_name, df in Q_dict.items():
        if 'Q' not in df.columns:
            continue  # Skip if 'Q' column is missing

        q_valid = df['Q'].dropna()
        max_Q = q_valid.max()
        valid_days = len(q_valid)

        if max_Q > threshold_max and valid_days >= min_length_days:
            eval_stations[station_name] = df
        # Optionally log excluded stations:
        # else:
        #     print(f"Excluded {station_name}: max Q = {max_Q}, valid days = {valid_days}")

    return eval_stations


#============================================================================================================================
def extract_netCDF_timeseries(dataset_path, stations_file, var):
    """
    Extracts time series data from a NetCDF file for multiple stations.
    Parameters:
    dataset_path (str): Path to the NetCDF file.
    stations_file (str): Path to the CSV file containing station names and coordinates.
    var (str): Variable name to extract from the NetCDF file.

    Returns:
    pd.DataFrame: DataFrame containing the extracted time series data.
    """


    # Load the NetCDF file using xarray
    dataset = xr.open_dataset(dataset_path)

    # Check variable exists
    if var not in dataset.variables:
        raise ValueError(f"Variable '{var}' not found in dataset.")

    # Load station list
    stations = pd.read_csv(stations_file)

    # Ensure required columns exist
    if not {'name', 'lat', 'lon'}.issubset(stations.columns):
        raise ValueError("CSV file must contain 'name', 'lat', 'lon' columns.")

    # Output directory
    output_dir = f'extracted_timeseries/{var}'
    os.makedirs(output_dir, exist_ok=True)

    # Loop over stations
    for _, row in stations.iterrows():
        station_name = row['name']
        lat = row['lat']
        lon = row['lon']

        try:
            # Extract time series at nearest grid point
            data = dataset[var].sel(lat=lat, lon=lon, method='nearest')

            # Convert to DataFrame and save
            df = data.to_pandas()
            df.index.name = 'Date'
            df.columns = ['Q']
            df.to_csv(os.path.join(output_dir, f"{station_name}.csv"), header=True, index=True)

            print(f"Saved: {station_name}.csv", end='\r')
        except Exception as e:
            print(f"Error processing station '{station_name}' ({lat}, {lon}): {e}")

    print("All stations processed!")

#=============================================================================================================================
#MODEL PERFORMANCE

def compute_model_metrics(observed, simulated, epsilon=1e-6):
    """
    Computes NSE, KGE, PBIAS, and LNSE between observed and simulated data.

    Parameters:
    - observed: array-like of observed values
    - simulated: array-like of simulated values
    - epsilon: small constant to avoid log(0) in LNSE

    Returns:
    - metrics (dict): Dictionary with rounded values of all metrics
    """
    observed = np.array(observed)
    simulated = np.array(simulated)

    # Drop NaNs and align

    if len(observed) == 0 or np.std(observed) == 0:
        return {'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan, 'LNSE': np.nan}

    # NSE
    nse_denom = np.sum((observed - np.mean(observed)) ** 2)
    nse = 1 - (np.sum((observed - simulated) ** 2) / nse_denom) if nse_denom != 0 else np.nan

    # KGE
    if np.std(simulated) == 0 or np.mean(observed) == 0:
        kge = np.nan
    else:
        # Convert and flatten arrays
        x = np.array(observed, dtype=float).flatten()
        y = np.array(simulated, dtype=float).flatten()

        # Drop NaNs from both
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        # Now safely calculate correlation
        if len(x) >= 2:
            r = np.corrcoef(x, y)[0, 1]
        else:
            r = np.nan  # Not enough data

        # Calculate KGE components
        if np.std(simulated) == 0 or np.mean(observed) == 0:
            kge = np.nan
        else:
            beta = np.mean(simulated) / np.mean(observed)
            gamma = np.std(simulated) / np.std(observed)
            kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

    # PBIAS
    pbias = 100 * np.sum(simulated - observed) / np.sum(observed) if np.sum(observed) != 0 else np.nan

    # LNSE
    # Make sure you're working with NumPy arrays of float type
    observed = np.array(observed, dtype=float)
    simulated = np.array(simulated, dtype=float)

    # Drop any NaNs (or infinite values if needed)
    mask = ~np.isnan(observed) & ~np.isnan(simulated)
    observed = observed[mask]
    simulated = simulated[mask]

    # Now apply log safely
    epsilon = 1e-10
    log_obs = np.log(observed + epsilon)
    log_sim = np.log(simulated + epsilon)
    lnse_denom = np.sum((log_obs - np.mean(log_obs)) ** 2)
    lnse = 1 - (np.sum((log_obs - log_sim) ** 2) / lnse_denom) if lnse_denom != 0 else np.nan


    return {
        'NSE': np.round(nse, 2),
        'KGE': np.round(kge, 2),
        'PBIAS': np.round(pbias, 2),
        'LNSE': np.round(lnse, 2)
    }
#=============================================================================================================================
#CONVERT DICTIONARY TO DATAFRAME
def dict_to_df(metrics_dict):
    """
    Convert a dictionary of metrics to a DataFrame.
    
    Parameters:
    metrics_dict (dict): Dictionary of metrics with station names as keys.
    
    Returns:
    pd.DataFrame: DataFrame containing the metrics.
    """
    df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'name'}, inplace=True)
    return df

#=============================================================================================================================
#TIMESERIES PLOTTING

def plot_station_timeseries(station_name, observed_dict, sim_dir):
    """
    Plot observed and simulated time series for a given station.
    """
    # Load observed
    if station_name not in observed_dict:
        print(f"Station {station_name} not found in observed data.")
        return
    obs = observed_dict[station_name].copy()
    obs = obs.replace(-9999, pd.NA).resample('D').mean()

    # Load simulated
    sim_file = os.path.join(sim_dir, f"{station_name}.csv")
    if not os.path.exists(sim_file):
        print(f"Simulation file not found for station: {sim_file}")
        return
    sim = pd.read_csv(sim_file, parse_dates=['time'], index_col='time')

    # Merge and align
    df = pd.concat([obs, sim], axis=1)
    df.columns = ['Observed', 'Simulated']
    #df = df.dropna()

    # Trim to overlapping date range
    start_date = max(obs.dropna().index.min(), sim.dropna().index.min())
    end_date = min(obs.dropna().index.max(), sim.dropna().index.max())
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    df = df.sort_index()

    # Plot
    plt.figure(figsize=(11, 3.5), dpi=200)
    df.plot(ax=plt.gca(), linewidth=1, color=['black', 'dodgerblue'])
    plt.title(f"Discharge at {station_name}")
    plt.ylabel("Discharge (m$^3$/s)")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

#=============================================================================================================================
#MAP MODEL PERFORMANCE

def map_model_stats(boundary_shp_path, rivers_shp_path, stats_gdf, performance_statistic, cmap):
    """
    Maps model statistics on a geographical plot.
    
    Parameters:
    boundary_shp (str): Path to the shapefile for the boundary.
    stats_gdf (GeoDataFrame): GeoDataFrame containing the statistics to be plotted.
    performance_statistic (str): The name of the performance statistic to be plotted.
    c_bar (str): The label for the color bar.

    Returns:
    None: Displays the plot.
    """
    shp = gpd.read_file(boundary_shp_path)
    rivers = gpd.read_file(rivers_shp_path)

    gdf = stats_gdf[stats_gdf[performance_statistic]>0] .copy() # Keep only positive values

    stat_min = gdf[performance_statistic].min()
    stat_max = gdf[performance_statistic].max()

    if stat_max != stat_min:
        norm_kge = (gdf[performance_statistic] - stat_min) / (stat_max - stat_min)
        marker_sizes = norm_kge * 150
    else:
        marker_sizes = np.full(len(gdf), 40)
    
    # Normalize color range for KGE
    norm = Normalize(vmin=0.2, vmax=0.9)
    cmap = plt.get_cmap(cmap)

    # Set up figure and axis
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 8), dpi=110)

    # Plot shapefile boundary
    shp.boundary.plot(ax=ax, edgecolor='b', linewidth=1.0, alpha=0.5, transform=ccrs.PlateCarree())
    # Plot rivers
    rivers.plot(ax=ax, edgecolor='gray', linewidth=0.5, alpha=0.3, transform=ccrs.PlateCarree())

    # Scatter plot of GeoDataFrame with color and size
    sc = ax.scatter(
        gdf.geometry.x, gdf.geometry.y,
        c=gdf[performance_statistic],
        s=marker_sizes,
        cmap=cmap,
        norm=norm,
        edgecolor='gray',
        linewidth=0.4,
        transform=ccrs.PlateCarree()
    )

    #Add grids to the map
    ax.gridlines(draw_labels=True, color='gray', lw=0.6, alpha=0.2)


    # Add colorbar
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label(performance_statistic, fontsize=12)

    # Optional
    ax.set_title(f'{performance_statistic}', fontsize=14)
    plt.tight_layout()





