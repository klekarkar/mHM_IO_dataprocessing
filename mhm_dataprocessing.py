import pandas as pd
import xarray as xr
from xarray import DataArray
import numpy as np
import geopandas as gpd
import rioxarray as rio
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import mapping
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


#function
def clip_to_region(shapefile, xr_dataset):
    """
    This function clips an xarray dataset to a given shapefile.

    Parameters
    ----------
    shapefile : geopandas.GeoDataFrame
        The shapefile to clip the dataset to.
    xr_dataset : xarray.Dataset
        The dataset to clip.

    Returns
    -------
    xarray.Dataset
        The clipped dataset.
    """
        #set shapefile to crs 4326
    shapefile = shapefile.to_crs('epsg:4326')

    #drop bnds dimension
    xr_dataset = xr_dataset.drop_dims("bnds", errors="ignore")

    #set spatial dimensions
    xr_dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

    #write crs
    xr_dataset.rio.write_crs("EPSG:4326", inplace=True)

    #clip
    clipped = xr_dataset.rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True)

    return clipped

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def compute_optimal_h(data, bandwidths):
    """Function to find the optimal bandwidth for a kernel density estimate using cross-validated grid search.

    Parameters
    ----------
    data : numpy array
        Array of values to estimate the density function

    Returns
    -------
    optimal_h : float
        Optimal bandwidth for the kernel density estimate
    """
    # Define a range of bandwidths to test 
    # e.g. bandwidths = np.linspace(0.001, 0.9, 500)

    # Perform cross-validated grid search for bandwidth h
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=5)  # 5-fold cross-validation
    grid.fit(data[:, None])

    # Optimal bandwidth
    optimal_h = grid.best_params_['bandwidth']
    return optimal_h


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def compute_standardized_anomaly(ds: DataArray, freq) -> DataArray:
    """
    Compute the standardized anomaly of a DataArray.
    The standardized anomaly of each month with respect to all the corresponding months in the time series.
    For each month, the standardized anomaly is calculated as the anomaly divided by the standard deviation of the anomaly.
    Parameters
    ----------
    ds : DataArray
        The input rainfall data. At daily temporal scale.
    freq: string
        The frequency of target resampled data. 'M'/'ME' for monthly
    Returns
    -------
    DataArray
        The standardized anomaly.
    """
    # Step 1: Compute monthly total rainfall
    monthly_mean = ds.resample(time=freq).sum('time')

    # Step 2: Compute mean of each month across all years
    monthly_mean_grouped = monthly_mean.groupby('time.month').mean()

     # Step 3: Compute monthly anomalies
    # vectorized more efficient method
    ds_anomalies = monthly_mean.groupby('time.month') - monthly_mean_grouped

    # Step 4: Calculate the standard deviation of the anomalies for each month
    # Group anomalies by month and compute standard deviation over the time dimension
    anomalies_stdev_monthly = ds_anomalies.groupby('time.month').std()
    
    #compute the standardized monthly anomalies
    #Divide each monthly anomaly by the standard deviation of the corresponding month to get the standardized anomalies.
    standardized_anomalies = ds_anomalies.groupby('time.month') / anomalies_stdev_monthly

    return standardized_anomalies

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def group_weekly(dataset):
    """
    Groups an xarray dataset into weekly intervals, ensuring that:
    - The first week (Week 0) runs from January 1st to January 7th.
    - Each week consists of exactly 7 days, except for the final week, 
      which may be 1 or 2 days to include all remaining days in the year.

    Parameters:
    ----------
    dataset : xarray.Dataset
        An xarray dataset containing a time dimension.

    Returns:
    -------
    weekly_ds : xarray.Dataset
        A new dataset with weekly means computed, where:
        - The "time" coordinate represents the last day of each week.
        - All other dimensions (e.g., lat, lon) are preserved.
    """
    # Compute the custom week number
    weeks = ((dataset['time'].dt.dayofyear - 1) // 7)  # Subtract 1 to ensure 01/01 is within week 0

    # Initialize a list to store weekly data
    weekly_means = []

    for year in np.unique(dataset['time'].dt.year):  # Iterate over all years
        for i in np.unique(weeks.values):  # Iterate over unique weeks

            # Select data for the year
            datayear = dataset.sel(time=dataset['time'].dt.year == year)

            # Select data for the given week for the selected year
            dataweek = datayear.sel(time=(datayear['time'].dt.dayofyear - 1) // 7 == i)

            if dataweek['time'].size > 0:  # Ensure there is valid data
                startdate = dataweek['time'][0].values
                enddate = dataweek['time'][-1].values  # Last day of the selected week

                # Select data for the given time range and compute weekly mean
                weekly_data = dataset.sel(time=slice(startdate, enddate)).mean(dim='time')

                # Assign the weekly end date as the new time coordinate
                weekly_data = weekly_data.assign_coords(time=enddate, week=i)

                # Append the computed weekly mean to the list
                weekly_means.append(weekly_data)

    # Combine all weekly data into a single xarray dataset along the "time" dimension
    weekly_ds = xr.concat(weekly_means, dim="time")

    return weekly_ds
