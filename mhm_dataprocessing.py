import pandas as pd
import xarray as xr
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
    # Define a range of bandwidths to test e.g. bandwidths = np.linspace(0.001, 0.9, 500)

    # Perform cross-validated grid search for bandwidth h
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=5)  # 5-fold cross-validation
    grid.fit(data[:, None])

    # Optimal bandwidth
    optimal_h = grid.best_params_['bandwidth']
    return optimal_h
