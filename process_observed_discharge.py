
import pandas as pd
import os
import numpy as np
import shutil
import openpyxl
import glob

import zipfile

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

    return station_Q, info_df

#============================================================================================================================

def extract_timeseries_waterinfo(source_folder):
    """
    Extracts time series data and metadata from Waterinfo csv files.

    Parameters:
    source_folder (str): Folder containing Waterinfo .xlsx files.

    Returns:
    station_Q (dict): Dictionary of station names → time series DataFrames.
    info_df (pd.DataFrame): DataFrame with station names as index and lat/lon as columns.
    """
    station_info = {}  # Metadata per station
    station_Q = {}     # Time series per station

    files = glob.glob(os.path.join(source_folder, '*.csv'))

    for f in files:
        try:
            df = pd.read_excel(f, engine='openpyxl', header=None)
            filename = os.path.basename(f)
            
            # Extract header info
            df_header = df.head(6)
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

    return station_Q, info_df







