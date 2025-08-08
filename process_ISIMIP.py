import os
import glob
import xarray as xr
import logging

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
