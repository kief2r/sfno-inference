import os
from dotenv import load_dotenv

from earth2studio.io import ZarrBackend
from SFNO_update import SFNO
import earth2studio.data as data
from earth2studio.models.auto import Package
from deterministic_update import deterministic

from datetime import datetime, timedelta
import xarray as xr
import sys
import gc
import numpy as np

import torch


if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
import time
time_start = time.time()


########## to Select #############
INITIAL_DATA_FP = "/PROJECT_DIRECTORY/Initializations/Initialize_2022_12_24T00_nsteps12.nc" # filepath for initial data
RESULTS_FP = "/PROJECT_DIRECTORY/sfno-inference/Forecasts/" # ressults will be saved here with the name format "Checkpoint{n_epoch}_{inference_name}.nc"
START_DATETIME = '2022-12-24T00:00:00' # this should match the initial timestep
VARIABLE_TO_SAVE = ['msl', 'tcwv'] #Only save selected variables - it slows down inference SIGNIFICANTLY to save all 74 variables
N_STEPS = 12 # each n_step = 6hrs, so n_steps=12 means 3 days of forecast, n_steps=20 means 5 days of forecast, etc.
EPOCH_SELECT = 70 # select specific epoch or select "All" to run all epochs
#################################


# Create the inference name based on the start datetime and number of steps
inference_name = datetime.fromisoformat(START_DATETIME).strftime("%Y_%m_%dT%H")+'_nsteps'+str(N_STEPS)
# Calculate the final datetime based from the start datetime and number of steps
final_datetime = (datetime.fromisoformat(START_DATETIME) + timedelta(hours = int(N_STEPS*6))).isoformat() 


if os.path.exists(INITIAL_DATA_FP):
    print(f"Data already preprocessed: {INITIAL_DATA_FP}")
else:
    sys.exit(f"Initial Data Not Found at: {INITIAL_DATA_FP}")

#make this xarray into a dataarray file for earth2studio
initial_data = data.DataArrayFile(INITIAL_DATA_FP)

time_1 = time.time()
print(f"Data loaded in {time_1 - time_start:.2f} seconds")

if EPOCH_SELECT == "All":
    epoch_list = np.arange(1,71,1)
else:
    epoch_list = [EPOCH_SELECT]



for n_epoch in epoch_list:
    time_2 = time.time()

    results_out_fp = os.path.join(RESULTS_FP, f"Checkpoint{n_epoch}_{inference_name}.nc")

    # Check if the results file already exists
    if os.path.exists(results_out_fp):
        print(f"Results file {results_out_fp} already exists. Skipping to next epoch.")
        continue  # Skip the rest of the loop and go to the next iteration
    else:
        os.makedirs(os.path.dirname(results_out_fp), exist_ok=True)

        load_dotenv()  
        # model checkpoint
        src_dir = "/projectnb/eb-general/shared_data/data/processed/FourCastNet_sfno/Checkpoints_SFNO/sfno_linear_74chq_sc3_layers8_edim384_dt6h_wstgl2/v0.1.0-seed999/"

        # Load the model package from storage
        model_package = Package(src_dir, cache = False)
        model = SFNO.load_model(model_package, checkpoint_name = 'ckpt_mp0_epoch'+str(n_epoch)+'.tar')

        # Create the IO handler, store in memory
        io = ZarrBackend()
        
        with torch.no_grad():
            # run inference
            io = deterministic([START_DATETIME], N_STEPS, model, initial_data, io, variables_list=VARIABLE_TO_SAVE)

        print(io.root.tree())


        # save results to netcdf
        # Open the Zarr group from the in-memory store using xarray
        ds = xr.open_zarr(io.root.store)

        # Convert the 'time' coordinate in ds to datetime64 format
        ds["time"] = ds["time"].astype("datetime64[ns]")

        # Convert lead_time from nanoseconds to timedelta64[ns]
        base_time = ds["time"].values  # shape (n_time,)
        lead_timedelta = ds["lead_time"].values.astype("timedelta64[ns]")  # shape (n_lead_time,)
        # Broadcast to 2D: (time, lead_time)
        valid_timesteps = (base_time[:, None] + lead_timedelta[None, :]).flatten() 
        # Drop the old lead_time coordinate
        ds = ds.drop_vars("lead_time")

        # Assume ds has dimensions (time, lead_time, lat, lon) and only one time
        initial_time = str(ds["time"].values[0])  # Save the initial time as a string
        # Remove the time dimension by selecting the first (and only) time
        ds = ds.isel(time=0).drop_vars("time")
        # Add the initial time as a global attribute
        ds.attrs["initial_time"] = initial_time

        # Create valid_time by adding lead_timedelta to base_time
        ds = ds.rename({"lead_time": "valid_time"})
        # Assign valid_time as a coordinate
        ds = ds.assign_coords(valid_time=(("valid_time",), valid_timesteps))

        # only save the final time step
        if np.datetime64(final_datetime) in ds["valid_time"].values:
            ds = ds.sel(valid_time=[final_datetime])
            ds = ds[VARIABLE_TO_SAVE]
            ds.to_netcdf(results_out_fp, mode="w", format="NETCDF4")
            print(f"Results saved to {results_out_fp}")
        else:
            print(f"ERROR: final_datetime {final_datetime} not found in ds['valid_time']. No file saved.")


        #some cleanup
        torch.cuda.empty_cache()
        del model_package
        del model
        del io
        del ds
        gc.collect()
        time_3 = time.time()
        print(f"Epoch {n_epoch} done: {time_3 - time_2:.2f} seconds")


