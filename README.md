# Running SFNO inference with Earth2Studio

This folder contains inference scripts for running SFNO forecasts with earth2studio 0.10.0.

## Environment setup *specific to BU SCC*

```
conda create -n e2s-new python=3.12 -y 
conda activate e2s-new
pip install uv
export UV_CACHE_DIR="INSERT_YOUR_PERSONAL_PROJECT_DIRECTORY/uv_cache"
uv pip install "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.10.0"
uv pip install "earth2studio[fcn]"
uv pip install numpy matplotlib pandas xarray cartopy cmocean tqdm scikit-learn
uv pip install "makani @ git+https://github.com/NVIDIA/modulus-makani.git@28f38e3e929ed1303476518552c64673bbd6f722"
uv pip install earth2studio[sfno]
```

- Run this to check earth2studio wasn't installed in the home directory
```
python -c "import earth2studio; print('Found at:', earth2studio.__file__)"
```

- the UV cache directory resets to be the home directory on the BU SCC after each session ends, so you may want to add the export UV_CACHE_DIR line to your .bashrc file with:
```
echo 'export UV_CACHE_DIR="INSERT_YOUR_PERSONAL_PROJECT_DIRECTORY/uv_cache"' >> ~/.bashrc
source ~/.bashrc
echo $UV_CACHE_DIR # to verify it worked!
```

## Directory contents:
##### `deterministic_update.py`

- rewrites the earth2studio deterministic function to handle saving only specific variables in the output.

##### `SFNO_update.py`

- rewrites the earth2studio SFNO code (and some relevant makani code) to handle loading specific SFNO checkpoints and running inference with them.

##### `utils.py`
- helper functions e.g. creating Initialization files, opening files.

##### `inference.py`
- main script to run SFNO inference with earth2studio.

#### `inference_job_arr.sh`
- uses a job array to parallelize inference runs on BU SCC.

#### `inference.sh`
- single-job (non-array) variant of the submission script for running one experiment without splitting across tasks.

#### `configs/`
- contains .json config files for different experiments and inference runs.

#### `examples/`
- contains example notebooks and plotting/metric helpers:
  - `example_Hurricane_Ian_inference.ipynb` — end-to-end inference example for Hurricane Ian.
  - `example_create_init_file.ipynb` — walkthrough for building initialization files.
  - `example_visualize_inference_runs_atmosphericRiver.ipynb` — visualizing atmospheric river inference output.
  - `viz_perturbations_3.4.ipynb` — visualizing perturbation experiments.
  - `plot_metric_utils.py` — helper functions for plotting and metrics (mse, IoU, amplitude) from inference runs.
  - `figures/` — saved figures produced by the example notebooks.

## Running inference

### Before your first submission
Open `inference_job_arr.sh` (and/or `inference.sh`) and fill in the following placeholders:

| Placeholder | Where | What to set it to |
| :--- | :--- | :--- |
| `PROJECT_NAME` | `#$ -P PROJECT_NAME` | Your SCC project name. |
| `INSERT_LOG_DIRECTORY` | `LOG_DIR="INSERT_LOG_DIRECTORY/Experiment${EXP_NUM}"` | Directory for job stdout/stderr logs. The script will create `Experiment{N}/` under it automatically. |
| `INSERT_PROJECT_DIRECTORY` | `cd INSERT_PROJECT_DIRECTORY` | Path to your local checkout of this repo (where `inference.py` lives). |
| `earth2studio` | `conda activate e2s-new` | Make sure this matches the env name you created in the *Environment setup* section above (e.g. `e2s-new`). |

You also need to set the initialization file directory inside `inference.py`:

| Placeholder | Where | What to set it to |
| :--- | :--- | :--- |
| `INSERT_YOUR_DIRECTORY_OF_INITIALIZATION_FILES` | `data_create_fp = f"/INSERT_YOUR_DIRECTORY_OF_INITIALIZATION_FILES/Initialize_..."` in `inference.py` | Directory where `Initialize_*.nc` files are stored / will be cached. |

And in your experiment config (`configs/expN.json`), set `paths.base_output_dir` to the directory where forecast NetCDFs should be saved.

### Submitting a job
For a parallelized run using a job array:
```
qsub inference_job_arr.sh <experiment_number>
```
- `<experiment_number>` corresponds to the config file `configs/expN.json`.
- Omitting the number defaults to experiment 2 (`configs/exp2.json`).
- The number of array tasks is controlled by `#$ -t 1-4` at the top of `inference_job_arr.sh` — increase the upper bound to spread epochs across more GPUs.

For a single (non-array) run:
```
qsub inference.sh <experiment_number>
```

### Monitoring & cleanup
```
qstat -u $USER              # list your running/queued jobs
qstat -j <job_id>            # detailed status for one job
qdel <job_id>                # cancel a job (or job array)
```
Job logs go in `LOG_DIR/Experiment{N}/inference_parallel_<JOB_ID>_<TASK_ID>.log` (any print statements, inference info, etc.).

## Configuration options

## 1. Experiment Setup (`experiment_setup`)

| Key | Type | Description | Options / Examples |
| :--- | :--- | :--- | :--- |
| **`event_type`** | `string` | The specific extreme event category. | `"atmospheric_river"`, `"tropical_cyclone"`, `"heat_wave"`, `"severe_convection"` |
| **`valid_timestep`** | `string` | The target datetime to forecast for (ISO 8601 format). | `"2022-12-27T00:00:00"` |
| **`leadtimes_days`** | `list[int]` | The number of days *prior* to the valid timestep to initialize the model. | `[3, 5, 7]` (Forecasts initialized 3, 5, and 7 days before the event) |
| **`variables_to_save`** | `list[str]` | Specific variables to save to the output NetCDF to save space/time. | `["tcwv", "u700", "v700", "z500"]`, `["msl", "u10m", "v10m"]` |
| **`ema`** | `bool` | Whether to load **Exponential Moving Average** weights or standard checkpoint weights. | `true` (use EMA), `false` (standard) |
| **`compute_ivt`** | `bool` | Flag to calculate Integrated Vapor Transport (IVT) during inference. | `true` or `false` |
| **`bounding_box`** | `dict` | optional bounding box to crop data to a specific region. | {"latitude_min": 25.0, "latitude_max": 56.2, "longitude_min": 220.0, "longitude_max": 252.8} |

---

## 2. Model Parameters (`model_parameters`)

| Key | Type | Description | Options / Examples |
| :--- | :--- | :--- | :--- |
| **`fine_tuning_start_epoch`** | `int` | The epoch number where the directory structure switches from pre-training to fine-tuning. | `71` |
| **`epochs_to_run`** | `str` or `list` | Determines which checkpoints to process. | `"all"` (1-90)<br>`"odds"` (1, 3, 5...)<br>`"evens"` (2, 4, 6...)<br>`[10, 20, 85]` (Specific list) |

---

## 3. Paths (`paths`)

| Key | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| **`base_output_dir`** | `string` | The root directory where forecast NetCDF output is saved (under `Experiment{N}/{date}/`). Job stdout/stderr logs go to `LOG_DIR` set in the submission script. |  |

---

#### 📝 Example Configuration
```json
{
    "experiment_setup": {
        "event_type": "atmospheric_river",
        "valid_timestep": "2022-12-27T00:00:00",
        "leadtimes_days": [3, 5, 7],
        "variables_to_save": ["tcwv", "u700", "v700", "z500"],
        "ema": false,
        "compute_ivt": true
    },
    "model_parameters": {
        "fine_tuning_start_epoch": 71,
        "epochs_to_run": "evens"
    },
    "paths": {
        "base_output_dir": "INSERT_YOUR_PERSONAL_PROJECT_DIRECTORY"
    }
}
```
### Variable options within this SFNO model: 
[
      "u10m",
      "v10m",
      "u100m",
      "v100m",
      "t2m",
      "sp",
      "msl",
      "tcwv",
      "d2m", # 2m dewpoint temperature (additional channel from FCNv2)
      "u50",
      "u100",
      "u150",
      "u200",
      "u250",
      "u300",
      "u400",
      "u500",
      "u600",
      "u700",
      "u850",
      "u925",
      "u1000",
      "v50",
      "v100",
      "v150",
      "v200",
      "v250",
      "v300",
      "v400",
      "v500",
      "v600",
      "v700",
      "v850",
      "v925",
      "v1000",
      "z50",
      "z100",
      "z150",
      "z200",
      "z250",
      "z300",
      "z400",
      "z500",
      "z600",
      "z700",
      "z850",
      "z925",
      "z1000",
      "t50",
      "t100",
      "t150",
      "t200",
      "t250",
      "t300",
      "t400",
      "t500",
      "t600",
      "t700",
      "t850",
      "t925",
      "t1000",
      "q50",
      "q100",
      "q150",
      "q200",
      "q250",
      "q300",
      "q400",
      "q500",
      "q600",
      "q700",
      "q850",
      "q925",
      "q1000"
    ]
