# Generic Frequentist Profile Likelihood Scanner

A highly optimized, physics-agnostic, and massively parallelized framework for extracting multi-parameter frequentist confidence intervals and topological profile likelihood contours. 

This codebase minimizes $-2 \ln \mathcal{L}$ (or $\Delta\chi^2$) objective functions, mapping highly complex multi-dimensional parameter spaces. Built for High-Performance Computing (HPC) environments, it ensures robust mathematical convergence even in the presence of steep gradients and disconnected topological phase spaces.

## Salient Features
* **Physics-Agnostic Design**: Decouples the mathematical likelihood scanning engine from your specific physics simulation.
* **Massive Parallelization**: Exploits Python's `multiprocessing.Pool` across grid meshes with intelligent OpenBLAS/MKL thread suppression to prevent node deadlocks.
* **Numba JIT Compilation**: Core physics kernels are pre-compiled into C-speed machine code, with automatic graceful fallbacks to standard Python if Numba is unavailable.
* **Autonomous Topology Repair**: Built-in, self-healing algorithms detect and recalculate artificial numerical spikes, slope discontinuities, and gradient cliffs caused by `iminuit` stalling.
* **Adaptive Mesh Refinement (AMR)**: Generates high-density interpolated meshes using monotonic PCHIP splines from sparse internal evaluations.
* **MCMC Warm Start**: Integrates `emcee` to stochastically map the global minimum basin before migrating to gradient-based minimization.
* **Empirical Prior Injection**: Automatically loads external 1D/2D data matrices, wrapping them in continuous multi-dimensional splines.
* **Dynamic Overrides**: Full parameter priority hierarchy allowing seamless transitions between JSON configuration payloads and terminal CLI overrides.

---

## Quick Start

### 1. Execution
To run the standard baseline execution using the built-in mathematical test model, run the codebase as a module from the root directory:
```bash
python -m src.main --config_file config/myconfig.json
```

### 2. Help Menu & Flags
You can view every possible configuration parameter, flag, and their default fallback values by calling the help menu:
```bash
python -m src.main -h
```

### 3. The Built-In Example
The framework ships with a lightweight, 3-parameter physics kernel located in `src/physics/kernel.py`:
$$ f(\alpha, \beta, \gamma) = \alpha^2 + 2\beta - \sin(\gamma) + \sum(\text{static\_data}) $$
The scanner will attempt to find the optimal global fit against an observation of `10.0`, while respecting the bounds and external 2D prior penalties mapped in `config/myconfig.json`.

### 4. Outputs
All execution artifacts are saved to the directory defined by the `--output_dir` flag (defaults to `results/` if unassigned).
* **`results/info/`**: Contains raw JSON coordinate arrays for every 1D and 2D parameter scan, as well as `frequentist_results.json` containing the extracted $1\sigma$, $2\sigma$, and $3\sigma$ intervals.
* **`results/plots/`**: Contains publication-ready `.pdf` graphics:
  * `frequentist_summary_1d.pdf`: Multi-panel 1D $\Delta\chi^2$ plots.
  * `frequentist_corner_2d.pdf`: Nested multi-dimensional contour matrices mapping phase spaces.

---

## File Structure

The framework is strictly modularized into functional namespaces.
```text
FrequentistAnalysis/
├── config/
│   └── myconfig.json            # Example configuration payload
├── results/                     # Generated automatically during execution
├── src/
│   ├── main.py                  # CLI entry point, argument parsing, orchestration
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # JSON parsing, comment stripping, unit conversions
│   │   ├── defaults.py          # Global default dictionaries and CLI fallbacks
│   │   └── likelihood.py        # GenericLikelihoodWrapper, empirical prior generation
│   ├── physics/
│   │   ├── __init__.py
│   │   └── kernel.py            # The ONLY file you need to modify! Houses your math.
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── optimizers.py        # Minuit descents, MCMC warm-starts, sanity checks
│   │   └── scanner.py           # 1D/2D parallel multiprocessing mesh grid iterators
│   ├── topology/
│   │   ├── __init__.py
│   │   └── repair.py            # Outlier, continuity, and gradient defect repair loops
│   └── analysis/
│       ├── __init__.py
│       ├── statistics.py        # Spline-root extraction for confidence intervals
│       └── plotting.py          # matplotlib 1D/2D contour generation routines
└── README.md
```

---

## The Analysis Workflow

### Likelihood Function
The core mathematical objective of the code is minimizing the $\Delta\chi^2$ space.
1. **Instantiation**: The `GenericLikelihoodWrapper` (in `src/core/likelihood.py`) reads the JSON configurations, establishes initial coordinate dictionaries, and generates multi-dimensional empirical penalty splines.
2. **Minimization**: The wrapper translates dynamic python dictionaries into strict `numpy` arrays.
3. **Execution**: The optimized float array is handed to `compute_user_model_single` (in `src/physics/kernel.py`), which calculates the physical prediction.
4. **Evaluation**: The wrapper computes the residual between the prediction and the target observation, adds any bounding prior penalties, and returns the objective scalar to Minuit.

*Recommendation:* Ensure that your physical kernel inside `compute_user_model_single` is strictly vectorized using standard `numpy` operations. Avoid python lists, raw `for` loops, or complex external class instantiations inside this function, as it will break the Numba JIT compiler and force a massive CPU bottleneck.

### Repairs
`iminuit` gradient descents often fail when navigating flat plateaus, steep cliffs, or highly disconnected islands, registering artificial $\Delta\chi^2$ jumps. The `src/topology/repair.py` loops execute autonomously after a grid scan completes:
* **Spike/Outlier Repair**: Scans the 1D/2D grids for isolated coordinate jumps. If found, it injects the nuisance parameter coordinates from the nearest successful neighbor and forces a re-minimization.
* **Continuity Repair**: Evaluates the second derivative of the scanned curves. If a jagged elbow or unphysical slope change is detected, it utilizes a Nelder-Mead simplex polish to bridge the gap.
* **Gradient Repair**: Fixes unphysical boundary walls by enforcing strict threshold drops relative to physical confidence boundaries.

### Output Files and Plotting
Outputs are highly modular. 
* **1D Intervals**: The code uses `UnivariateSpline` to extract roots across the $1\sigma$, $2\sigma$, and $3\sigma$ confidence limits. If `merge_disconnected_islands` is active, it bridges non-continuous limit bounds.
* **2D Contours**: Using `gaussian_filter`, the raw 2D $N \times N$ matrices are smoothed and projected via `pcolormesh`. 
If a 1D scan crashes or is intentionally skipped, the plotting algorithm is smart enough to autonomously extract and marginalize a ghost 1D curve from a populated 2D contour grid.

---

## Adapting the code to your project

The framework is designed so that you only need to interact with two files to inject your own physics: `src/physics/kernel.py` and `src/core/likelihood.py`.

1. **Update your variable names & labels**
Open `src/physics/kernel.py` and replace `LATEX_LABELS`.
```python
LATEX_LABELS = {
    "mass": r"M_{\chi}",
    "coupling": r"g_{X}",
    "phase": r"\phi"
}
```

2. **Update your Physics Kernel**
In `src/physics/kernel.py`, replace `compute_user_model_single`. Ensure it accepts a 1D array of floats (`params`) matching the strict sequence of your model.
```python
@njit
def compute_user_model_single(params, static_data):
    # params[0] = mass, params[1] = coupling, params[2] = phase
    theoretical_flux = (params[0] * params[1]) / np.cos(params[2])
    return theoretical_flux * static_data[0]
```

3. **Update the Likelihood Sequence**
Open `src/core/likelihood.py` and modify the `__init__` sequence inside `GenericLikelihoodWrapper`. Ensure `self.all_params` matches the exact string names of your parameters, and matches the array index order you defined in Step 2.
```python
self.all_params = ['mass', 'coupling', 'phase'] 
```
*Note:* You can also update `self.static_data` and `self.observed_data` inside this block to load real external telescope, detector, or statistical observation data.

---

## Setting Parameters

To maximize flexibility and reproducibility, the scanner utilizes a strict cascading priority hierarchy for parameters:
1. **CLI Arguments (Highest Priority)**: Explicit terminal flags (e.g., `--num_cores 16`) will forcefully override any other configuration.
2. **JSON Configuration File**: Values defined in `myconfig.json` map the core physical parameters and boundaries.
3. **Default Fallbacks (Lowest Priority)**: Any parameter missing from both the CLI and the JSON will safely default to the values hardcoded in `src/core/defaults.py`.

*Best Practice:* Always construct a dedicated JSON configuration file for a specific physics scenario to ensure your grid bounds, start values, and step sizes remain perfectly reproducible.

---

## The Config File

### Contents
Here is the baseline structure of the JSON payload (`myconfig.json`). Every feature can be toggled here.
```json
{
    "_comment_global": "Master config for generic frequentist profile likelihood scan.",
    "experiment": "generic_project_name",
    "output_dir": "../results/",
    
    "_comment_scan_settings": "Core configuration for 1D and 2D profile likelihood execution.",
    "scan_settings": {
        "num_cores": 64,
        "use_checkpointing": true,
        "skip_sanity_check": false,
        "merge_disconnected_islands": true,
        "island_merge_tolerance": 0.1,
        "enforce_spline_monotonicity": false,
        "n_steps": 40,
        "scan_range_sigma": 4.0,
        "force_rigid_bounds": false,
        "use_n_points_internal": true,
        "n_points_internal": 100,
        "scan_2d": true,
        "scan_2d_only": false,
        "n_steps_2d": 20,
        "scan_range_sigma_2d": 3.5,
        "use_explicit_step_sizes": true,
        "use_linear_interp": false,
        "use_fuzzy_cache": false,
        "clip_contour_smearing": false,
        "smooth_marginalized_1d": false,
        "save_nuisance_evolution": false,
        "log_scan_parameters": [],
        "target_subset_1d": [],
        "target_subset_2d": []
    },
    
    "_comment_repair": "Settings for the topological defect repair loops.",
    "repair_settings": {
        "disable_outlier_repair": false,
        "use_simplex_polish": true,
        "minuit_retries": 3,
        "spike_threshold": 0.2,
        "continuity_threshold": 1.0,
        "gradient_threshold": 50.0,
        "max_repair_passes": 10
    },
    
    "_comment_steps": "Initial step sizes Minuit uses to calculate gradients.",
    "step_sizes": {
        "param1": 0.01,
        "param2": 0.1
    },
    
    "_comment_limits": "Hard mathematical walls Minuit cannot cross. Format: [min, max]. Use null for unbounded.",
    "parameter_limits": {
        "param1": [-10.0, 10.0],
        "param2": [0.0, 100.0]
    },
    
    "_comment_fixed": "Parameters locked permanently in place. Can include angular units.",
    "fixed_params": {
        "param3": 1.5,
        "param4": [90.0, "degrees"]
    },
    
    "_comment_starts": "Initial guesses for floating parameters prior to global minimization.",
    "start_values": {
        "param1": 0.0,
        "param2": 5.0
    },
    
    "_comment_fallbacks": "Gaussian penalty priors applied to parameters not restricted by an input file. Format: ['gauss', mu, sigma, optional_units].",
    "fallbacks": {
        "param1": ["gauss", 0.0, 1.0],
        "param4": ["gauss", 180.0, 15.0, "degrees"]
    },
    
    "_comment_coupled_priors": "Define multi-parameter penalties here or implement dynamically in `_compute_user_coupled_priors`.",
    "coupled_priors": {},
    
    "_comment_asimov": "Parameter overrides used exclusively to generate the mock observation payload when the --asimov flag is active.",
    "asimov_injections": {
        "param1": 0.0
    },
    
    "_comment_inputs": "Paths to empirical 1D/2D chi2 grid files to act as dynamic prior penalties.",
    "input_configs": [
        {
            "file": "prior_data/example_2d_prior.txt",
            "type": "2D",
            "params": ["param1", "param4"],
            "units": ["none", "degrees"]
        }
    ],
    
    "_comment_fallback_setting": "If true, treats values outside prior grid bounds as infinite. If false, extrapolates via PCHIP/nearest-neighbor.",
    "use_quadratic_fallback": false,
    
    "_comment_confidence": "Delta chi2 thresholds used to draw boundaries and extract formal uncertainties.",
    "confidence_thresholds": {
        "1sigma": 1.0, 
        "2sigma": 4.0, 
        "3sigma": 9.0,
        "1sigma_2d": 2.30, 
        "2sigma_2d": 4.61, 
        "3sigma_2d": 11.83
    }
}
```

### Writing your own config file
To adapt the generic JSON to your physics project, you must:
* Update `output_dir` to point to a safe extraction folder.
* Rename the parameter strings under `step_sizes`, `parameter_limits`, `fixed_params`, and `start_values` to exactly match the variable strings you assigned in `self.all_params`.
* Define `parameter_limits`. Note that if a parameter is highly bounded near zero, you should add its string key to `"log_scan_parameters": ["your_param"]` to utilize high-resolution geometric mesh spacing.
* Configure `input_configs` by pointing `"file"` directly to any external empirical text files governing constraints. 

---

## Useful Recipes

### Only perform 1D Scans
If 2D nested grids are taking too long, or you simply want to extract isolated confidence intervals, disable the 2D contour generation via CLI:
```bash
python -m src.main --config_file myconfig.json --no_scan_2d
```

### Only perform 2D Scans
If you've already completed 1D bounds extractions, or you only care about topological contour intersections, you can bypass the 1D logic entirely:
```bash
python -m src.main --config_file myconfig.json --scan_2d_only
```

### My code is taking too long in the repairs
If the multi-dimensional phase space is chaotic, the repair loops might trigger hundreds of times, causing the execution to stall on a single parameter point. You can ease the repair strictness dynamically:
```bash
# Reduce the number of times the repair loop sweeps the grid
python -m src.main --config_file myconfig.json --max_repair_passes 2

# Disable the expensive Nelder-Mead "polish" prior to gradient descent
python -m src.main --config_file myconfig.json --no_use_simplex_polish

# Disable repairs entirely (WARNING: Results may contain artificial jagged spikes)
python -m src.main --config_file myconfig.json --disable_outlier_repair
```

### Smoothening Marginalized Results
If the 2D contour grid is low-resolution, and the scanner is forced to marginalize the 1D profile directly from the Z-axis array, the resulting 1D plot might appear jagged. You can force the graphics engine to apply a PCHIP smoothing spline to the output:
```bash
python -m src.main --config_file myconfig.json --smooth_marginalized_1d
```

### Avoiding Local Minimum Traps
If your physical model features multiple isolated basins (local minimums), `iminuit` might lock into a local valley and fail to find the true objective baseline. You can use the `emcee` stochastic walk sequence to aggressively map the space before profiling:
```bash
python -m src.main --config_file myconfig.json --use_mcmc_warm_start
```

### Asimov Sensitivities
If you are generating expected sensitivity projections rather than fitting against real observational data, you can force the scanner to generate a "perfect" mock dataset utilizing injected null-hypothesis parameters defined in `asimov_injections`:
```bash
python -m src.main --config_file myconfig.json --asimov
```

---

## Authorship and License
**Author**: Mauricio Bustamante (mbustamante@gmail.com)

**License**: This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007. See the `LICENSE` file in the repository root for details.