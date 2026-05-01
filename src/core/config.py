"""
config.py

Handles the parsing, serialization, and initial processing of the JSON payload.
Responsible for stripping comments and converting mathematical constraints (like degrees).

Example config file (myconfig.json):

{
    "_comment_global": "Master config for generic frequentist profile likelihood scan.",
    "experiment": "generic_project_name",
    "output_dir": "../results/",
    
    "_comment_scan_settings": "Core configuration for 1D and 2D profile likelihood execution.",
    "scan_settings": {
        "_comment_num_cores": "Explicit override for multiprocessing CPU cores.",
        "num_cores": 64,
        "_comment_use_checkpointing": "If true, skips parameters with existing json files in the info directory.",
        "use_checkpointing": true,
        "_comment_skip_sanity_check": "If true, bypasses the initial 10% parameter perturbation test.",
        "skip_sanity_check": false,
        "_comment_merge_disconnected_islands": "If true, treats disjoint 1D limits as continuous if the delta-chi2 gap is small.",
        "merge_disconnected_islands": true,
        "_comment_island_merge_tolerance": "Delta-chi2 threshold allowed when bridging disjoint parameter islands.",
        "island_merge_tolerance": 0.1,
        "_comment_enforce_spline_monotonicity": "If true, explicitly sorts X arrays before interval extraction to prevent UnivariateSpline crashes on reversed bounds.",
        "enforce_spline_monotonicity": false,
        "_comment_n_steps": "Number of final interpolated grid points for 1D parameter profile scans.",
        "n_steps": 40,
        "_comment_scan_range_sigma": "How far to scan left/right of the minimum in units of parameter sigma/error.",
        "scan_range_sigma": 4.0,
        "_comment_force_rigid_bounds": "If true, 1D/2D scans ignore local sigma and strictly scan absolute parameter_limits.",
        "force_rigid_bounds": false,
        "_comment_use_n_points_internal": "Adaptive Mesh Refinement. If true, evaluates n_points_internal and interpolates to n_steps.",
        "use_n_points_internal": true,
        "_comment_n_points_internal": "Number of coarse internal points to calculate before PCHIP interpolation.",
        "n_points_internal": 100,
        "_comment_scan_2d": "If true, performs nested 2D contour profile scans after 1D scans finish.",
        "scan_2d": true,
        "_comment_scan_2d_only": "If true, completely skips 1D scans and ONLY runs 2D scans.",
        "scan_2d_only": false,
        "_comment_n_steps_2d": "Grid resolution strictly for 2D profile contour mapping.",
        "n_steps_2d": 20,
        "_comment_scan_range_sigma_2d": "How far to scan left/right of the minimum during 2D profiling.",
        "scan_range_sigma_2d": 3.5,
        "_comment_use_explicit_step_sizes": "If true, Minuit uses the 'step_sizes' block to initialize gradient derivatives.",
        "use_explicit_step_sizes": true,
        "_comment_use_linear_interp": "If true, uses linear interpolation for 1D prior files instead of PCHIP cubic splines.",
        "use_linear_interp": false,
        "_comment_use_fuzzy_cache": "If true, rounds objective function inputs slightly to maximize cache hit rates.",
        "use_fuzzy_cache": false,
        "_comment_log_scan_parameters": "List of variables that should use geometric (logarithmic) spacing during 1D scans to resolve bounds near 0.",
        "log_scan_parameters": [],
        "_comment_target_subset_1d": "If defined, ONLY these parameters will be processed during the 1D scan loop.",
        "target_subset_1d": [],
        "_comment_target_subset_2d": "If defined, ONLY pairs within this list will be processed during the 2D scan loop.",
        "target_subset_2d": [],
        "_comment_clip_contour_smearing": "If true, clips massive delta-chi2 values before gaussian smoothing to prevent contour distortion.",
        "clip_contour_smearing": false,
        "_comment_smooth_marginalized_1d": "If true, uses spline interpolation to smooth out the jagged 1D curves marginalized from 2D contour grids.",
        "smooth_marginalized_1d": false,
        "_comment_save_nuisance_evolution": "If true, explicitly formats and saves parameter evolution tracks for diagnostic plotting.",
        "save_nuisance_evolution": false
    },
    
    "_comment_repair": "Settings for the topological defect repair loops.",
    "repair_settings": {
        "_comment_disable_outlier_repair": "If true, disables all outlier, continuity, and gradient repair loops.",
        "disable_outlier_repair": false,
        "_comment_use_simplex_polish": "If true, runs Nelder-Mead simplex prior to migrad gradient descent during a repair.",
        "use_simplex_polish": true,
        "_comment_minuit_retries": "Number of times to kick a stuck parameter with gaussian random noise before failing.",
        "minuit_retries": 3,
        "_comment_spike_threshold": "Delta-chi2 jump magnitude required to classify a point as an isolated spike.",
        "spike_threshold": 0.2,
        "_comment_continuity_threshold": "Magnitude of slope change required to flag and repair a second-derivative discontinuity.",
        "continuity_threshold": 1.0,
        "_comment_gradient_threshold": "Delta-chi2 drop required to flag an unphysical gradient cliff.",
        "gradient_threshold": 50.0,
        "_comment_max_repair_passes": "Maximum number of times the repair loop will sweep the grid to clear cascading spikes.",
        "max_repair_passes": 10
    },
    
    "_comment_steps": "Initial step sizes Minuit uses to calculate gradients.",
    "step_sizes": {
        "param1": 0.01,
        "param2": 0.1
    },
    
    "_comment_limits": "Hard mathematical walls Minuit cannot cross.",
    "parameter_limits": {
        "param1": [-10.0, 10.0],
        "param2": [0.0, 100.0]
    },
    
    "_comment_fixed": "Parameters locked permanently in place.",
    "fixed_params": {
        "param3": 1.5,
        "param4": [90.0, "degrees"]
    },
    
    "_comment_starts": "Initial guesses for floating parameters.",
    "start_values": {
        "param1": 0.0,
        "param2": 5.0
    },
    
    "_comment_fallbacks": "Gaussian penalty priors applied to parameters not restricted by an input file.",
    "fallbacks": {
        "param1": ["gauss", 0.0, 1.0],
        "param4": ["gauss", 180.0, 15.0, "degrees"]
    },
    
    "_comment_coupled_priors": "Define multi-parameter penalties here or implement in _compute_user_coupled_priors.",
    "coupled_priors": {},
    
    "_comment_asimov": "Parameter overrides used exclusively to generate the mock null-hypothesis when --asimov is active.",
    "asimov_injections": {
        "param1": 0.0
    },
    
    "_comment_inputs": "Paths to 1D/2D chi2 grid files to act as dynamic prior penalties.",
    "input_configs": [
        {
            "file": "prior_data/example_2d_prior.txt",
            "type": "2D",
            "params": ["param1", "param4"],
            "units": ["none", "degrees"]
        }
    ],
    "use_quadratic_fallback": false,
    
    "_comment_confidence": "Delta chi2 thresholds used to draw boundaries on the 1D and 2D plots.",
    "confidence_thresholds": {
        "1sigma": 1.0, 
        "2sigma": 4.0, 
        "3sigma": 9.0,
        "1sigma_2d": 2.30, 
        "2sigma_2d": 4.61, 
        "3sigma_2d": 11.83
    }
}
"""

import os
import json
import numpy as np

def load_config(config_path):
    """
    Loads JSON configuration files, strips metadata comment lines, and executes 
    angular unit conversions on bounded variables.

    Parameters:
    - config_path (str): Filepath to the active JSON config.

    Returns:
    - dict: Cleaned and processed configuration dictionary. Defaults to {} if none exists.
    """
    if not config_path or not os.path.exists(config_path):
        return {}
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    def strip_comments(d):
        if isinstance(d, dict):
            return {k: strip_comments(v) for k, v in d.items() if not k.startswith("_comment")}
        elif isinstance(d, list):
            return [strip_comments(i) for i in d]
        return d
        
    config = strip_comments(config)
    
    raw_fixed = config.get('fixed_params', {})
    processed_fixed = {}
    for param, val_entry in raw_fixed.items():
        if isinstance(val_entry, list) and len(val_entry) >= 2:
            val = float(val_entry[0])
            unit = str(val_entry[1]).lower()
            processed_fixed[param] = np.radians(val) if 'degree' in unit else val
        else:
            processed_fixed[param] = float(val_entry)
    config['fixed_params'] = processed_fixed
    
    return config