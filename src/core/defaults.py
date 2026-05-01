"""
defaults.py

Provides the core initialization dictionaries for the frequentist scanner. 
These variables construct the baseline logic pathways and form the foundation 
of the dynamic CLI override system.
"""

DEFAULT_SCAN_SETTINGS = {
    "num_cores": 64,
    "use_checkpointing": True,
    "skip_sanity_check": False,
    "merge_disconnected_islands": True,
    "island_merge_tolerance": 0.1,
    "enforce_spline_monotonicity": False,
    "n_steps": 40,
    "scan_range_sigma": 4.0,
    "force_rigid_bounds": False,
    "use_n_points_internal": True,
    "n_points_internal": 100,
    "scan_2d": True,
    "scan_2d_only": False,
    "n_steps_2d": 20,
    "scan_range_sigma_2d": 3.5,
    "use_explicit_step_sizes": True,
    "use_linear_interp": False,
    "use_fuzzy_cache": False,
    "clip_contour_smearing": False,
    "smooth_marginalized_1d": False,
    "save_nuisance_evolution": False,
    "log_scan_parameters": [],
    "target_subset_1d": [],
    "target_subset_2d": []
}

DEFAULT_REPAIR_SETTINGS = {
    "disable_outlier_repair": False,
    "use_simplex_polish": True,
    "minuit_retries": 3,
    "spike_threshold": 0.2,
    "continuity_threshold": 1.0,
    "gradient_threshold": 50.0,
    "max_repair_passes": 10
}