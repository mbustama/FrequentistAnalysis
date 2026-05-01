"""
main.py

The central orchestration module. Parses user commands, enforces overriding logic paths,
integrates execution frameworks, and structures overall operational deployment loops.
"""

import os
import sys

# Add the parent directory of 'src' to the Python path to resolve absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import multiprocessing
import numpy as np
from iminuit import Minuit
from multiprocessing import Pool, cpu_count

# ==============================================================================
# 1. HPC CONFIGURATION & DEADLOCK PREVENTION
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Import explicit mapping pathways and logic environments 
from src.core.defaults import DEFAULT_SCAN_SETTINGS, DEFAULT_REPAIR_SETTINGS
from src.core.config import load_config
from src.core.likelihood import GenericLikelihoodWrapper
from src.engines.optimizers import estimate_param_names, estimate_start_values, run_mcmc_warm_start, perform_sanity_check
from src.engines.scanner import scan_parameter_parallel, scan_pair_parallel
from src.topology.repair import repair_scan_outliers, repair_scan_continuity, repair_steep_gradients, repair_2d_scan_outliers
from src.analysis.statistics import calculate_intervals
from src.analysis.plotting import plot_1d_analysis, plot_2d_analysis

def run_2d_analysis(args, m, param_names, limits, func, global_min_chi2, scan_settings, confidence_cfg, repair_settings, pool=None, num_cores=1):
    """Executes the secondary operational branch routing parameter permutations across coupled multidimensional limits."""
    print("\n" + "="*60 + "\nSTARTING 2D PROFILE SCANS\n" + "="*60)
    surfaces_2d = {}
    use_checkpointing = scan_settings.get("use_checkpointing", True)
    
    target_subset = scan_settings.get("target_subset_2d", [])
    scan_param_names = target_subset if target_subset else param_names
    
    n_params = len(scan_param_names)
    for i in range(n_params):
        for j in range(n_params):
            if i > j: 
                param_y, param_x = scan_param_names[i], scan_param_names[j]
                print(f"  -> Scanning 2D: {param_y} vs {param_x}...")
                
                info_path = os.path.join(args.output_dir, 'info', f"profile_2d_{param_x}_vs_{param_y}.json")
                checkpoint_loaded = False
                
                if use_checkpointing and os.path.exists(info_path):
                    try:
                        with open(info_path, 'r') as f: data = json.load(f)
                        x_grid, y_grid, Z_chi2 = np.array(data["x"]), np.array(data["y"]), np.array(data["z"])
                        param_grid = data.get("param_grid", None) 
                        checkpoint_loaded = True
                        print(f"  -> [Checkpoint] Loading existing 2D scan for {param_y} vs {param_x} from disk...")
                    except Exception as e:
                        print(f"  -> [Checkpoint] WARNING: Corrupted JSON detected ({e}). Deleting {info_path} and re-running scan...")
                        try: os.remove(info_path)
                        except OSError: pass
                        
                if not checkpoint_loaded:
                    x_grid, y_grid, Z_chi2, param_grid = scan_pair_parallel(
                        m, param_x, param_y, limits, func, param_names,
                        n_steps=scan_settings.get("n_steps_2d", 20), 
                        scan_range_sigma=scan_settings.get("scan_range_sigma_2d", 3.5), 
                        minuit_retries=repair_settings.get("minuit_retries", 3), 
                        force_rigid_bounds=scan_settings.get("force_rigid_bounds", False),
                        pool=pool, num_cores=num_cores
                    )
                    
                    if not repair_settings.get("disable_outlier_repair", False):
                        x_grid, y_grid, Z_chi2 = repair_2d_scan_outliers(
                            x_grid, y_grid, Z_chi2, param_grid, (param_x, param_y), func, limits, param_names, dict(zip(m.parameters, m.errors)),
                            use_simplex=repair_settings.get("use_simplex_polish", True), threshold=repair_settings.get("spike_threshold", 0.2),
                            max_passes=repair_settings.get("max_repair_passes", 10), minuit_retries=repair_settings.get("minuit_retries", 3),
                            pool=pool, num_cores=num_cores
                        )
                    
                    os.makedirs(os.path.join(args.output_dir, 'info'), exist_ok=True)
                    
                    save_nuisance_evolution = scan_settings.get("save_nuisance_evolution", False)
                    json_payload = {"x": x_grid.tolist(), "y": y_grid.tolist(), "z": Z_chi2.tolist(), "param_grid": param_grid}
                    
                    if save_nuisance_evolution and param_grid:
                        nuisance_data = {}
                        for k in m.parameters:
                            nuisance_data[k] = [[param_grid[row][col].get(k, None) if isinstance(param_grid[row][col], dict) and param_grid[row][col] else None for col in range(len(param_grid[0]))] for row in range(len(param_grid))]
                        json_payload["nuisance_evolution"] = nuisance_data
                        
                    with open(info_path, 'w') as f: 
                        json.dump(json_payload, f, indent=4)

                current_scan_min = np.nanmin(Z_chi2)
                if current_scan_min < 1e5 and current_scan_min < global_min_chi2 - 1e-4:
                    diff = global_min_chi2 - current_scan_min
                    print(f"  [WARNING] Better minimum found during 2D scan of {param_y} vs {param_x}! Diff: {diff:.4f}")
                    global_min_chi2 = current_scan_min
                    
                    if param_grid:
                        best_idx_flat = int(np.nanargmin(Z_chi2))
                        best_i, best_j = np.unravel_index(best_idx_flat, Z_chi2.shape)
                        new_best_params = param_grid[best_i][best_j]
                        if new_best_params:
                            for k_p, v_p in new_best_params.items():
                                m.values[k_p] = v_p

                # [Fix #1]: Extract baseline directly from the evaluated numerical plane
                norm_min_2d = min(global_min_chi2, np.nanmin(Z_chi2))
                if np.isinf(global_min_chi2) and norm_min_2d < 1e5:
                    print(f"  [Fix #1] Recovered valid 2D baseline minimum from grid: {norm_min_2d:.4f}")
                Z_dchi2 = np.nan_to_num(Z_chi2 - norm_min_2d, nan=1e9)
                Z_dchi2[Z_dchi2 < 0] = 0
                surfaces_2d[f"{param_y}_vs_{param_x}"] = {"x": x_grid.tolist(), "y": y_grid.tolist(), "z": Z_dchi2.tolist()}
                
    with open(os.path.join(args.output_dir, 'info', "profile_scan_2d_data.json"), 'w') as f: json.dump(surfaces_2d, f, indent=4)
    plot_2d_analysis(args.output_dir, scan_param_names, confidence_cfg, global_min_chi2, scan_settings)

def main():
    """Initializes sequence parsing logic invoking systematic execution pathways across operational topologies."""
    parser = argparse.ArgumentParser(
        description="Generic Frequentist Profile Likelihood Scanner.\n\n"
                    "Order of Priority for Execution Parameters:\n"
                    "1. CLI Arguments (Highest Priority)\n"
                    "2. JSON Configuration File\n"
                    "3. Default Fallbacks (Lowest Priority)\n",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("--output_dir", type=str, default="./results", help="Target directory path for generated graphics and serialized JSON outputs.")
    parser.add_argument("--config_file", type=str, default="", help="Active target configuration payload defining empirical variables.")
    parser.add_argument("--use_mcmc_warm_start", action="store_true", help="Invokes Emcee randomized basin evaluation prior to gradient profiling sequences.")
    parser.add_argument("--asimov", action="store_true", help="Configures baseline validation payload assessing exact null-hypothesis physics sensitivity boundaries.")
    
    scan_group = parser.add_argument_group('Scan Settings (CLI Overrides)')
    for k, v in DEFAULT_SCAN_SETTINGS.items():
        if isinstance(v, bool):
            scan_group.add_argument(f"--{k}", action='store_true', default=v, help=f"Enable {k} (Default: {v})")
            scan_group.add_argument(f"--no_{k}", dest=k, action='store_false', help=f"Disable {k}")
        elif isinstance(v, (int, float, str)):
            scan_group.add_argument(f"--{k}", type=type(v), default=v, help=f"Default numerical value: {v}")
        elif isinstance(v, list):
            scan_group.add_argument(f"--{k}", nargs='*', type=str, default=v, help=f"List sequence parameter. Default: {v}")

    repair_group = parser.add_argument_group('Repair Settings (CLI Overrides)')
    for k, v in DEFAULT_REPAIR_SETTINGS.items():
        if isinstance(v, bool):
            repair_group.add_argument(f"--{k}", action='store_true', default=v, help=f"Enable {k} (Default: {v})")
            repair_group.add_argument(f"--no_{k}", dest=k, action='store_false', help=f"Disable {k}")
        elif isinstance(v, (int, float, str)):
            repair_group.add_argument(f"--{k}", type=type(v), default=v, help=f"Default numerical value: {v}")

    args = parser.parse_args()

    # Base dictionary injection loading sequence
    config = load_config(args.config_file)
    if not config.get("output_dir"): 
        config["output_dir"] = args.output_dir
    os.makedirs(config["output_dir"], exist_ok=True)

    args.output_dir = config["output_dir"]
    
    # Establish precise user overrides circumventing fallback parameters 
    cli_provided_args = {arg.lstrip('-').split('=')[0] for arg in sys.argv[1:] if arg.startswith('--')}
    cli_keys = set()
    for k in cli_provided_args:
        if k.startswith('no_') and k[3:] in DEFAULT_SCAN_SETTINGS.keys() | DEFAULT_REPAIR_SETTINGS.keys():
            cli_keys.add(k[3:])
        else:
            cli_keys.add(k)
            
    if "scan_settings" not in config: config["scan_settings"] = {}
    for k, def_val in DEFAULT_SCAN_SETTINGS.items():
        cli_val = getattr(args, k)
        if k in cli_keys:
            if k in config["scan_settings"] and config["scan_settings"][k] != cli_val:
                print(f"  [CLI Override] Executing override on 'scan_settings' element '{k}' modifying {config['scan_settings'][k]} down to {cli_val}.")
            config["scan_settings"][k] = cli_val
        elif k not in config["scan_settings"]:
            config["scan_settings"][k] = def_val

    if "repair_settings" not in config: config["repair_settings"] = {}
    for k, def_val in DEFAULT_REPAIR_SETTINGS.items():
        cli_val = getattr(args, k)
        if k in cli_keys:
            if k in config["repair_settings"] and config["repair_settings"][k] != cli_val:
                print(f"  [CLI Override] Executing override on 'repair_settings' element '{k}' modifying {config['repair_settings'][k]} down to {cli_val}.")
            config["repair_settings"][k] = cli_val
        elif k not in config["repair_settings"]:
            config["repair_settings"][k] = def_val
            
    scan_settings = config.get("scan_settings", {})
    req_cores = scan_settings.get("num_cores", cpu_count())
    actual_cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
    num_cores = max(1, min(req_cores, actual_cores))
    print(f"\nUsing up to {num_cores} CPU core(s) (Requested: {req_cores}, Available: {actual_cores}).")

    param_names = estimate_param_names(config)
    start_vals = estimate_start_values(param_names, config)
    
    if args.asimov:
        asimov_injections = config.get("asimov_injections", {})
        if asimov_injections:
            print(f"\n  [Asimov] Initialization override: Forcing start values to mock baseline {asimov_injections}")
            start_vals.update({k: v for k, v in asimov_injections.items() if k in param_names})
            
    func = GenericLikelihoodWrapper(config, param_names, asimov=args.asimov)
    limits = {p: tuple(config.get("parameter_limits", {}).get(p, (None, None))) for p in param_names}

    with Pool(processes=num_cores) as pool:
        if args.use_mcmc_warm_start:
            start_vals = run_mcmc_warm_start(func, param_names, start_vals, limits, pool=pool)
            
        m = Minuit(func, name=param_names, **start_vals)
        m.errordef = Minuit.LEAST_SQUARES 
        
        if scan_settings.get("use_explicit_step_sizes", False):
            for p in param_names:
                if p in config.get("step_sizes", {}): m.errors[p] = config["step_sizes"][p]
                    
        for p, lim in limits.items(): m.limits[p] = lim
        
        try:
            m.simplex(); m.migrad()
            global_min_chi2 = m.fval
            print(f"Global Best Fit Chi2: {global_min_chi2:.4f}")
        except Exception as e:
            print(f"  [WARNING] Initial global minimization failed ({e}). Proceeding to grid scans; repair loops will attempt to find the true minimum organically.")
            global_min_chi2 = np.inf
        
        if not scan_settings.get("skip_sanity_check", False):
            perform_sanity_check(func, m)
        
        use_checkpointing = scan_settings.get("use_checkpointing", True)
        repair_cfg = config.get("repair_settings", {})
        
        final_results_dict = {}
        freq_results_path = os.path.join(config["output_dir"], 'info', 'frequentist_results.json')
        if use_checkpointing and os.path.exists(freq_results_path):
            try:
                with open(freq_results_path, 'r') as f:
                    final_results_dict = json.load(f)
            except Exception: pass
        
        scan_2d_only = scan_settings.get("scan_2d_only", False)
        
        if not scan_2d_only:
            target_subset_1d = scan_settings.get("target_subset_1d", [])
            scan_params_1d = target_subset_1d if target_subset_1d else param_names
            log_params = scan_settings.get("log_scan_parameters", [])
            
            for p in scan_params_1d:
                print(f"\nScanning {p}...")
                info_path = os.path.join(config["output_dir"], 'info', f"profile_1d_{p}.json")
                use_log_scan = p in log_params
                checkpoint_loaded = False
                
                if use_checkpointing and os.path.exists(info_path):
                    try:
                        with open(info_path, 'r') as f: data = json.load(f)
                        x, y = np.array(data["x"]), np.array(data["y"])
                        hist = data.get("param_hist", None)
                        checkpoint_loaded = True
                        print(f"  -> [Checkpoint] Loading existing 1D scan for {p} from disk...")
                    except Exception as e:
                        print(f"  -> [Checkpoint] WARNING: Corrupted JSON detected ({e}). Deleting {info_path} and re-running scan...")
                        try: os.remove(info_path)
                        except OSError: pass
                
                if not checkpoint_loaded:
                    x, y, hist = scan_parameter_parallel(m, p, m.values[p], max(m.errors[p], 0.1), func, limits, param_names, config["output_dir"], 
                                                         n_steps=scan_settings.get("n_steps", 40), use_n_points_internal=scan_settings.get("use_n_points_internal", False),
                                                         n_points_internal=scan_settings.get("n_points_internal", 100), scan_range_sigma=scan_settings.get("scan_range_sigma", 4.0),
                                                         use_log_scan=use_log_scan, force_rigid_bounds=scan_settings.get("force_rigid_bounds", False), pool=pool, num_cores=num_cores)
                    
                    if not repair_cfg.get("disable_outlier_repair", False):
                        x, y = repair_scan_outliers(x, y, hist, p, func, limits, param_names, dict(zip(m.parameters, m.errors)), 
                                                    threshold=repair_cfg.get("spike_threshold", 0.2), max_passes=repair_cfg.get("max_repair_passes", 10),
                                                    use_simplex=repair_cfg.get("use_simplex_polish", True), minuit_retries=repair_cfg.get("minuit_retries", 3), pool=pool, num_cores=num_cores)
                        x, y = repair_scan_continuity(x, y, hist, p, func, limits, param_names, dict(zip(m.parameters, m.errors)), 
                                                      continuity_threshold=repair_cfg.get("continuity_threshold", 1.0), max_passes=repair_cfg.get("max_repair_passes", 10),
                                                      use_simplex=repair_cfg.get("use_simplex_polish", True), minuit_retries=repair_cfg.get("minuit_retries", 3), pool=pool, num_cores=num_cores)
                        x, y = repair_steep_gradients(x, y, hist, p, func, limits, param_names, dict(zip(m.parameters, m.errors)), 
                                                      threshold=repair_cfg.get("gradient_threshold", 50.0), max_passes=repair_cfg.get("max_repair_passes", 10),
                                                      use_simplex=repair_cfg.get("use_simplex_polish", True), minuit_retries=repair_cfg.get("minuit_retries", 3), pool=pool, num_cores=num_cores)
                    
                    os.makedirs(os.path.join(config["output_dir"], 'info'), exist_ok=True)
                    
                    save_nuisance_evolution = scan_settings.get("save_nuisance_evolution", False)
                    json_payload = {"x": x.tolist(), "y": y.tolist(), "param_hist": hist}
                    
                    if save_nuisance_evolution and hist:
                        json_payload["nuisance_evolution"] = {k: [step.get(k, None) if isinstance(step, dict) and step else None for step in hist] for k in m.parameters}
                        
                    with open(info_path, 'w') as f: 
                        json.dump(json_payload, f)

                current_scan_min = np.nanmin(y)
                if current_scan_min < 1e5 and current_scan_min < global_min_chi2 - 1e-4:
                    diff = global_min_chi2 - current_scan_min
                    print(f"  [WARNING] Better minimum found during scan of {p}! Diff: {diff:.4f}")
                    global_min_chi2 = current_scan_min
                    
                    if hist:
                        best_idx = int(np.nanargmin(y))
                        new_best_params = hist[best_idx]
                        if new_best_params:
                            for k_p, v_p in new_best_params.items(): m.values[k_p] = v_p
                
                # [Fix #1]: Extract baseline exclusively from evaluating numerical iterations.
                norm_min = min(global_min_chi2, np.nanmin(y))
                if np.isinf(global_min_chi2) and norm_min < 1e5:
                    print(f"  [Fix #1] Recovered valid 1D baseline minimum from grid: {norm_min:.4f}")
                dchi2 = np.nan_to_num(y - norm_min, nan=1e9)
                dchi2[dchi2 < 0] = 0
                
                final_results_dict[p] = calculate_intervals(
                    x, dchi2, p, 
                    config.get("confidence_thresholds", {}), 
                    merge_gaps=scan_settings.get("merge_disconnected_islands", True), 
                    gap_tolerance=scan_settings.get("island_merge_tolerance", 0.1),
                    enforce_monotonicity=scan_settings.get("enforce_spline_monotonicity", False)
                )
                
                with open(os.path.join(config["output_dir"], 'info', 'frequentist_results.json'), 'w') as f: 
                    json.dump(final_results_dict, f, indent=4)
                
            plot_1d_analysis(config["output_dir"], scan_params_1d, config.get("confidence_thresholds", {}), global_min_chi2)
        
        # --- 2D Scans ---
        if scan_settings.get("scan_2d", False) or scan_2d_only:
            run_2d_analysis(args, m, param_names, limits, func, global_min_chi2, 
                            scan_settings, config.get("confidence_thresholds", {}), repair_cfg, 
                            pool=pool, num_cores=num_cores)

if __name__ == "__main__":
    try: multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main()