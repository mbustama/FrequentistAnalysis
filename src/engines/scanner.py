"""
scanner.py

Coordinates the execution of parameter mesh grids utilizing High-Performance Computing
parallel processing architecture (Python Multiprocessing Pools).
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from src.engines.optimizers import minimize_point, minimize_point_2d

def _global_1d_scan_wrapper(task_args):
    """Exposes 1D tasks for generic mapping functions safely avoiding partial closures."""
    return minimize_point(*task_args)

def _global_2d_scan_wrapper(task_args):
    """Exposes 2D tasks for generic mapping functions safely avoiding partial closures."""
    return minimize_point_2d(*task_args)

def scan_parameter_parallel(m_global, param_name, start_val, sigma_guess, func, limits, param_names, output_dir, 
                            n_steps=40, use_n_points_internal=False, n_points_internal=100, scan_range_sigma=4.0, 
                            minuit_retries=3, use_log_scan=False, force_rigid_bounds=False, pool=None, num_cores=1):
    """
    Constructs distributed HPC tasks dynamically iterating bounds constraints and returns 
    synthesized 1D profiling vectors. Features intelligent mesh refinements.
    """
    if force_rigid_bounds and param_name in limits and limits[param_name][0] is not None and limits[param_name][1] is not None:
        scan_min, scan_max = limits[param_name][0], limits[param_name][1]
        print(f"  [1D Scan] Enforcing rigid bounds for {param_name}: [{scan_min}, {scan_max}]")
    else:
        calc_min = start_val - scan_range_sigma * sigma_guess
        calc_max = start_val + scan_range_sigma * sigma_guess
        lim_min = limits.get(param_name, [None, None])[0]
        lim_max = limits.get(param_name, [None, None])[1]
        
        scan_min = max(calc_min, lim_min) if lim_min is not None else calc_min
        scan_max = min(calc_max, lim_max) if lim_max is not None else calc_max
    
    if use_log_scan:
        scan_min = max(scan_min, 1e-30); scan_max = max(scan_max, 2e-30)
        base_grid = np.geomspace(scan_min, scan_max, n_steps)
        exec_grid = np.geomspace(scan_min, scan_max, n_points_internal) if use_n_points_internal else base_grid
    else:
        base_grid = np.linspace(scan_min, scan_max, n_steps)
        exec_grid = np.linspace(scan_min, scan_max, n_points_internal) if use_n_points_internal else base_grid

    global_start_values = dict(zip(m_global.parameters, m_global.values))
    global_errors = dict(zip(m_global.parameters, m_global.errors))
    tasks = [(v, param_name, global_start_values, global_errors, func, limits, param_names, False, minuit_retries) for v in exec_grid]
    
    chunksize = max(1, len(tasks) // (num_cores * 4)) if pool else 1
    if pool is None or len(tasks) < 5:
        results = [_global_1d_scan_wrapper(t) for t in tasks]
    else:
        try:
            from tqdm import tqdm
            results = list(tqdm(pool.imap(_global_1d_scan_wrapper, tasks, chunksize=chunksize), total=len(tasks), desc=f"  Progress"))
        except ImportError:
            results = pool.map(_global_1d_scan_wrapper, tasks, chunksize=chunksize)
            
    chi2_arr_exec = np.array([r[0] for r in results])
    param_hists_exec = [r[1] for r in results]

    if use_n_points_internal and n_points_internal < n_steps:
        valid_mask = chi2_arr_exec < 1e5
        if np.sum(valid_mask) > 1:
            pchip = PchipInterpolator(exec_grid[valid_mask], chi2_arr_exec[valid_mask])
            chi2_arr = np.maximum(pchip(base_grid), 0.0)
        else: chi2_arr = np.full(n_steps, np.inf)

        param_histories = [param_hists_exec[np.abs(exec_grid - x).argmin()] for x in base_grid]
        return base_grid, chi2_arr, param_histories
        
    return exec_grid, chi2_arr_exec, param_hists_exec

def scan_pair_parallel(m_global, param_x, param_y, limits, func, param_names, n_steps=20, 
                       scan_range_sigma=3.5, minuit_retries=3, force_rigid_bounds=False, pool=None, num_cores=1):
    """Iterates a distributed two-dimensional mesh matrix computing profile isolines contour projections."""
    sigma_x = m_global.errors[param_x] if m_global.errors[param_x] > 0 else 0.1
    sigma_y = m_global.errors[param_y] if m_global.errors[param_y] > 0 else 0.1
    val_x, val_y = m_global.values[param_x], m_global.values[param_y]
    
    if force_rigid_bounds and param_x in limits and limits[param_x][0] is not None and limits[param_x][1] is not None:
        min_x, max_x = limits[param_x][0], limits[param_x][1]
    else:
        calc_min_x = val_x - scan_range_sigma*sigma_x
        calc_max_x = val_x + scan_range_sigma*sigma_x
        lim_min_x = limits.get(param_x, [None, None])[0]
        lim_max_x = limits.get(param_x, [None, None])[1]
        min_x = max(calc_min_x, lim_min_x) if lim_min_x is not None else calc_min_x
        max_x = min(calc_max_x, lim_max_x) if lim_max_x is not None else calc_max_x
        
    if force_rigid_bounds and param_y in limits and limits[param_y][0] is not None and limits[param_y][1] is not None:
        min_y, max_y = limits[param_y][0], limits[param_y][1]
    else:
        calc_min_y = val_y - scan_range_sigma*sigma_y
        calc_max_y = val_y + scan_range_sigma*sigma_y
        lim_min_y = limits.get(param_y, [None, None])[0]
        lim_max_y = limits.get(param_y, [None, None])[1]
        min_y = max(calc_min_y, lim_min_y) if lim_min_y is not None else calc_min_y
        max_y = min(calc_max_y, lim_max_y) if lim_max_y is not None else calc_max_y
        
    x_grid = np.linspace(min_x, max_x, n_steps)
    y_grid = np.linspace(min_y, max_y, n_steps)
    tasks = [(x, y) for y in y_grid for x in x_grid]
    
    global_vals = dict(zip(m_global.parameters, m_global.values))
    global_errs = dict(zip(m_global.parameters, m_global.errors))
    scan_tasks = [(task, (param_x, param_y), global_vals, global_errs, func, limits, param_names, True, minuit_retries) for task in tasks]
                     
    n_tasks = len(scan_tasks)
    if pool is None or n_tasks < 5: 
        results = [_global_2d_scan_wrapper(t) for t in scan_tasks]
    else:
        chunksize = max(1, n_tasks // (num_cores * 4))
        try:
            from tqdm import tqdm
            results = list(tqdm(pool.imap(_global_2d_scan_wrapper, scan_tasks, chunksize=chunksize), total=n_tasks, desc=f"  2D Grid ({n_steps}x{n_steps})"))
        except ImportError:
            results = pool.map(_global_2d_scan_wrapper, scan_tasks, chunksize=chunksize)
            
    chi2_list, params_list = [r[0] for r in results], [r[1] for r in results]
    Z = np.array(chi2_list).reshape((n_steps, n_steps))
    param_grid = [[params_list[i*n_steps + j] for j in range(n_steps)] for i in range(n_steps)]
    return x_grid, y_grid, Z, param_grid