"""
Generic Frequentist Profile Likelihood Scanner

This script provides a highly optimized, parallelized frequentist analysis framework
for multi-parameter profile likelihood scanning. It is designed to be physics-agnostic.

Core Optimizations Included:
- MKL/OpenBLAS thread suppression to prevent multiprocessing deadlocks.
- Hardware-aware CPU clamping (respects Slurm and cgroups).
- Numba JIT compilation support with automatic standard-Python fallback.
- Topological Repair Loops (Outlier, Continuity, Gradient) to fix false Minuit valleys.
- Adaptive Mesh Refinement (AMR) using Monotonic PCHIP interpolation.
- Logarithmic and Geometric scanning for parameters near boundary limits.
- MCMC Warm Start integration using 'emcee' for global minimum basin location.
- Multi-dimensional 2D nested contour scanning and spatial nearest-neighbor repair.
- HPC Checkpointing for instant crash recovery of completed grid points.
- Global Minimum Tracking to actively patch Minuit "amnesia" during sequential scans.
- Pre-Flight Sanity Check to diagnose mathematically disconnected parameters.
- Explicit Interval Extraction with automatic discontinuous island merging.
- Automated Prior Interpolator Factory for loading 1D/2D empirical penalty files.
- Dynamic Parameter Unpacking for robust variable renaming.
- Angular Unit Conversion for cyclic parameter boundaries.
- Asimov Mode for computing expected sensitivities on mock data.
- Nuisance Parameter Evolution extraction for diagnostic contour tracing.
- Static Data Injection passing array payloads directly into Numba-compiled kernels.
- Rigid Bound Forcing to override dynamic sigma scaling during grid construction.

Order of Priority for Execution Parameters:
1. Command Line Interface (CLI) Arguments (Highest Priority)
2. JSON Configuration File
3. Default Fallbacks (Lowest Priority)

USER INSTRUCTIONS FOR NEW PROJECTS:
Look for comments tagged with "USER MODIFICATION REQUIRED HERE" throughout the code.
1. Update `LATEX_LABELS` with your new parameter names.
2. Replace `compute_user_model_single` with your actual fast physics/math kernel.
3. Update `GenericLikelihoodWrapper` to load any project-specific data (e.g., flux tables, detector responses).
4. Update `GenericLikelihoodWrapper.all_params` array with the EXACT parameter order your math kernel expects.
"""

import os
import sys
import json
import argparse
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.interpolate import UnivariateSpline, RegularGridInterpolator, interp1d, PchipInterpolator
from scipy.ndimage import gaussian_filter

# ==============================================================================
# 1. HPC CONFIGURATION & DEADLOCK PREVENTION
# ==============================================================================
# Disables implicitly nested parallelization in underlying C libraries to prevent node locking.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ==============================================================================
# 2. NUMBA JIT COMPILATION (WITH FALLBACK)
# ==============================================================================
try:
    from numba import njit
    NUMBA_AVAILABLE = True
    print("Numba is available. JIT compilation enabled for core math kernels.")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba is not installed. Falling back to standard Python execution.")
    def njit(*args, **kwargs):
        """Dummy decorator for systems lacking Numba. Preserves code functionality."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

# ==============================================================================
# GLOBAL DEFAULT CONFIGURATIONS (CLI EXPOSURE)
# ==============================================================================
# These dictionaries define the standard default behavior of the script, enabling
# dynamic loading into the ArgParse menu for full CLI user control.

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

# ==============================================================================
# 3. HELPER: PRIOR FILE LOADING & CONFIG PARSING
# ==============================================================================

def load_and_grid_2d(filename):
    """
    Loads empirical 2D array coordinates and maps them onto a uniform, sorted grid
    for use in SciPy's RegularGridInterpolator.

    Parameters:
    - filename (str): Path to a 3-column text file (param1, param2, delta-chi2).

    Returns:
    - tuple: (unique param1 values, unique param2 values, structured chi2 grid)
             Returns (None, None, None) if the file cannot be accessed.
    """
    try: data = np.loadtxt(filename)
    except Exception as e:
        print(f"  [Prior Error] Could not load {filename}: {e}")
        return None, None, None
    
    p1, p2, chi2 = data[:, 0], data[:, 1], data[:, 2]
    u_p1, u_p2 = np.unique(p1), np.unique(p2)
    u_p1.sort(); u_p2.sort()
    
    grid_chi2 = np.full((len(u_p1), len(u_p2)), 9999.0)
    for v1, v2, c2 in zip(p1, p2, chi2):
        i, j = np.searchsorted(u_p1, v1), np.searchsorted(u_p2, v2)
        if i < len(u_p1) and j < len(u_p2): grid_chi2[i, j] = c2
    return u_p1, u_p2, grid_chi2

class PchipBoundsWrapper:
    """
    Picklable wrapper class for SciPy's PchipInterpolator to handle out-of-bounds 
    errors consistently without relying on lambda closures, ensuring safe HPC multiprocessing.
    """
    def __init__(self, pchip_func): 
        self.pchip = pchip_func
    
    def __call__(self, x):
        val = self.pchip(x)
        if np.isnan(val): raise ValueError("Out of bounds")
        return val

def load_config(config_path):
    """
    Loads JSON configuration files, strips metadata comment lines, and executes 
    angular unit conversions on bounded variables.

    Parameters:
    - config_path (str): Filepath to the active JSON config.

    Returns:
    - dict: Cleaned and processed configuration dictionary.
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
    
    # Process mathematical unit scaling (e.g. converting explicit degrees to radians)
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

# ==============================================================================
# 4. USER IMPLEMENTATION DOMAIN (PHYSICS / MATH KERNELS)
# ==============================================================================

# USER MODIFICATION REQUIRED HERE: Update parameter dictionary with real project variable names
LATEX_LABELS = {
    "param1": r"\alpha",
    "param2": r"\beta",
    "param3": r"\gamma"
}

def get_label(param):
    """Fetches LaTeX labels for plotting, parsing internal underscores if unassigned."""
    base_label = LATEX_LABELS.get(param, param.replace('_', r'\_'))
    return rf"${base_label}$"

def format_title_stats(param, results_dict):
    """
    Parses frequentist extracted limits and format them dynamically into LaTeX strings
    used for matplotlib sub-plot titles.

    Parameters:
    - param (str): Target physics parameter name.
    - results_dict (dict): The active frequentist_results.json block.

    Returns:
    - str: A formatted LaTeX title (e.g. \beta = 0.5 +0.1 -0.2).
    """
    base_label = LATEX_LABELS.get(param, param.replace('_', r'\_'))
    if param not in results_dict: return rf"${base_label}$"
    res = results_dict[param]
    bf = res.get('best_fit')
    if bf is None: return rf"${base_label}$"
    intervals = res.get('1sigma')
    if not intervals or len(intervals) == 0: return rf"${base_label} = {bf:.3g}$"
    
    # Isolate the explicit interval island containing the best fit point
    target_interval = intervals[0]
    for interval in intervals:
        if interval[0] <= bf <= interval[1]:
            target_interval = interval
            break
    low, high = target_interval
    return rf"${base_label} = {bf:.3g}_{{-{bf - low:.3g}}}^{{+{high - bf:.3g}}}$"

# USER MODIFICATION REQUIRED HERE: Replace this function content with your actual physical model.
@njit
def compute_user_model_single(p1, p2, p3, static_data):
    """
    Numba JIT-compiled mathematical kernel. This executes the heavy-lifting simulation physics.
    
    Parameters:
    - p1, p2, p3: Numerical values evaluating active coordinate space.
    - static_data: Immutable data block (e.g., bin distributions, energy weights).

    Returns:
    - float: Absolute theoretical output mapped to target observation.
    """
    return (p1**2) + (p2 * 2.0) - np.sin(p3) + np.sum(static_data)

class GenericLikelihoodWrapper:
    """
    State-machine encapsulating parameter evaluation. Acts as a bridge transferring 
    dynamic variables mapped by Minuit/Emcee into the isolated Numba physics kernel.
    """
    def __init__(self, config, param_names, asimov=False):
        """
        Initializes the log-likelihood environment, cache management, and data structures.
        """
        self.config = config
        self.param_names = param_names
        self.asimov = asimov
        
        # USER MODIFICATION REQUIRED HERE: Update array with EXACT variable order the math kernel requires.
        self.all_params = ['param1', 'param2', 'param3'] 
        
        self.fixed_params_cfg = config.get('fixed_params', {})
        self.use_fuzzy_cache = config.get("scan_settings", {}).get("use_fuzzy_cache", False)
        
        # MCMC Memory Guard Toggle
        self._cache = {}
        self._cache_enabled = True
        
        if self.asimov:
            print("  [Asimov] Mode activated. Generating mock expected data...")
            asimov_injections = self.config.get("asimov_injections", {})
            if asimov_injections:
                print(f"  [Asimov] Injecting null-hypothesis parameters: {asimov_injections}")
            
            # Simulated data override
            self.static_data = np.zeros(10, dtype=np.float64) 
            
            # Replace empty assumptions by computing the true target payload using the injected parameters.
            mock_args = [asimov_injections.get(p, self.fixed_params_cfg.get(p, 0.0)) for p in self.all_params]
            mock_args.append(self.static_data)
            self.observed_data = compute_user_model_single(*mock_args)
            print(f"  [Asimov] Mock expectation initialized at: {self.observed_data:.4f}")
        else:
            self.static_data = np.ones(10, dtype=np.float64) 
            self.observed_data = 10.0 # Default fixed observation for real data mode
        
        self._build_prior_interpolators()
        
    def _get_val(self, vals, key):
        """Fetches dynamic floating values or static pinned values uniformly."""
        if key in vals: return vals[key]
        return self.fixed_params_cfg.get(key)
        
    def _build_prior_interpolators(self):
        """
        Automatically ingests external empirical chi2 datafiles specified in JSON, 
        compiling them into active regular grids or cubic splines.
        """
        self.prior_interpolators = []
        self.covered_params = set()
        input_configs = self.config.get('input_configs', [])
        bounds_error_setting = self.config.get('use_quadratic_fallback', False)
        use_linear_interp = self.config.get("scan_settings", {}).get("use_linear_interp", False)
        
        for entry in input_configs:
            fname, ftype, params = entry.get('file'), entry.get('type'), entry.get('params')
            units = entry.get('units', [])
            if not fname or not os.path.exists(fname): continue
                
            if ftype == '1D':
                try:
                    data = np.loadtxt(fname)
                    x_val, chi2_val = data[:, 0], data[:, 1]
                    if use_linear_interp:
                        fill_val = None if bounds_error_setting else "extrapolate"
                        interp = interp1d(x_val, chi2_val, kind='linear', bounds_error=bounds_error_setting, fill_value=fill_val)
                    else:
                        pchip = PchipInterpolator(x_val, chi2_val, extrapolate=not bounds_error_setting)
                        interp = PchipBoundsWrapper(pchip) if bounds_error_setting else pchip
                    self.prior_interpolators.append({'type': '1D', 'func': interp, 'params': params, 'units': units})
                    self.covered_params.add(params[0])
                except Exception as e: print(f"  [Prior Error] Loading 1D {fname}: {e}")

            elif ftype == '2D':
                try:
                    u_p1, u_p2, grid_chi2 = load_and_grid_2d(fname)
                    if u_p1 is None: continue
                    interp = RegularGridInterpolator((u_p1, u_p2), grid_chi2, bounds_error=bounds_error_setting, fill_value=None)
                    self.prior_interpolators.append({'type': '2D', 'func': interp, 'params': params, 'units': units})
                    self.covered_params.update(params)
                except Exception as e: print(f"  [Prior Error] Loading 2D {fname}: {e}")

    def _calculate_gaussian_fallback(self, param_name, val):
        """Computes quadratic penalties for parameters without empirical constraints."""
        fallbacks = self.config.get('fallbacks', {})
        if param_name in fallbacks:
            spec = fallbacks[param_name]
            if isinstance(spec, list) and len(spec) >= 3 and str(spec[0]).lower() == 'gauss':
                mu, sigma = spec[1], spec[2]
                if len(spec) > 3 and 'degree' in str(spec[3]).lower(): 
                    val_check = np.degrees(val)
                    diff = (val_check - mu) % 360.0
                    if diff > 180.0: diff -= 360.0
                    return (diff / sigma)**2
                else: 
                    return ((val - mu) / sigma)**2
        return 0.0

    def _compute_user_coupled_priors(self, vals):
        """User injection hook for mathematical multi-variable priors (e.g., bounds constraints)."""
        penalty = 0.0
        return penalty

    def _compute_prior_penalty(self, vals):
        """Aggregates active 1D/2D empirical constraints and fallback penalties."""
        penalty = 0.0
        for prior in self.prior_interpolators:
            p_vals = []
            for idx, p in enumerate(prior['params']):
                val = self._get_val(vals, p)
                if val is None: break
                
                # Execute unit transformation dynamically
                units = prior.get('units', [])
                if idx < len(units):
                    unit_str = str(units[idx]).lower()
                    if 'degree' in unit_str: val = np.degrees(val)
                    elif 'log10' in unit_str: val = np.log10(np.abs(val) + 1e-30)
                    elif '1e-3' in unit_str: val = np.abs(val) * 1e3
                p_vals.append(val)
                
            if len(p_vals) != len(prior['params']): continue
            
            try:
                if prior['type'] == '1D': chi2_i = prior['func'](p_vals[0])
                elif prior['type'] == '2D': chi2_i = prior['func'](np.array([p_vals]))[0]
                if np.isnan(chi2_i): chi2_i = 0.0
                penalty += max(0.0, float(chi2_i)) 
            except ValueError: pass
                
        for p in self.param_names:
            if p not in self.covered_params:
                val = self._get_val(vals, p)
                if val is not None:
                    penalty += self._calculate_gaussian_fallback(p, val)
                    
        penalty += self._compute_user_coupled_priors(vals)
        return penalty

    def __call__(self, *args):
        """
        The objective function invoked directly by Minuit gradient algorithms and MCMC samplers.
        Validates the math space and safely rejects infinities.
        """
        if self.use_fuzzy_cache: cache_key = tuple(round(x, 5) for x in args)
        else: cache_key = tuple(args)
            
        # Conditional Memory Check
        if self._cache_enabled and cache_key in self._cache: return self._cache[cache_key]
        vals = dict(zip(self.param_names, args))
        
        # Unpack floating coordinates in strict order sequence expected by compiler.
        model_args = [self._get_val(vals, p) for p in self.all_params]
        model_args = [val if val is not None else 0.0 for val in model_args]
        model_args.append(self.static_data)
        
        try: model_prediction = compute_user_model_single(*model_args)
        except Exception: return 1e9 
            
        chi2_data = abs(model_prediction - self.observed_data) 
        if np.isnan(chi2_data) or np.isinf(chi2_data): chi2_data = 1e9

        chi2_prior = self._compute_prior_penalty(vals)
        if np.isnan(chi2_prior) or np.isinf(chi2_prior): chi2_prior = 1e9 if chi2_prior > 0 else 0.0

        total_chi2 = chi2_data + chi2_prior
        if np.isnan(total_chi2) or np.isinf(total_chi2): total_chi2 = 1e9
        
        # Save evaluation to tracking dictionary only if toggle allows it.
        if self._cache_enabled: self._cache[cache_key] = total_chi2
        return total_chi2

# ==============================================================================
# 5. CONFIGURATION & MCMC UTILITIES
# ==============================================================================

def estimate_param_names(config):
    """Filters floating variables from user-assigned fixed parameters."""
    potential_params = ['param1', 'param2', 'param3'] 
    fixed_params = config.get('fixed_params', {})
    return [p for p in potential_params if p not in fixed_params]

def estimate_start_values(param_names, config):
    """Determines safe optimization origin points utilizing available boundaries and overrides."""
    starts = {}
    limits = config.get('parameter_limits', {})
    fallbacks = config.get('fallbacks', {})
    manual_starts = config.get('start_values', {})
    defaults = {'param1': 0.0, 'param2': 1.0, 'param3': -1.0} 

    for p in param_names:
        val = None
        if p in manual_starts: val = manual_starts[p]
        elif p in limits and len(limits[p]) >= 2: val = 0.5 * (limits[p][0] + limits[p][1])
        elif p in fallbacks and isinstance(fallbacks[p], list) and len(fallbacks[p]) >= 3:
             val = fallbacks[p][1] if isinstance(fallbacks[p][0], str) else fallbacks[p][0]
        
        if val is None: val = defaults.get(p, 0.0) 
        starts[p] = val
    return starts

class MCMCLikelihoodWrapper:
    """Isolates the parameter bounds checking sequence for strict MCMC probability generation."""
    def __init__(self, func, limits):
        self.func = func
        self.limits = limits

    def __call__(self, theta):
        for val, (low, high) in zip(theta, self.limits):
            if low is not None and high is not None and np.isclose(low, high):
                if val < low - 1e-6 or val > high + 1e-6: return -np.inf
            else:
                if low is not None and val < low: return -np.inf
                if high is not None and val > high: return -np.inf
        chi2 = self.func(*theta)
        if np.isnan(chi2) or chi2 > 10000.0: return -np.inf
        return -0.5 * chi2

def run_mcmc_warm_start(func, param_names, start_values, limits_dict, pool=None):
    """
    Executes a high-density Emcee randomized stochastic walk to discover the 
    global minimum basin within highly complex multi-valley phase topologies.
    """
    try: import emcee
    except ImportError:
        print("  [Warning] 'emcee' missing. Bypassing MCMC warm start.")
        return start_values
        
    print("\n" + "="*60 + "\nSTARTING MCMC WARM START\n" + "="*60)
    ndim = len(param_names)
    n_walkers = max(32, ndim * 4)
    n_steps = 300
    
    limits = [limits_dict.get(p, (None, None)) for p in param_names]
    log_prob_func = MCMCLikelihoodWrapper(func, limits)
    
    start_arr = np.array([start_values[p] for p in param_names])
    pos = []
    
    for _ in range(n_walkers):
        jitter = np.random.randn(ndim) * 0.05
        p_pos = start_arr + jitter
        for i, (low, high) in enumerate(limits):
            if low is not None and high is not None and np.isclose(low, high):
                p_pos[i] = low + np.random.randn() * 1e-8
            else:
                margin = min(1e-5, (high - low) * 0.05) if (low is not None and high is not None) else 1e-5
                if low is not None and p_pos[i] <= low: p_pos[i] = low + np.random.uniform(1e-8, margin)
                if high is not None and p_pos[i] >= high: p_pos[i] = high - np.random.uniform(1e-8, margin)
        pos.append(p_pos)
        
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob_func, pool=pool)
    
    # Actively bypass evaluation caching during high-iteration continuous random generation
    # to safeguard computational node limits from RAM leaks.
    print("  [MCMC] Disabling internal wrapper cache to prevent RAM exhaustion...")
    func._cache_enabled = False
    
    sampler.run_mcmc(pos, n_steps, progress=True)
    
    # Ensure memory traces are entirely scrubbed before enabling component scanning.
    func._cache.clear()
    func._cache_enabled = True
    
    flat_samples = sampler.get_chain(flat=True)
    best_idx = np.argmax(sampler.get_log_prob(flat=True))
    return {param_names[i]: flat_samples[best_idx][i] for i in range(ndim)}

# ==============================================================================
# 6. MINIMIZATION & REPAIR ROUTINES
# ==============================================================================

def perform_sanity_check(func, m):
    """
    Identifies completely isolated or physically unlinked model parameters by executing
    microscopic permutations against current Minuit bounds structures.
    """
    print("\n" + "="*60 + "\nSANITY CHECK: Verifying Parameter Connectivity\n" + "="*60)
    base_chi2 = m.fval
    params = m.parameters
    vals = dict(zip(params, m.values))
    failed_params = []
    for p in params:
        test_vals = vals.copy()
        val_curr = test_vals[p]
        
        err = m.errors[p] if m.errors[p] > 0 else 0.1
        step = max(err, 0.05 * abs(val_curr)) if abs(val_curr) > 1e-9 else 0.01
        
        test_val = val_curr + step
        lim = m.limits[p]
        if lim:
            low, high = lim
            if high is not None and test_val > high:
                test_val = val_curr - step 
            if low is not None and test_val < low:
                test_val = low + 1e-6
            if high is not None and test_val > high:
                test_val = high - 1e-6
                
        test_vals[p] = test_val
        
        try:
            test_chi2 = func(*[test_vals[k] for k in params])
            diff = abs(test_chi2 - base_chi2)
            status = "CONNECTED" if diff > 1e-10 else "DISCONNECTED (FLAT)"
            print(f"  {p:<15}: Base={val_curr:.3g}, Perturbed={test_vals[p]:.3g} -> Chi2 Diff={diff:.2e} [{status}]")
            if diff <= 1e-10: failed_params.append(p)
        except Exception as e:
            print(f"  {p:<15}: CRASHED ({e})")
            failed_params.append(p)
    if failed_params:
        print("\n[WARNING] The following parameters appear DISCONNECTED:")
        for fp in failed_params: print(f"  - {fp}")
    else: print("All parameters are connected. Proceeding to Scan.\n")

def minimize_point(val, param_name, global_start_values, global_errors, func, limits, param_names, use_simplex=False, minuit_retries=3):
    """
    Executes an isolated 1D profile sequence locking the specific parameter array slice, 
    permitting gradients to optimize secondary components dynamically.
    """
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    final_best_fval = np.inf
    final_best_params = {}

    try:
        m_scan = Minuit(func, name=param_names, **global_start_values)
        m_scan.errordef = Minuit.LEAST_SQUARES
        m_scan.tol = 0.1 
        for p, lim in limits.items(): m_scan.limits[p] = lim
        for p, err in global_errors.items(): 
            if err > 0: m_scan.errors[p] = err
        m_scan.values[param_name] = val
        m_scan.fixed[param_name] = True
        
        if use_simplex: m_scan.simplex()

        best_fval, best_params = np.inf, {}
        for retries in range(minuit_retries):
            m_scan.migrad()
            if m_scan.fmin is not None and m_scan.fval < best_fval:
                best_fval = m_scan.fval
                best_params = dict(zip(m_scan.parameters, m_scan.values))
            if m_scan.valid: break
                
            for p in param_names:
                if not m_scan.fixed[p]:
                    curr, err = m_scan.values[p], m_scan.errors[p]
                    step = max(err, 0.05*abs(curr)) if abs(curr) > 1e-9 else 0.01
                    new_val = curr + np.random.normal(0, step)
                    if p in limits:
                        low, high = limits[p]
                        if low is not None: new_val = max(new_val, low + 1e-6)
                        if high is not None: new_val = min(new_val, high - 1e-6)
                    m_scan.values[p] = new_val
                    
        if best_fval < final_best_fval:
            final_best_fval = best_fval
            final_best_params = best_params

    except Exception: pass
    return (final_best_fval, final_best_params) if final_best_fval < np.inf else (np.inf, {})

def minimize_point_2d(val_pair, name_pair, global_start_values, global_errors, func, limits, param_names, use_simplex=False, minuit_retries=3):
    """Executes isolated parameter minimization pinning exactly two structural array dimensions."""
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    val_x, val_y = val_pair
    name_x, name_y = name_pair
    
    final_best_fval = np.inf
    final_best_params = {}
    
    for attempt in range(2):
        try:
            m_scan = Minuit(func, name=param_names, **global_start_values)
            m_scan.errordef = Minuit.LEAST_SQUARES
            m_scan.tol = 0.1
            for p, lim in limits.items(): m_scan.limits[p] = lim
            for p, err in global_errors.items(): 
                if err > 0: m_scan.errors[p] = err
                
            m_scan.values[name_x], m_scan.fixed[name_x] = val_x, True
            m_scan.values[name_y], m_scan.fixed[name_y] = val_y, True
            
            if attempt > 0 or use_simplex:
                m_scan.simplex()
                if attempt > 0:
                    for p in param_names:
                        if not m_scan.fixed[p]:
                            curr = m_scan.values[p]
                            new_val = curr + np.random.normal(0, 0.01)
                            if p in limits:
                                low, high = limits[p]
                                if low is not None: new_val = max(new_val, low + 1e-6)
                                if high is not None: new_val = min(new_val, high - 1e-6)
                            m_scan.values[p] = new_val

            best_fval, best_params = np.inf, {}
            for retries in range(minuit_retries):
                m_scan.migrad()
                if m_scan.fmin is not None and m_scan.fval < best_fval:
                    best_fval = m_scan.fval
                    best_params = dict(zip(m_scan.parameters, m_scan.values))
                        
                if m_scan.valid: break
                
                for p in param_names:
                    if not m_scan.fixed[p]:
                        curr, err = m_scan.values[p], m_scan.errors[p]
                        step = max(err, 0.05*abs(curr)) if abs(curr) > 1e-9 else 0.01
                        new_val = curr + np.random.normal(0, step)
                        if p in limits:
                            low, high = limits[p]
                            if low is not None: new_val = max(new_val, low + 1e-6)
                            if high is not None: new_val = min(new_val, high - 1e-6)
                        m_scan.values[p] = new_val
            
            if best_fval < final_best_fval:
                final_best_fval = best_fval
                final_best_params = best_params
            
            if final_best_fval < np.inf: break
        except Exception: pass
    
    return (final_best_fval, final_best_params) if final_best_fval < np.inf else (np.inf, {})

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

def repair_scan_outliers(x_scan, chi2_scan, param_histories, param_name, func, limits, param_names, global_errors, 
                         threshold=0.2, max_passes=10, use_simplex=True, minuit_retries=3, pool=None, num_cores=1):
    """Scans computational array identifying artificial positive data jumps triggered by Minuit stalling and recalculates."""
    chi2_rep = chi2_scan.copy()
    for _ in range(max_passes):
        tasks, indices = [], []
        for i in range(len(chi2_rep)):
            if chi2_rep[i] > 10000.0: continue
            val_p = chi2_rep[i-1] if i > 0 else np.inf
            val_n = chi2_rep[i+1] if i < len(chi2_rep)-1 else np.inf
            
            if chi2_rep[i] > val_p + threshold and chi2_rep[i] > val_n + threshold:
                neighbor = param_histories[i-1] if val_p < val_n else param_histories[i+1]
                tasks.append((x_scan[i], param_name, neighbor, global_errors, func, limits, param_names, use_simplex, minuit_retries))
                indices.append(i)
                
        n_tasks = len(tasks)
        if not tasks: break
        
        if pool is None or n_tasks < 5:
            results = [_global_1d_repair_wrapper(t) for t in tasks]
        else:
            chunksize = max(1, n_tasks//(num_cores*4))
            try:
                from tqdm import tqdm
                results = list(tqdm(pool.imap(_global_1d_repair_wrapper, tasks, chunksize=chunksize), total=n_tasks, desc="       Repairing (Spikes)"))
            except ImportError:
                results = pool.map(_global_1d_repair_wrapper, tasks, chunksize=chunksize)
                
        fixes = sum(1 for k, (fval, p) in enumerate(results) if fval < chi2_rep[indices[k]] and (chi2_rep.__setitem__(indices[k], fval) or param_histories.__setitem__(indices[k], p) or True))
        if fixes == 0: break
    return x_scan, chi2_rep

def repair_scan_continuity(x_scan, chi2_scan, param_histories, param_name, func, limits, param_names, global_errors, 
                           use_simplex=True, continuity_threshold=1.0, max_passes=5, minuit_retries=3, pool=None, num_cores=1):
    """Verifies gradient transitions targeting localized unphysical numerical slope derivations for recalculation."""
    print(f"  [Continuity] Checking for slope discontinuities in {param_name} scan...")
    chi2_rep = chi2_scan.copy()
    for pass_idx in range(max_passes):
        repair_tasks, repair_indices = [], []
        for i in range(1, len(chi2_rep) - 1):
            val_p, val_c, val_n = chi2_rep[i-1], chi2_rep[i], chi2_rep[i+1]
            if val_c > 10000.0 or val_p > 10000.0 or val_n > 10000.0: continue
            slope_change = abs((val_n - val_c) - (val_c - val_p))
            deviation = val_c - 0.5 * (val_p + val_n)
            
            if slope_change > continuity_threshold and deviation > 0.1:
                repair_tasks.extend([(x_scan[i], param_name, param_histories[i-1], global_errors, func, limits, param_names, use_simplex, minuit_retries),
                                     (x_scan[i], param_name, param_histories[i+1], global_errors, func, limits, param_names, use_simplex, minuit_retries)])
                repair_indices.extend([i, i])
        
        n_tasks = len(repair_tasks)
        if n_tasks == 0: break
        
        if pool is None or n_tasks < 5:
            results = [_global_1d_repair_wrapper(t) for t in repair_tasks]
        else:
            chunksize = max(1, n_tasks//(num_cores*4))
            try:
                from tqdm import tqdm
                results = list(tqdm(pool.imap(_global_1d_repair_wrapper, repair_tasks, chunksize=chunksize), total=n_tasks, desc="       Repairing (Continuity)"))
            except ImportError:
                results = pool.map(_global_1d_repair_wrapper, repair_tasks, chunksize=chunksize)
        
        fixes = sum(1 for k, (fval, p) in enumerate(results) if fval < chi2_rep[repair_indices[k]] and (chi2_rep.__setitem__(repair_indices[k], fval) or param_histories.__setitem__(repair_indices[k], p) or True))
        if fixes == 0: break
    return x_scan, chi2_rep

def repair_steep_gradients(x_scan, chi2_scan, param_histories, param_name, func, limits, param_names, global_errors, 
                           use_simplex=True, threshold=3.0, max_passes=5, minuit_retries=3, pool=None, num_cores=1):
    """Repairs jagged structural edges ensuring physically continuous objective surfaces."""
    print(f"  [Gradient] Checking for steep gradients in {param_name} scan...")
    chi2_rep = chi2_scan.copy()
    for pass_idx in range(max_passes):
        repair_tasks, repair_indices = [], []
        for i in range(len(chi2_rep) - 1):
            val_a, val_b = chi2_rep[i], chi2_rep[i+1]
            if val_a > 10000.0 or val_b > 10000.0: continue
            
            if abs(val_a - val_b) > threshold:
                bad_idx, good_idx = (i, i+1) if val_a > val_b else (i+1, i)
                if bad_idx in repair_indices: continue
                repair_tasks.append((x_scan[bad_idx], param_name, param_histories[good_idx], global_errors, func, limits, param_names, use_simplex, minuit_retries))
                repair_indices.append(bad_idx)
                
        n_tasks = len(repair_tasks)
        if n_tasks == 0: break
        
        if pool is None or n_tasks < 5:
            results = [_global_1d_repair_wrapper(t) for t in repair_tasks]
        else:
            chunksize = max(1, n_tasks//(num_cores*4))
            try:
                from tqdm import tqdm
                results = list(tqdm(pool.imap(_global_1d_repair_wrapper, repair_tasks, chunksize=chunksize), total=n_tasks, desc="       Repairing (Gradients)"))
            except ImportError:
                results = pool.map(_global_1d_repair_wrapper, repair_tasks, chunksize=chunksize)
        
        fixes = sum(1 for k, (fval, p) in enumerate(results) if fval < chi2_rep[repair_indices[k]] and (chi2_rep.__setitem__(repair_indices[k], fval) or param_histories.__setitem__(repair_indices[k], p) or True))
        if fixes == 0: break
    return x_scan, chi2_rep

def _global_repair_wrapper(task_args):
    return minimize_point_2d(*task_args)

def repair_2d_scan_outliers(x_grid, y_grid, Z_chi2, param_grid, name_pair, func, limits, param_names, global_errors,
                            use_simplex=True, threshold=0.2, max_passes=100, minuit_retries=3, pool=None, num_cores=1):
    """Maps nearest-neighbor evaluations tracing 2D anomalies to mathematically verify structural depressions."""
    Z_repaired = Z_chi2.copy()
    rows, cols = Z_repaired.shape
    for pass_idx in range(max_passes):
        repair_tasks, repair_indices = [], []
        for i in range(rows):
            for j in range(cols):
                if Z_repaired[i, j] > 10000.0: continue
                neighbors = [(n_i, n_j, Z_repaired[n_i, n_j]) for n_i, n_j in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)] if 0 <= n_i < rows and 0 <= n_j < cols]
                if not neighbors: continue
                min_n = min(neighbors, key=lambda x: x[2])
                if Z_repaired[i, j] > min_n[2] + threshold:
                    best_params = param_grid[min_n[0]][min_n[1]]
                    if best_params:
                        repair_tasks.append(((x_grid[j], y_grid[i]), name_pair, best_params, global_errors, func, limits, param_names, use_simplex, minuit_retries))
                        repair_indices.append((i, j))
        
        n_tasks = len(repair_tasks)
        if n_tasks == 0: break
        if pool is None or n_tasks < 5: 
            results = [_global_repair_wrapper(t) for t in repair_tasks]
        else: 
            chunksize = max(1, n_tasks // (num_cores * 4))
            try:
                from tqdm import tqdm
                results = list(tqdm(pool.imap(_global_repair_wrapper, repair_tasks, chunksize=chunksize), total=n_tasks, desc="       Repairing"))
            except ImportError:
                results = pool.map(_global_repair_wrapper, repair_tasks, chunksize=chunksize)
                
        fixes_applied = sum(1 for idx, (fval, params) in enumerate(results) if fval < Z_repaired[repair_indices[idx][0], repair_indices[idx][1]] and (Z_repaired.__setitem__((repair_indices[idx][0], repair_indices[idx][1]), fval) or param_grid[repair_indices[idx][0]].__setitem__(repair_indices[idx][1], params) or True))
        if fixes_applied == 0: break
    return x_grid, y_grid, Z_repaired

def calculate_intervals(x_scan, dchi2, name, thresholds, merge_gaps=True, gap_tolerance=0.1, enforce_monotonicity=False):
    """Locates specific zero-roots across splines deriving formal frequentist uncertainty exclusions."""
    if enforce_monotonicity:
        print(f"  [Intervals] Enforcing monotonicity on {name} arrays before root extraction.")
        sort_idx = np.argsort(x_scan)
        x_scan = x_scan[sort_idx]
        dchi2 = dchi2[sort_idx]
        
    results = {}
    valid_mask = dchi2 < 1e5
    if np.sum(valid_mask) < 2: return {"best_fit": float(x_scan[np.argmin(dchi2)])}
    
    x_valid = x_scan[valid_mask]
    y_valid = dchi2[valid_mask]
    
    spline = UnivariateSpline(x_valid, y_valid, s=0, k=1, ext=0)
    results['best_fit'] = float(x_scan[np.argmin(dchi2)])
    scan_min, scan_max = x_scan[0], x_scan[-1]

    for label, thresh in thresholds.items():
        if "2d" in label.lower(): continue
        roots = []
        for i in range(len(x_scan) - 1):
            y1, y2 = dchi2[i] - thresh, dchi2[i+1] - thresh
            if (y1 * y2 <= 0) and (y1 != y2): 
                roots.append(x_scan[i] - y1 * (x_scan[i+1] - x_scan[i]) / (y2 - y1))
        
        boundaries = [scan_min] + sorted(roots) + [scan_max]
        valid_intervals = [[float(s), float(e)] for s, e in zip(boundaries[:-1], boundaries[1:]) if e - s > 1e-9 and spline((s + e) / 2.0) < thresh]
        
        if merge_gaps and len(valid_intervals) > 1:
            merged_intervals = []
            c_s, c_e = valid_intervals[0]
            for n_s, n_e in valid_intervals[1:]:
                if np.max(spline(np.linspace(c_e, n_s, 20))) - thresh < gap_tolerance: c_e = n_e
                else: merged_intervals.append([c_s, c_e]); c_s, c_e = n_s, n_e
            merged_intervals.append([c_s, c_e])
            valid_intervals = merged_intervals

        results[label] = valid_intervals if valid_intervals else None
        print(f"  {label:<7}: {' U '.join([f'[{i[0]:.4g}, {i[1]:.4g}]' for i in valid_intervals]) if valid_intervals else 'Not found'}")
    return results

# ==============================================================================
# 7. PLOTTING ROUTINES
# ==============================================================================

def plot_1d_analysis(output_dir, param_names, confidence_cfg, global_min_chi2):
    """Synthesizes isolated JSON payload outputs formulating a consolidated 1D multi-panel graphic."""
    if not param_names: return
    rows = (len(param_names) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4*rows))
    axes = np.atleast_1d(axes).flatten()
    
    info_dir = os.path.join(output_dir, 'info')
    freq_results = json.load(open(os.path.join(info_dir, 'frequentist_results.json'))) if os.path.exists(os.path.join(info_dir, 'frequentist_results.json')) else {}
    
    for i, p in enumerate(param_names):
        json_path = os.path.join(info_dir, f"profile_1d_{p}.json")
        if not os.path.exists(json_path): axes[i].text(0.5, 0.5, "No Data", ha='center'); continue
            
        data = json.load(open(json_path))
        x_scan, dchi2 = np.array(data["x"]), np.nan_to_num(np.array(data["y"]) - global_min_chi2, nan=1e9)
        dchi2[dchi2 < 0] = 0
        
        axes[i].plot(x_scan, dchi2, 'b.-', lw=1.5)
        for sig in ["1sigma", "2sigma", "3sigma"]:
            if sig in confidence_cfg: axes[i].axhline(confidence_cfg[sig], color='gray', ls={'1sigma':'--', '2sigma':':', '3sigma':'-.'}[sig], label=rf'${sig[0]}\sigma$')
        
        axes[i].set_xlabel(get_label(p))
        axes[i].set_title(format_title_stats(p, freq_results))
        axes[i].set_ylim(0, 15)
        if i == 0 and axes[i].get_legend_handles_labels()[0]: axes[i].legend(loc='upper right')
        
    for j in range(i+1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "plots", "frequentist_summary_1d.pdf"))

def plot_2d_analysis(output_dir, param_names, confidence_cfg, global_min_chi2, scan_settings):
    """Produces the definitive multidimensional corner-plot grid illustrating topological phase combinations."""
    if not param_names: return
    n_params = len(param_names)
    info_dir = os.path.join(output_dir, 'info')
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
        
    freq_results = json.load(open(os.path.join(info_dir, 'frequentist_results.json'))) if os.path.exists(os.path.join(info_dir, 'frequentist_results.json')) else {}
            
    fig_corner, axes_corner = plt.subplots(n_params, n_params, figsize=(3*n_params, 3*n_params))
    if n_params == 1: axes_corner = np.array([[axes_corner]])
        
    for i in range(n_params):
        for j in range(n_params):
            ax = axes_corner[i, j]
            param_y, param_x = param_names[i], param_names[j]
            
            if i == j:
                json_path = os.path.join(info_dir, f"profile_1d_{param_x}.json")
                if os.path.exists(json_path):
                    data = json.load(open(json_path))
                    x_1d, y_1d = np.array(data["x"]), np.array(data["y"]) - global_min_chi2
                    y_1d[y_1d < 0] = 0
                    ax.plot(x_1d, y_1d, 'k-', lw=1.5)
                    for sig_lbl in ["1sigma", "2sigma", "3sigma"]:
                        if sig_lbl in confidence_cfg: ax.axhline(confidence_cfg[sig_lbl], color='gray', ls={'1sigma':'--', '2sigma':':', '3sigma':'-.'}[sig_lbl])
                    ax.set_ylim(0, 10); ax.set_xlim(np.min(x_1d), np.max(x_1d))
                    ax.set_title(format_title_stats(param_x, freq_results))
                else: 
                    marginalized = False
                    for other_param in param_names:
                        if other_param == param_x: continue
                        path1 = os.path.join(info_dir, f"profile_2d_{param_x}_vs_{other_param}.json")
                        path2 = os.path.join(info_dir, f"profile_2d_{other_param}_vs_{param_x}.json")
                        
                        valid_path = path1 if os.path.exists(path1) else (path2 if os.path.exists(path2) else None)
                        if valid_path:
                            print(f"  [Plotting] 1D data missing for {param_x}. Marginalizing from 2D surface: {os.path.basename(valid_path)}")
                            data2d = json.load(open(valid_path))
                            x_grid_2d, y_grid_2d, Z_chi2_2d = np.array(data2d["x"]), np.array(data2d["y"]), np.array(data2d["z"])
                            
                            if valid_path == path1:
                                marg_1d = np.nanmin(Z_chi2_2d, axis=0)
                                x_1d = x_grid_2d
                            else:
                                marg_1d = np.nanmin(Z_chi2_2d, axis=1) 
                                x_1d = y_grid_2d
                                
                            y_1d = np.nan_to_num(marg_1d - global_min_chi2, nan=1e9)
                            y_1d[y_1d < 0] = 0
                            
                            smooth_marginalized = scan_settings.get("smooth_marginalized_1d", False)
                            if smooth_marginalized and len(x_1d) > 3:
                                print(f"  [Plotting] Smoothing marginalized 1D curve for {param_x}.")
                                sort_idx = np.argsort(x_1d)
                                x_1d_s = x_1d[sort_idx]
                                y_1d_s = y_1d[sort_idx]
                                pchip = PchipInterpolator(x_1d_s, y_1d_s)
                                x_1d_plot = np.linspace(np.min(x_1d_s), np.max(x_1d_s), 100)
                                y_1d_plot = pchip(x_1d_plot)
                                y_1d_plot[y_1d_plot < 0] = 0
                                ax.plot(x_1d_plot, y_1d_plot, 'k--', lw=1.5, label='Marginalized (Smoothed)')
                            else:
                                ax.plot(x_1d, y_1d, 'k--', lw=1.5, label='Marginalized')
                                
                            for sig_lbl in ["1sigma", "2sigma", "3sigma"]:
                                if sig_lbl in confidence_cfg: ax.axhline(confidence_cfg[sig_lbl], color='gray', ls={'1sigma':'--', '2sigma':':', '3sigma':'-.'}[sig_lbl])
                            ax.set_ylim(0, 10); ax.set_xlim(np.min(x_1d), np.max(x_1d))
                            ax.set_title(format_title_stats(param_x, freq_results))
                            marginalized = True
                            break
                            
                    if not marginalized:
                        ax.text(0.5, 0.5, "No Data", ha='center')
                if i == n_params-1: ax.set_xlabel(get_label(param_x))
                else: ax.set_xticklabels([])
                ax.set_ylabel(r'$\Delta \chi^2$')
                ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
                
            elif i > j:
                json_path = os.path.join(info_dir, f"profile_2d_{param_x}_vs_{param_y}.json")
                if os.path.exists(json_path):
                    data = json.load(open(json_path))
                    x_grid, y_grid, Z_chi2 = np.array(data["x"]), np.array(data["y"]), np.array(data["z"])
                    
                    Z_dchi2 = np.nan_to_num(Z_chi2 - global_min_chi2, nan=1e9)
                    Z_dchi2[Z_dchi2 < 0] = 0
                    if len(Z_dchi2.shape) == 1: Z_dchi2 = Z_dchi2.reshape((int(np.sqrt(len(Z_dchi2))), int(np.sqrt(len(Z_dchi2)))))
                    
                    clip_smearing = scan_settings.get("clip_contour_smearing", False)
                    if clip_smearing:
                        Z_dchi2_for_smooth = np.clip(Z_dchi2, a_min=None, a_max=50.0)
                        Z_smooth = gaussian_filter(Z_dchi2_for_smooth, sigma=0.5)
                    else:
                        Z_smooth = gaussian_filter(Z_dchi2, sigma=0.5)
                    
                    levels = [confidence_cfg.get(l, def_v) for l, def_v in [("1sigma_2d", 2.30), ("2sigma_2d", 4.61), ("3sigma_2d", 11.83)] if l.split('_')[0] in confidence_cfg] or [2.30, 4.61, 11.83]
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(levels)]
                    
                    X, Y = np.meshgrid(x_grid, y_grid)
                    ax.pcolormesh(X, Y, Z_smooth, shading='auto', cmap='viridis_r', vmin=0, vmax=max(levels)*1.2)
                    try:
                        ax.contourf(X, Y, Z_smooth, levels=[0]+levels, colors=colors, alpha=0.3)
                        ax.contour(X, Y, Z_smooth, levels=levels, colors=colors, linewidths=1.0)
                    except ValueError: pass

                if i == n_params-1: ax.set_xlabel(get_label(param_x))
                else: ax.set_xticklabels([])
                if j == 0: ax.set_ylabel(get_label(param_y))
                else: ax.set_yticklabels([])
            else: ax.axis('off')
    
    plt.savefig(os.path.join(plot_dir, "frequentist_corner_2d.pdf"), bbox_inches='tight')

# ==============================================================================
# 8. MAIN EXECUTION PIPELINE
# ==============================================================================

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

                norm_min_2d = min(global_min_chi2, np.nanmin(Z_chi2))
                if np.isinf(global_min_chi2) and norm_min_2d < 1e5:
                    print(f"  [Fix #2] Recovered valid 2D baseline minimum from grid: {norm_min_2d:.4f}")
                Z_dchi2 = np.nan_to_num(Z_chi2 - norm_min_2d, nan=1e9)
                Z_dchi2[Z_dchi2 < 0] = 0
                surfaces_2d[f"{param_y}_vs_{param_x}"] = {"x": x_grid.tolist(), "y": y_grid.tolist(), "z": Z_dchi2.tolist()}
                
    with open(os.path.join(args.output_dir, "profile_scan_2d_data.json"), 'w') as f: json.dump(surfaces_2d, f, indent=4)
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

    config = load_config(args.config_file)
    if not config.get("output_dir"): 
        config["output_dir"] = args.output_dir
    os.makedirs(config["output_dir"], exist_ok=True)
    
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