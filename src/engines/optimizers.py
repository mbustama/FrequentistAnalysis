"""
optimizers.py

Governs the operational minimizers. Contains point-specific Minuit gradient descents,
sanity bounds checks, and the Emcee stochastic basin hopper.
"""

import os
import numpy as np
from iminuit import Minuit

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
    
    # [Fix #3: Memory Guard]
    print("  [MCMC] Disabling wrapper evaluation cache for random walker burn-in to prevent RAM leaks...")
    func._cache_enabled = False
    
    sampler.run_mcmc(pos, n_steps, progress=True)
    
    # Ensure memory traces are entirely scrubbed before enabling component scanning.
    func._cache.clear()
    func._cache_enabled = True
    
    flat_samples = sampler.get_chain(flat=True)
    best_idx = np.argmax(sampler.get_log_prob(flat=True))
    return {param_names[i]: flat_samples[best_idx][i] for i in range(ndim)}

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