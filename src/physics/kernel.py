"""
kernel.py

USER MODIFICATION REQUIRED HERE:
This module isolates the actual physics evaluation and mathematical payload.
Configure your specific observable topologies, arrays, and labels here.
"""

import numpy as np

# ==============================================================================
# NUMBA JIT COMPILATION (WITH FALLBACK)
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

# USER MODIFICATION REQUIRED HERE: Update parameter dictionary with real project variable names
LATEX_LABELS = {
    "param1": r"\alpha",
    "param2": r"\beta",
    "param3": r"\gamma"
}

def get_label(param):
    """
    Fetches LaTeX labels for plotting, parsing internal underscores if unassigned.
    
    Parameters:
    - param (str): Exact dictionary parameter key.
    
    Returns:
    - str: Fully formatted LaTeX string.
    """
    base_label = LATEX_LABELS.get(param, param.replace('_', r'\_'))
    return rf"${base_label}$"

def format_title_stats(param, results_dict):
    """
    Parses frequentist extracted limits and formats them dynamically into LaTeX strings
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
# [Fix #2]: Array passing ensures stable C-compilation without JIT unpacking errors.
@njit
def compute_user_model_single(params, static_data):
    """
    Numba JIT-compiled mathematical kernel. This executes the heavy-lifting simulation physics.
    
    Parameters:
    - params (np.ndarray): Contiguous 1D array of active float evaluations.
    - static_data (np.ndarray): Immutable data block (e.g., bin distributions, weights).

    Returns:
    - float: Absolute theoretical output mapped to target observation.
    """
    return (params[0]**2) + (params[1] * 2.0) - np.sin(params[2]) + np.sum(static_data)