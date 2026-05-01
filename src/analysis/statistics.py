"""
statistics.py

Handles advanced mathematical evaluation of the compiled grid results, generating roots
and extrapolating continuous exclusion constraints structurally independent of graphics.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline

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