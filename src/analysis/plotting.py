"""
plotting.py

Generates publication-quality physics diagrams directly from validated data arrays.
Maintains functional isolation preventing headless remote nodes from crashing during rendering.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter
from src.physics.kernel import get_label, format_title_stats

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