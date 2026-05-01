"""
config.py

Handles the parsing, serialization, and initial processing of the JSON payload.
Responsible for stripping comments and converting mathematical constraints (like degrees).
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