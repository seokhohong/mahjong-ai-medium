"""
Data loading utilities for AC learning.
"""
from __future__ import annotations

import pickle
from typing import Any
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_gsv_scaler(data_path: str | None) -> StandardScaler | None:
    """Load the StandardScaler from a dataset if available.

    Args:
        data_path: Path to the .npz dataset file, or None

    Returns:
        The fitted StandardScaler for game state vectors, or None if not available

    Example:
        # Load dataset and scaler
        data = np.load('training_data/dataset.npz')
        scaler = load_gsv_scaler('training_data/dataset.npz')

        # Use at inference time
        scaled_gsv = scaler.transform([raw_gsv])
    """
    if data_path is None:
        return None

    import os
    if not os.path.exists(data_path):
        print(f"Warning: Scaler dataset not found: {data_path}")
        return None

    try:
        data = np.load(data_path, allow_pickle=True)
        if 'gsv_scaler' in data:
            scaler_bytes = data['gsv_scaler']
            scaler = pickle.loads(scaler_bytes)
            print(f"Loaded GSV scaler from: {data_path}")
            return scaler
        else:
            print(f"Warning: No GSV scaler found in {data_path}")
            return None
    except Exception as e:
        print(f"Warning: Could not load GSV scaler from {data_path}: {e}")
        return None
