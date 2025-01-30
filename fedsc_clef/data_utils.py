"""
Data loading utilities for CLEF dataset (Sec 5.1 of FedSC paper).
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def load_clef_data(client_id: int) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Load CLEF genetic dataset from partitioned client files.
    
    Args:
        client_id: Client identifier (0-based index)
    
    Returns:
        (data, labels): Tuple containing
            - data: 2D array of shape (n_samples, n_features)
            - labels: 1D array of cluster labels or None
            
    Paper Reference: Section 5.1 - Dataset Description
    """
    base_path = "path/to/Hereditary/third_workshop"
    data_file = os.path.join(base_path, f"client_{client_id}_data.csv")
    label_file = os.path.join(base_path, f"client_{client_id}_labels.csv")

    data = pd.read_csv(data_file).values.astype(np.float32)
    labels = pd.read_csv(label_file).values.flatten() if os.path.exists(label_file) else None
    return data, labels

def build_rbf_affinity(data: np.ndarray, gamma: float = 0.1) -> np.ndarray:
    """
    Construct RBF kernel matrix (Eq 2 in FedSC paper).
    
    Args:
        data: Input data matrix of shape (n_samples, n_features)
        gamma: Kernel bandwidth parameter
    
    Returns:
        affinity: Symmetric affinity matrix of shape (n_samples, n_samples)
    """
    return rbf_kernel(data, gamma=gamma).astype(np.float32)
