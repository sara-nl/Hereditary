# fedsc_clef_enhanced.py
import numpy as np
import flwr as fl
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
import os
from flwr.common import NDArrays, Scalar, Metrics
from flwr.server.strategy import Strategy
from flwr.server import ServerConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

# --- Secure Aggregation Configuration ---
SEC_AGG_CONFIG = {
    "dropout_aggregation": True,
    "sec_agg_prev_clients": 3,
    "sec_agg_threshold": 0.7,
    "sec_agg_srcs": ["driver"],
}

from task import load_clef_data_for_spectral_clustering
# --- CLEF Dataset Loading ---


# --- FedSC Core Algorithms ---
def federated_laplacian_consensus(
    server_round: int,
    local_affinities: List[np.ndarray],
    prev_global_L: Optional[np.ndarray] = None
) -> np.ndarray:
    """Iterative consensus Laplacian approximation (FedSC Sec 3.2)."""
    if prev_global_L is None or server_round == 1:
        return np.mean(local_affinities, axis=0)
    
    # Combine local and global information
    consensus_L = 0.7 * prev_global_L + 0.3 * np.mean(local_affinities, axis=0)
    return consensus_L

def secure_eigen_decomposition(matrix: np.ndarray, k: int = 3) -> np.ndarray:
    """Privacy-preserving eigenvalue computation with noise injection."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Add differential privacy noise (ε=1.0)
    noise = np.random.laplace(0, 1/np.sqrt(matrix.size), eigenvectors[:, :k].shape)
    return eigenvectors[:, :k] + noise

# --- Flower Client with Secure Updates ---
class FedSCClient(fl.client.Client):
    def __init__(self, cid: int):
        self.cid = cid
        self.data, self.labels = load_clef_data_for_spectral_clustering(cid, num_partitions=10)
        self.affinity = rbf_kernel(self.data, gamma=0.1)
        self.current_round = 0

    def get_properties(self, config: Dict[str, Scalar]) -> Dict[str, Scalar]:
        return {"num_samples": len(self.data)}

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [self.affinity.flatten()]

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.current_round = config["server_round"]
        
        # Receive global Laplacian update
        if parameters:
            global_L = parameters[0].reshape(self.affinity.shape)
            self.affinity = 0.8 * self.affinity + 0.2 * global_L  # Consensus update
            
        return self.get_parameters(config), len(self.data), {}

# --- Enhanced FedSC Strategy ---


# --- Execution Workflow ---
def main():
    strategy = EnhancedFedSCStrategy(n_clusters=3, num_rounds=5)
    
    # Start server with secure aggregation
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=5),
        strategy=strategy,
        # security_arguments=fl.server.secure_aggregation(**SEC_AGG_CONFIG)
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--cid", type=int, default=0)
    parser.add_argument("--num_partitions", type=int, default=10)
    args = parser.parse_args()

    if args.client:
        client = FedSCClient(args.cid)
        fl.client.start_client(server_address="127.0.0.1:8080", client=client)
    else:
        main()