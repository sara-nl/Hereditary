"""
Core FedSC algorithm implementation (Sec 3.2-3.3 of paper).
"""
import numpy as np
import flwr as fl
from typing import List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from flwr.common import NDArrays, Scalar, FitRes
from flwr.server.aggregator import Aggregator
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .config import DEFAULT_PARAMS
from .data_utils import load_clef_data

class EnhancedFedSCAggregator(Aggregator):
    """
    Federated aggregator implementing consensus-based spectral clustering.
    
    Paper Reference:
    - Sec 3.2: Federated Laplacian Consensus
    - Sec 4.1: Global Clustering
    - Sec 4.2: Privacy Preservation
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.config = {**DEFAULT_PARAMS, **kwargs}
        self.global_L: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.metrics = []

        # Pre-load ground-truth labels for evaluation
        self.all_labels = []
        for cid in range(self.config["num_clients"]):
            _, labels = load_clef_data(cid)
            if labels is not None:
                self.all_labels.extend(labels)
        self.all_labels = np.array(self.all_labels)

    def federated_consensus(self, local_affinities: List[np.ndarray], server_round: int) -> np.ndarray:
        """
        Compute consensus Laplacian (Eq 5).
        
        Args:
            local_affinities: List of client affinity matrices
            server_round: Current federation round
            
        Returns:
            global_L: Consensus Laplacian matrix
        """
        if self.global_L is None or server_round == 1:
            return np.mean(local_affinities, axis=0)
        
        alpha = self.config["consensus_alpha"]
        return alpha * self.global_L + (1 - alpha) * np.mean(local_affinities, axis=0)

    def secure_spectral_decomp(self, matrix: np.ndarray) -> np.ndarray:
        """
        Differentially private eigen decomposition (Sec 4.2).
        
        Args:
            matrix: Input Laplacian matrix
            
        Returns:
            eigenvectors: Top-k eigenvectors with DP noise
        """
        eigvals, eigvecs = np.linalg.eigh(matrix)
        idx = np.argsort(eigvals)[::-1][: self.config["n_clusters"]]
        
        # Noise scale: dp_epsilon / sqrt(n*m), where n*m = matrix.size
        noise_scale = self.config["dp_epsilon"] / np.sqrt(matrix.size)
        noise = np.random.laplace(scale=noise_scale, size=eigvecs[:, idx].shape)
        
        return eigvecs[:, idx] + noise

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[NDArrays], dict]:
        """
        Main aggregation workflow:
        1. Collect client updates
        2. Compute consensus Laplacian
        3. Perform private spectral decomposition
        4. Cluster embeddings
        """
        # Extract client affinities
        # Hard-coded shape (100, 100); adapt to real shape if needed
        client_affs = [res.parameters[0].reshape(100, 100) for _, res in results]

        # Update global Laplacian
        self.global_L = self.federated_consensus(client_affs, server_round)
        
        # Private eigen decomposition
        embeddings = self.secure_spectral_decomp(self.global_L)
        
        # Global clustering
        self.cluster_labels = KMeans(
            n_clusters=self.config["n_clusters"],
            random_state=42
        ).fit_predict(embeddings)
        
        # Return flattened Laplacian to the clients
        return [self.global_L.flatten()], {}

    def evaluate(self, server_round: int, parameters: NDArrays) -> Optional[Tuple[float, dict]]:
        """
        Compute clustering metrics (Sec 5.2 Evaluation Protocol).
        
        Returns:
            (loss, metrics): Tuple containing
                - loss: Placeholder (0.0)
                - metrics: Dictionary of evaluation scores
        """
        if len(self.all_labels) != len(self.cluster_labels):
            return None
        
        ari = adjusted_rand_score(self.all_labels, self.cluster_labels)
        sil = silhouette_score(self.global_L, self.cluster_labels)
        consensus_error = float(np.mean(np.var(self.global_L, axis=0)))
        
        metrics_dict = {
            "ari": ari,
            "silhouette": sil,
            "consensus_error": consensus_error,
        }
        self.metrics.append(metrics_dict)
        
        return 0.0, metrics_dict
