from logging import INFO, WARNING
from typing import Dict, List, Optional, Union, cast

from flwr.common import Context, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.common.logger import log
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
from task import load_clef_data_for_spectral_clustering, federated_laplacian_consensus, federated_laplacian_approximation, secure_eigen_decomposition

class EnhancedFedSCStrategy(Strategy):
    def __init__(self, n_clusters: int = 3, num_rounds: int = 5):
        super().__init__()
        self.n_clusters = n_clusters
        self.num_rounds = num_rounds
        self.global_L = None
        self.cluster_labels = None
        self.metrics_history = []

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[NDArrays]:
        return None  # Initial parameters not needed

    def configure_fit(
        self,
        server_round: int,
        parameters: NDArrays,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, Dict[str, Scalar]]]:
        config = {
            "server_round": server_round,
            "current_global_L": self.global_L.flatten().tolist() if self.global_L is not None else []
        }
        return [(client, config) for client in client_manager.all().values()]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[NDArrays], Dict[str, Scalar]]:
        # Collect client affinities with secure aggregation
        client_affinities = [res.parameters[0].reshape(100, 100) for _, res in results]
        
        # Consensus Laplacian update
        self.global_L = federated_laplacian_consensus(
            server_round,
            client_affinities,
            self.global_L
        )

        # Perform secure spectral decomposition
        embeddings = secure_eigen_decomposition(self.global_L, k=self.n_clusters)

        # Global clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.cluster_labels = kmeans.fit_predict(embeddings)

        # Prepare next round parameters
        parameters = [self.global_L.flatten()]
        return parameters, {}

    def evaluate(
        self,
        server_round: int,
        parameters: NDArrays
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Collect true labels from all clients
        all_labels = []
        for cid in range(3):  # Assuming 3 clients
            _, labels = load_clef_data_for_spectral_clustering(cid, num_partitions=10)
            if labels is not None:
                all_labels.extend(labels)
        
        if len(all_labels) == len(self.cluster_labels):
            ari = adjusted_rand_score(all_labels, self.cluster_labels)
            sil_score = silhouette_score(self.global_L, self.cluster_labels)
            metrics = {
                "adjusted_rand_index": ari,
                "silhouette_score": sil_score,
                "consensus_error": np.mean(np.var(self.global_L, axis=0))
            }
            self.metrics_history.append(metrics)
            return 0.0, metrics  # Loss not meaningful in clustering
        return None
    
    aggregate_evaluate, configure_evaluate = aggregate_fit, configure_fit


def server_fn(context: Context):
    config = ServerConfig(num_rounds=3)
    components = ServerAppComponents(
        config=config,
        strategy = EnhancedFedSCStrategy(),
        client_manager=SimpleClientManager(),
    )
    return components


# Create ServerApp
app = ServerApp(
    server_fn=server_fn,
)
