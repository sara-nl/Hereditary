# Flower ClientApp
from flwr.common.context import Context
from flwr.client import ClientApp, Client
from flwr.common import Scalar, NDArrays
from typing import Dict, Tuple
from task import load_clef_data_for_spectral_clustering, build_rbf_affinity
from sklearn.metrics.pairwise import rbf_kernel

# --- Flower Client with Secure Updates ---
class FedSCClient(Client):
    def __init__(self, cid: int, num_partitions: int):
        self.cid = cid
        self.num_partitions = num_partitions
        self.data, self.labels = load_clef_data_for_spectral_clustering(cid, num_partitions)
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

    
def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    client = FedSCClient(partition_id, num_partitions)
    return client

app = ClientApp(
    client_fn,
)