"""
FedSC client implementation (Sec 3.1 Local Computation).
"""
import flwr as fl
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar

from .data_utils import load_clef_data, build_rbf_affinity

class FedSCClient(fl.client.Client):
    """
    Client node performing local affinity computation and consensus updates.
    
    Paper Reference:
    - Sec 3.1: Local Affinity Matrix Construction
    - Sec 3.2: Local-Global Consensus Update
    """
    
    def __init__(self, cid: int):
        super().__init__()
        self.cid = cid
        self.data, self.labels = load_clef_data(cid)
        self.affinity = build_rbf_affinity(self.data)
        self.round = 0

    def get_properties(self, config: Dict[str, Scalar]) -> Dict[str, Scalar]:
        """Return local properties (e.g., # of samples)."""
        return {"num_samples": len(self.data)}

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, dict]:
        """
        Update local affinity matrix with global consensus.
        
        Args:
            parameters: Global Laplacian matrix (flattened)
            config: Configuration dictionary
            
        Returns:
            (parameters, num_samples, metrics): Updated client state
        """
        self.round = config.get("server_round", 0)
        
        if parameters:
            global_L = parameters[0].reshape(self.affinity.shape)
            alpha_local = config.get("alpha", 0.8)
            self.affinity = alpha_local * self.affinity + (1 - alpha_local) * global_L
        
        return [self.affinity.flatten()], len(self.data), {}
