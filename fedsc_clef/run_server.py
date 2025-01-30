# fedsc_clef/run_server.py

import flwr as fl
from flwr.server.app import start_supernode  # For aggregator-based approach
from .aggregator import EnhancedFedSCAggregator
from .config import DEFAULT_NUM_CLIENTS, DEFAULT_NUM_ROUNDS, SEC_AGG_CONFIG

def main_server():
    num_clients = DEFAULT_NUM_CLIENTS
    num_rounds = DEFAULT_NUM_ROUNDS

    aggregator = EnhancedFedSCAggregator(
        num_clients=num_clients, 
        n_clusters=3, 
        num_rounds=num_rounds
    )

    server_config = fl.server.ServerConfig(num_rounds=num_rounds)

    start_supernode(
        server_address="0.0.0.0:8080",
        config=server_config,
        aggregator=aggregator,
        # security_arguments=fl.server.secure_aggregation(**SEC_AGG_CONFIG)  # adapt to your Flower version
    )

if __name__ == "__main__":
    main_server()
