"""
Orchestration script for FedSC experiments.
"""
import argparse
import flwr as fl

from .aggregator import EnhancedFedSCAggregator
from .client import FedSCClient
from .config import DEFAULT_PARAMS, SEC_AGG_CONFIG

def main():
    parser = argparse.ArgumentParser(description="FedSC Clustering")
    parser.add_argument("--client", action="store_true", help="Run as client process")
    parser.add_argument("--cid", type=int, default=0, help="Client ID")
    args = parser.parse_args()

    if args.client:
        # Launch a single FedSCClient
        client = FedSCClient(args.cid)
        fl.client.start_client(server_address="127.0.0.1:8080", client=client)
    else:
        # Launch the aggregator ("supernode") server
        aggregator = EnhancedFedSCAggregator()
        fl.server.start_supernode(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=DEFAULT_PARAMS["num_rounds"]),
            aggregator=aggregator,
            # If your version of Flower supports secure aggregation:
            # security_arguments=fl.server.secure_aggregation(**SEC_AGG_CONFIG),
        )

if __name__ == "__main__":
    main()
