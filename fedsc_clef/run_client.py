# fedsc_clef/run_client.py

import argparse
import flwr as fl
from .client import FedSCClient

def main_client(cid: int):
    client = FedSCClient(cid)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, default=0, help="Client ID")
    args = parser.parse_args()

    main_client(args.cid)
