"""demo server app."""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp
from flwr.common import Message, RecordDict
import xgboost as xgb
import logging
import time

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    logging.basicConfig(filename="SERVER_LOG_NEW.TXT", level=logging.INFO)
    logging.info("My xgboost version is: ", xgb.__version__)

    expected_clients = 3
    logging.info(f"Waiting for {expected_clients} clients to register...")
    while True:
        node_ids = grid.get_node_ids()
        if len(node_ids) >= expected_clients:
            logging.info(f"All {len(node_ids)} clients connected successfully!")
            break
        logging.info(f"Only {len(node_ids)} clients connected. Retrying in 2 seconds...")
        time.sleep(2)

    # Get the list of all available node IDs
    node_ids = grid.get_node_ids()
    print(f"Querying {len(node_ids)} nodes...")

    # Create a message for each node
    messages = [
        Message(
            content=RecordDict(),
            dst_node_id=node_id,
            message_type="query",
        )
        for node_id in node_ids
    ]

    # Send messages and wait for replies
    replies = grid.send_and_receive(messages)

    # Check the results
    for reply in replies:
        # Extract the version from the RecordDict
        # We look into configs_records["info"] as set by the client
        info = reply.content.configs_records.get("info", {})
        node_version = info.get("xgboost_version", "unknown")
        nvflare_version = info.get("nvflare_version", "unknown")
        logging.info(f"Node {reply.metadata.src_node_id} has xgboost version: {node_version} and nvflare version: {nvflare_version}")
