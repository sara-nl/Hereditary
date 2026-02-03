"""Messaging API example app"""

import time
from flwr.app import Context, Message, RecordDict, ConfigRecord
from flwr.serverapp import Grid, ServerApp

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # 1. Wait for nodes to become available
    print("Waiting for nodes to become available...")
    node_ids = []
    while len(node_ids) < 2:  # Wait for at least 2 nodes for this example
        node_ids = list(grid.get_node_ids())
        if len(node_ids) < 2:
            time.sleep(1)
    
    print(f"Found {len(node_ids)} nodes: {node_ids}")

    # 2. Request averages from each node
    messages = []
    for node_id in node_ids:
        # Construct message
        # message_type="train" will be handled by @app.train() on the client
        message = Message(
            content=RecordDict({}), # Empty content or some config if needed
            message_type="train",
            dst_node_id=node_id,
            group_id="messaging_example",
        )
        messages.append(message)

    print(f"Sending requests to {len(messages)} nodes...")
    
    # 3. Send and receive
    replies = grid.send_and_receive(messages, timeout=30.0)
    
    # 4. Process results
    averages = []
    counts = []
    for reply in replies:
        if reply.has_content():
            # uncomment to check what a response looks like:
            # print(reply.content)
            metrics = reply.content.get("metrics")
            if metrics and "average" in metrics:
                avg = metrics["average"]
                count = metrics["count"]
            if reply.content.get("config"):
                partition_id = reply.content.get("config")["partition-id"]
            averages.append(avg)
            counts.append(count)
            print(f"Received average {avg} from partition {partition_id} with count {count}")

    # 5. Calculate and log global average
    if averages:
        weighted_sum = sum(a * c for a, c in zip(averages, counts))
        total_count = sum(counts)
        global_avg = weighted_sum / total_count
        print(f"\nGlobal average over {len(averages)} partitions: {global_avg}")
    else:
        print("No averages received from nodes.")
