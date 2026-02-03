"""Messaging API example app"""

import random
from flwr.app import Context, Message, RecordDict, ConfigRecord, MetricRecord
from flwr.clientapp import ClientApp

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Sample 10 numbers and return their average."""
    # uncomment to check what the context looks like
    # print(context)
    partition_id = context.node_config["partition-id"]
    # Sample 10 numbers
    n_numbers = 10
    numbers = [random.uniform(0, 100) for _ in range(n_numbers)]
    
    # Log to console
    print(f"Node {context.node_id}: sampled numbers: {numbers}")
    
    # Calculate average
    avg = sum(numbers) / len(numbers)
    
    # Construct and return reply Message
    # We use ConfigRecord for simple values
    metrics = MetricRecord({"average": avg, "count": n_numbers})
    configrecord = ConfigRecord({"partition-id": partition_id})
    content = RecordDict({"metrics": metrics, "config": configrecord})
    
    return Message(content=content, reply_to=msg)
