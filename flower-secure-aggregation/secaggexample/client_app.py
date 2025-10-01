"""secaggexample: A Flower with SecAgg+ app."""

import time

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from secaggexample.task import get_weights, load_data, make_net, set_weights, test, train


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate, timeout, partition_id: int):
        self.net = make_net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)
        # For demonstration purposes only
        self.timeout = timeout
        self.partition_id = partition_id

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        results = {}
        # Use learning rate from config if provided, otherwise use the one from initialization
        lr = config.get("learning_rate", self.lr)
        data_percentage = config.get("data_percentage", 1)
        print("data_percentage: ", data_percentage)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            lr,
            self.device,
            data_percentage=data_percentage,
        )
        # Attach partition id for server-side logging
        results["partition_id"] = self.partition_id
        results["server_round"] = config.get("server_round")
        ret_vec = get_weights(self.net)

        # Force a significant delay for testing purposes
        # if self.is_demo:
        #     if config.get("drop", False):
        #         print(f"Client dropped for testing purposes.")
        #         time.sleep(self.timeout)
        #     else:
        #         print(f"Client uploading parameters: {ret_vec[0].flatten()[:3]}...")
        return ret_vec, len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, accuracy = 0.0, 0.0
        loss, accuracy = test(self.net, self.valloader, self.device)
        print("eval acc: ", accuracy)
        # Include loss and partition id in eval metrics
        return (
            loss,
            len(self.valloader.dataset),
            {"accuracy": accuracy, "loss": loss, "partition_id": self.partition_id},
        )


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
    local_epochs = context.run_config["local-epochs"]
    lr = context.run_config["learning-rate"]
    # For demostrations purposes only
    timeout = context.run_config["timeout"]

    # Return Client instance
    return FlowerClient(trainloader, valloader, local_epochs, lr, timeout, partition_id).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[
        secaggplus_mod,
    ],
)
