import flwr as fl
import numpy as np

# Custom Federated Analytics Client
class AnalyticsClient(fl.client.NumPyClient):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def get_parameters(self):
        """Not used for analytics purposes, so return None."""
        return None

    def fit(self, parameters, config):
        """No training involved, return None."""
        return None, 0, {}

    def evaluate(self, parameters, config):
        """Perform data analytics on local dataset."""
        local_data = self.data_loader.load_data()

        # Example: Calculate the mean of a specific feature locally (federated analytics)
        feature_column = config.get("feature_column", 0)
        local_feature_mean = np.mean(local_data[:, feature_column])

        # Return the local analysis result to the server
        return local_feature_mean, len(local_data), {}

# Custom Aggregation Function on Server
def weighted_avg(metrics):
    """Weighted average of clients' local analytics results."""
    total_sum = 0.0
    total_count = 0
    for metric, num_examples in metrics:
        total_sum += metric * num_examples
        total_count += num_examples
    return total_sum / total_count if total_count > 0 else 0.0

# Define Federated Strategy for Analytics
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.0,  # No model training, analytics only
    fraction_eval=1.0,  # Use all clients for analytics
    min_eval_clients=3,  # Minimum clients participating in analytics
    evaluate_metrics_aggregation_fn=weighted_avg,  # Custom aggregation
)

# Start the Flower server for federated analytics
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config={"num_rounds": 1},
    strategy=strategy
)

# Sample client code to connect and perform federated analytics
if __name__ == "__main__":
    from fets_data_provider import FetsDataProvider  # Import FeTS dataset loader
    
    data_loader = FetsDataProvider()
    client = AnalyticsClient(data_loader=data_loader)
    
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
