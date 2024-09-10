# 1. Setting privacy parameters
# Define epsilon (privacy budget) and delta (acceptable risk of privacy breach)
epsilon = 1.0  # Privacy budget: controls the trade-off between privacy and accuracy
delta = 1e-5   # Probability of violating the privacy guarantee

# Import necessary modules
import flwr as fl
from opacus import PrivacyEngine

# Client-side code
class DPClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, privacy_engine):
        self.model = model  # Local model
        self.train_loader = train_loader  # Local data loader
        
        # 2. Integrating DP into the training process
        # Attach the PrivacyEngine to the client model for adding noise
        self.privacy_engine = PrivacyEngine(
            self.model,
            batch_size=64,  # Batch size
            sample_size=len(train_loader.dataset),  # Size of local data
            epochs=1,  # Number of training epochs
            target_epsilon=epsilon,  # Epsilon (privacy budget)
            target_delta=delta,  # Delta (privacy tolerance)
            max_grad_norm=1.0  # Gradient clipping for DP
        )
        self.privacy_engine.attach()

    def get_parameters(self):
        """Return model parameters to the server."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train model on local data with differential privacy."""
        self.model.set_weights(parameters)  # Update local model with global parameters
        # Local training with differential privacy
        for epoch in range(config["epochs"]):
            for batch in self.train_loader:
                loss = self.model.train_step(batch)  # Train on local batch
                self.privacy_engine.step()  # Apply differential privacy after each step
        return self.model.get_weights(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on local data."""
        self.model.set_weights(parameters)  # Update local model with global parameters
        loss, accuracy = self.model.evaluate(self.train_loader)
        return loss, len(self.train_loader.dataset), {"accuracy": accuracy}


# 3. Configuring the server for DP
# Server-side code
def evaluate_round(server_round, parameters, config):
    """Server-side evaluation to track cumulative privacy loss."""
    loss, accuracy = fl.server.strategy.FedAvg.evaluate(parameters, config)
    
    # 4. Monitoring and adjusting privacy levels
    # Track and monitor cumulative privacy loss over rounds
    cumulative_epsilon, cumulative_delta = server_round.monitor_privacy_loss()
    
    if cumulative_epsilon > threshold:
        # Adjust privacy parameters or stop training
        print(f"Warning: Privacy budget exceeded in round {server_round}")

    return loss, accuracy


# Define the federated averaging strategy with differential privacy
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=10,  # Minimum number of clients participating in training
    min_available_clients=10,  # Minimum number of clients available
    evaluate_fn=evaluate_round,  # Custom evaluation function
)

# 4. Monitoring and adjusting privacy levels (cumulative loss tracked by server)
class DPServer(fl.server.Server):
    def __init__(self, strategy):
        super().__init__(strategy=strategy)
        self.cumulative_epsilon = 0.0  # Track cumulative epsilon over rounds
    
    def monitor_privacy_loss(self):
        """Track and return cumulative privacy loss."""
        # Accumulate privacy loss across rounds
        self.cumulative_epsilon += self.get_privacy_loss()
        return self.cumulative_epsilon, delta

# Start the Flower server with differential privacy support
fl.server.start_server("0.0.0.0:8080", strategy=strategy, server=DPServer(strategy))

