"""clef_vertical: A Flower / PyTorch app."""

import pickle

import torch
from flwr.common import Context, EvaluateIns, FitIns, Parameters, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, Strategy
from torch import optim
from torch.utils.data import DataLoader

from clef_vertical.data import get_labels
from clef_vertical.models import CombinedNetwork
from clef_vertical.network_types import NetworkType
from clef_vertical.utils import get_model_config


class SotaStrategy(Strategy):
    def __init__(self, *args, run_id: str, min_available_clients=2, model_config, **kwargs):
        super().__init__(*args, **kwargs)
        print("model config found: ", model_config)
        # model init
        self.net = CombinedNetwork(model_config)
        self.batch_size = model_config["batch_size"]
        self.fraction_fit = 1
        self.fraction_evaluate = 1
        self.min_available_clients = min_available_clients
        self.min_fit_clients = min_available_clients
        self.min_evaluate_clients = min_available_clients
        self.combined_optimizer = optim.Adam(self.net.parameters(), lr=model_config["learning_rate"])
        self.criterion = torch.nn.MSELoss()
        # data
        train_dataset, test_dataset, y_train_original, y_test_original = get_labels()
        self.y_train_original = y_train_original
        self.y_test_original = y_test_original
        self.trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = test_dataset
        self.testloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # progress keeping
        self.all_batches = []
        for batch in self.trainloader:
            self.all_batches.append(batch)
        self.curr_batch = 0
        self.personal_gradients = None
        self.clinical_gradients = None

    def initialize_parameters(self, client_manager):
        """No need to sync clients, as they have different models in vertical FL"""
        return None

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        batch_idx = self.all_batches[self.curr_batch][0].flatten().numpy().tolist()
        batch_idx_str = ",".join([str(idx) for idx in batch_idx])
        personal_gradient_bytes = pickle.dumps(self.personal_gradients)
        clinical_gradient_bytes = pickle.dumps(self.clinical_gradients)
        config = {
            "batch_idx": batch_idx_str,
            "round": server_round,
            "personal_gradients": personal_gradient_bytes,
            "clinical_gradients": clinical_gradient_bytes,
        }
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            print(f"Failures: {failures}")
            return None, {}

        self.net.train()

        # Load the embeddings from the results
        personal_embedding = None
        clinical_embedding = None
        for _, fitres in results:
            if fitres.metrics["type"] == NetworkType.PERSONAL.value:
                personal_embedding_bytes = fitres.metrics["embedding"]
                personal_embedding = pickle.loads(personal_embedding_bytes)
                personal_embedding = personal_embedding.clone().detach().requires_grad_(True)

            elif fitres.metrics["type"] == NetworkType.CLINICAL.value:
                clinical_embedding_bytes = fitres.metrics["embedding"]
                clinical_embedding = pickle.loads(clinical_embedding_bytes)
                clinical_embedding = clinical_embedding.clone().detach().requires_grad_(True)

        # Finish the forward pass and calculate the loss
        prediction = self.net(personal_embedding, clinical_embedding)
        targets = self.all_batches[self.curr_batch][1]
        loss = self.criterion(prediction, targets)
        loss.backward()
        self.combined_optimizer.step()
        self.combined_optimizer.zero_grad()

        # Get gradients for embeddings
        self.personal_gradients = personal_embedding.grad.numpy()
        self.clinical_gradients = clinical_embedding.grad.numpy()
        metrics = {}
        self.curr_batch += 1

        if self.curr_batch >= len(self.all_batches):
            self.curr_batch = 0
        return None, metrics

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure the next round of evaluation."""
        # Parameters and config, no config needed as we evaluate the whole dataset
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if failures:
            print(f"Failures: {failures}")
            return None, {}

        # collect embeddings from results
        personal_embedding = None
        clinical_embedding = None
        for _, evalres in results:
            if evalres.metrics["type"] == NetworkType.PERSONAL.value:
                personal_embedding = evalres.metrics["embedding"]
            elif evalres.metrics["type"] == NetworkType.CLINICAL.value:
                clinical_embedding = evalres.metrics["embedding"]

        # Deserialize the embeddings
        clinical_embedding = pickle.loads(clinical_embedding)
        personal_embedding = pickle.loads(personal_embedding)

        # Perform the forward pass and calculate the loss
        self.net.eval()
        prediction = self.net(personal_embedding, clinical_embedding)
        targets = self.test_dataset.tensors[1]
        loss = self.criterion(prediction, targets)
        print("loss: ", loss)
        return loss.item(), {"loss": loss.item()}

    def evaluate(self, server_round, parameters):
        # We don't evaluate on the server as we rely on embeddings from the clients
        pass


def server_fn(context: Context):
    # Read from config
    print(context.run_config)
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    model_config = get_model_config(context)

    # Define strategy
    strategy = SotaStrategy(
        run_id=context.run_id,
        min_available_clients=2,
        model_config=model_config,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
