import torch
import syft as sy
from torch import nn
from torch.utils.data import DataLoader
from fets_data_provider import FetsDataProvider  # Import FeTS dataset loader

# Initialize PySyft hook
hook = sy.TorchHook(torch)

# Define a virtual worker setup for federated analytics
client_1 = sy.VirtualWorker(hook, id="client_1")
client_2 = sy.VirtualWorker(hook, id="client_2")
client_3 = sy.VirtualWorker(hook, id="client_3")

# Load local datasets on each client (FeTS data)
data_loader_1 = FetsDataProvider().load_data().send(client_1)
data_loader_2 = FetsDataProvider().load_data().send(client_2)
data_loader_3 = FetsDataProvider().load_data().send(client_3)

# Perform federated analytics securely using SMPC
def secure_federated_sum(data_loaders):
    """Compute the secure sum of a feature column across clients."""
    feature_sum = 0
    for data_loader in data_loaders:
        local_data = data_loader.get()
        local_feature_sum = local_data[:, 0].sum()  # Sum over first column feature
        feature_sum += local_feature_sum.fix_precision().share(client_1, client_2, client_3)

    # Get back the securely computed sum in clear text
    return feature_sum.get().float_precision()

# Run the secure federated sum
secure_sum = secure_federated_sum([data_loader_1, data_loader_2, data_loader_3])
print(f"Securely computed federated sum: {secure_sum}")

# Federated Mean Calculation
def secure_federated_mean(data_loaders):
    """Compute the secure mean of a feature column across clients."""
    total_sum = secure_federated_sum(data_loaders)
    total_count = sum([len(dl.get()) for dl in data_loaders])
    
    secure_mean = total_sum / total_count
    return secure_mean

# Run secure mean computation
secure_mean = secure_federated_mean([data_loader_1, data_loader_2, data_loader_3])
print(f"Securely computed federated mean: {secure_mean}")
