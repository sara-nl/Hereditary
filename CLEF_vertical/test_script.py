from clef_vertical.data import get_labels
from torch.utils.data import DataLoader
from clef_vertical.models import CombinedNetwork
import torch
from flwr.common import Context, array_from_numpy, ArrayRecord, Array, ConfigRecord, ParametersRecord
import pickle

config = {
    # Training parameters
    'random_seed': 42,
    'num_epochs': 100,
    'batch_size': 128,
    'learning_rate': 0.0005,
    'print_freq': 10,
    
    # Network architecture
    'embedding_size': 16,
    'dropout_rate': 0.2,
    
    # Personal Network architecture
    'personal_hidden_sizes': [32],  # List of hidden layer sizes
    
    # Clinical Network architecture
    'clinical_hidden_sizes': [32],  # List of hidden layer sizes
    
    # Combined Network architecture
    'combined_hidden_sizes': [16, 8]  # List of hidden layer sizes
}

net = CombinedNetwork(config)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)

# Convert the state dicts to ArrayRecords
net_state_dict = net.state_dict()
optim_state_dict = optimizer.state_dict()

# METHOD 1: Use the class method to create an ArrayRecord from a PyTorch state dict
model_array_rec = ArrayRecord.from_torch_state_dict(net_state_dict)

# To store the optimizer state dict, we need to pickle it since it contains non-tensor objects
import numpy as np
optimizer_bytes = pickle.dumps(optim_state_dict)
optimizer_array = Array(np.frombuffer(optimizer_bytes, dtype=np.uint8))
optimizer_array_rec = ArrayRecord()
optimizer_array_rec["optimizer_state"] = optimizer_array

# Check that we can retrieve the model state dict
retrieved_model_state_dict = model_array_rec.to_torch_state_dict()
print("Model state dict keys:", list(retrieved_model_state_dict.keys()))

# Check that we can retrieve the optimizer state dict
optimizer_bytes_retrieved = optimizer_array_rec["optimizer_state"].numpy().tobytes()
retrieved_optimizer_state_dict = pickle.loads(optimizer_bytes_retrieved)
print("Optimizer state dict keys:", list(retrieved_optimizer_state_dict.keys()))

# METHOD 2: Store individual tensors of the model state dict
model_array_rec2 = ArrayRecord()
for key, tensor in net_state_dict.items():
    model_array_rec2[key] = Array(tensor.cpu().numpy())

# Retrieve individual tensors
for key in model_array_rec2.keys():
    print(f"Retrieved tensor '{key}' with shape: {torch.from_numpy(model_array_rec2[key].numpy()).shape}")