import torch
import torch.nn as nn


class PersonalNetwork(nn.Module):
    def __init__(self, input_size, config):
        super(PersonalNetwork, self).__init__()
        layers = []

        # Input layer
        prev_size = input_size
        for hidden_size in config["personal_hidden_sizes"]:
            layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(config["dropout_rate"])])
            prev_size = hidden_size

        # Output embedding layer
        layers.extend([nn.Linear(prev_size, config["embedding_size"]), nn.ReLU()])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Neural network for clinical data
class ClinicalNetwork(nn.Module):
    def __init__(self, input_size, config):
        super(ClinicalNetwork, self).__init__()
        layers = []

        # Input layer
        prev_size = input_size
        for hidden_size in config["clinical_hidden_sizes"]:
            layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(config["dropout_rate"])])
            prev_size = hidden_size

        # Output embedding layer
        layers.extend([nn.Linear(prev_size, config["embedding_size"]), nn.ReLU()])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Combined network for final prediction
class CombinedNetwork(nn.Module):
    def __init__(self, config):
        super(CombinedNetwork, self).__init__()
        layers = []

        # Input is concatenated embeddings
        prev_size = config["embedding_size"] * 2
        for hidden_size in config["combined_hidden_sizes"]:
            layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(config["dropout_rate"])])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x1, x2):
        combined = torch.cat((x1, x2), dim=1)
        return self.model(combined)
