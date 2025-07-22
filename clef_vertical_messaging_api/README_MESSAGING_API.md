# CLEF Vertical Federated Learning - Messaging API

This project implements vertical federated learning for the CLEF dataset using Flower's messaging API instead of the traditional strategy-based approach.

## Overview

This is a converted version of the original `CLEF-vertical` project that uses Flower's messaging API for communication between server and clients. The core functionality remains the same - vertical federated learning where:

- **Personal Client**: Handles personal/demographic data and generates embeddings
- **Clinical Client**: Handles clinical measurements and generates embeddings  
- **Server**: Coordinates training, combines embeddings, computes gradients, and manages the combined model

## Key Differences from Strategy-Based Approach

### Original Implementation (`CLEF-vertical`)
- Uses `SotaStrategy` class with `aggregate_fit()` and `aggregate_evaluate()` methods
- Clients communicate through the strategy's parameter exchange mechanism
- Server coordinates through centralized strategy methods

### Messaging API Implementation (This Project)  
- Uses direct message passing between server and clients
- Server sends specific messages (`load_data`, `forward_pass`, `backward_pass`, `evaluate_model`)
- Clients respond to messages with embeddings and status updates
- More explicit control over the training flow

## Architecture

### Message Flow

1. **Initialization**: 
   - Server sends `load_data` messages to assign network types to clients
   - Clients load data and initialize models based on assigned type

2. **Training Round**:
   - Server sends `forward_pass` with batch indices to both clients
   - Clients compute embeddings and return them
   - Server combines embeddings, computes loss, calculates gradients
   - Server sends `backward_pass` with gradients to respective clients
   - Clients apply gradients and update their models

3. **Evaluation**:
   - Server sends `evaluate_model` to both clients
   - Clients compute embeddings for validation data
   - Server combines embeddings and computes evaluation loss

### Project Structure

```
clef_vertical_messaging_api/
├── clef_vertical_messaging_api/
│   ├── __init__.py
│   ├── client_app.py          # Messaging API client implementation
│   ├── server_app.py          # Messaging API server implementation  
│   ├── task.py                # Utility functions for data/model handling
│   ├── data.py                # Data loading functions (copied from original)
│   ├── models.py              # Neural network models (copied from original)
│   ├── network_types.py       # Enum definitions (copied from original)
│   └── utils.py               # Configuration utilities (copied from original)
├── pyproject.toml             # Updated with messaging API dependencies
```

## Installation

1. Navigate to the project directory:
```bash
cd clef_vertical_messaging_api/clef-vertical-messaging-api
```

2. Install in editable mode:
```bash
pip install -e .
```


## Configuration

The configuration is defined in `pyproject.toml`:

```toml
[tool.flwr.app.config]
num-server-rounds = 200
learning-rate = 0.001
batch-size = 128
embedding-size = 16
dropout-rate = 0.2
personal-hidden-sizes = "32"
clinical-hidden-sizes = "32" 
combined-hidden-sizes = "16,8"
required-nodes = 2
node-timeout = 60.0
training-timeout = 30.0
num-epochs = 5
```

## Running the Application

### Local Simulation
```bash
flwr run . local-simulation
```

### Distributed Setup
1. Start the server:
```bash
flwr server-app clef_vertical_messaging_api.server_app:app
```

2. Start personal client:
```bash
flwr client-app clef_vertical_messaging_api.client_app:app --node-config partition-id=0
```

3. Start clinical client:
```bash  
flwr client-app clef_vertical_messaging_api.client_app:app --node-config partition-id=1
```

## Key Implementation Details

### Client State Management
The client maintains persistent state between message calls:
- Model and optimizer states
- Dataset references  
- Current batch information
- Computed embeddings

### Message Types

- **`load_data`**: Initialize client with network type and load appropriate data
- **`forward_pass`**: Compute embeddings for specified batch indices
- **`backward_pass`**: Apply received gradients and update model
- **`evaluate_model`**: Generate embeddings for evaluation dataset

### Serialization
- Uses `pickle` for gradient and embedding serialization



