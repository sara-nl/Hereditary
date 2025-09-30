# Federated Learning Workshop: CIFAR-10 with Privacy and Security

This repository contains implementations and benchmarks for federated learning on the CIFAR-10 dataset using the Flower framework, with a focus on privacy-preserving techniques including secure aggregation and differential privacy.

## Project Structure

###  CIFAR10_unfederated/
Contains the baseline implementation for non-federated (centralized) training on CIFAR-10 to establish default performance benchmarks.

- `train_cifar10.py` - Training script for centralized CIFAR-10 training

###  flower-secure-aggregation/
Implementation of federated learning with **Secure Aggregation (SecAgg+)** protocol to protect individual client updates during training.

**Key Features:**
- Uses Flower's `SecAggPlusWorkflow` for cryptographic protection
- Implements threshold secret sharing for model updates
- Configurable number of shares and reconstruction threshold

###  flower-differential-privacy/
Implementation of federated learning with **Differential Privacy** to provide formal privacy guarantees for training data.

**Key Features:**
- Noise injection during training using configurable noise multipliers
- Gradient clipping with adjustable norms
- Formal privacy accounting for ε-differential privacy

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- GPU support (optional, for faster training)

### 1. Create Virtual Environment

Create and activate a virtual environment for the project:

```bash
# Create virtual environment
python -m venv flwr-env

# Activate on Linux/macOS
source flwr-env/bin/activate

```

### 2. Install Dependencies
Requirements for all three components have been gathered in a requirements.txt, install them with:
```
pip install -r requirements.txt
```

## Running the Experiments

### Baseline Performance (Centralized)
```bash
cd CIFAR10_unfederated
python train_cifar10.py
```

### Secure Aggregation
```bash
cd flower-secure-aggregation
flwr run . # Uses configuration from pyproject.toml
```

### Differential Privacy
```bash
cd flower-differential-privacy
flwr run . # Uses configuration from pyproject.toml
```

## Configuration

Each implementation can be configured through their respective `pyproject.toml` files:

- **Training parameters**: `num-server-rounds`, `local-epochs`, `learning-rate`, `batch-size`
- **Federation settings**: Number of clients, simulation vs deployment mode
- **Privacy parameters**: Noise multipliers, clipping norms, secret sharing thresholds

## Benchmarking and Results

- Compare baseline centralized performance from `CIFAR10_unfederated/`
- Evaluate privacy-utility tradeoffs in the secure aggregation and differential privacy implementations
- Use TensorBoard for visualization: `tensorboard --logdir=runs/`

## Key Concepts Demonstrated

1. **Federated Learning**: Distributed training without centralizing raw data
2. **Secure Aggregation**: Cryptographic protection of model updates during federation
3. **Differential Privacy**: Formal privacy guarantees through noise injection
4. **Privacy-Utility Tradeoffs**: Balancing model performance with privacy constraints

## Troubleshooting

- Ensure virtual environment is activated before running commands
- Check that all dependencies are installed with `pip list`
- For GPU usage, ensure PyTorch GPU version is installed
- Review logs in each component's `runs/` directory for debugging
