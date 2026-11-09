# Hereditary Workshop 7

**Research use only. Not a medical device.** This software was developed in the HEREDITARY project as a research prototype. It has not been assessed or certified under the EU Medical Device Regulation (EU) 2017/745. It must not be used to diagnose, treat, monitor or make any other decision about individual patients.

In this workshop we will explore how the Flower framework can be integrated in NVFlare and what additional features this provides.

all workshops use UV to install dependencies (see this page on how to install: https://docs.astral.sh/uv/getting-started/installation/)

## Workshop Contents

### HereditaryWorkshop
Core Flower + NVFlare integration demonstrating native Flower SuperNode execution on NVFlare runtime. Includes CIFAR-10 PyTorch example with metrics tracking via TensorBoard, local simulation mode, and distributed Docker deployment.

### multinode-clients
Multi-node federated learning clients for supercomputer environments. Implements SLURM-based distributed training where low resource clients receive global weights, submit FSDP training jobs via `sbatch` to acquire more powerful resources, and return updated model weights for FedAvg aggregation.

### predeployed-experiments
Demonstrates server-predeployed-flwr patterns for Flower jobs on NVFlare. By doing so, clients can be prevented from submitting unvetted code, while still allowing researchers to run code with hyperparameters of their choice. 

### runtime-dependency
Explores NVFlare's `allow_runtime_dependency_installation` feature for Flower workflows. Allows experiments with different Python package versions (e.g., xgboost 3.1.0 vs 3.2.0) to run on the same federated network without restarting nodes.


## License and intended use

This code is released under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE) for attribution.
It is educational workshop material for federated learning infrastructure in the HEREDITARY project (Horizon Europe, GA 101137074).

Report misuse or security issues to douwe.vanderwal@surf.nl.
