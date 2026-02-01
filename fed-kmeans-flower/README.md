# fed-kmeans-flower: Federated K-means Clustering with Flower

This project implements federated k-means clustering using the Flower framework's messaging API.

The implementation is based on the following paper:
- **Federated K-Means Clustering**, published in *Advanced Data Mining and Applications* (ADMA 2024). [Link to paper](https://link.springer.com/chapter/10.1007/978-3-031-78166-7_8)


## Install using Docker:
Go to the `docker` directory and and follow the readme to build and access the docker container. 


## Running the experiment
Assuming you're using docker, everything should already be set up. If you're doing a manual installation, please follow the instructions below. For more information on how the test data is generated and what the full workflow is, please also check the sections below. 

### Run with the Simulation Engine

In the `fed-kmeans-flower` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```
### Running using the Deployment Engine 

If you want to run the full experiment yourself (potentially on different devices) then run the following commands in separate terminals to start the server and clients:

If you are running the superlink on a remote server, ensure that ports 9091, 9092 and 9093 are open for incoming TCP connections, and replace 127.0.0.1 with the IP address of the remote server.


```bash
# Start the server
flower-superlink --insecure

# Start the clients
flower-supernode --insecure --clientappio-api-address 127.0.0.1:9095 --node-config "num-partitions=3 partition-id=0"
flower-supernode --insecure --clientappio-api-address 127.0.0.1:9096 --node-config "num-partitions=3 partition-id=1"
flower-supernode --insecure --clientappio-api-address 127.0.0.1:9097 --node-config "num-partitions=3 partition-id=2"
```

The partitions are mapped as follows: {0: "T", 1: "L", 2: "U"}

The IP for the Hereditary server can be found on the group page.


## Manual installation
The dependencies are listed in the `pyproject.toml` and you can install them as follows:

```bash
pip install -e .
```
Python3.11 is recommend and ensure that you have the data downloaded and extracted to a folder, and that the `CLEF_DATA_PATH` environment variable is set, refer to the Dockerfile for how it's set.  

## Data Generation

Before running the federated k-means clustering on synthetic data, you need to generate synthetic data partitions for the clients. (This is already done automatically when building the Docker container for 2d and 10d data)

### Generate Data

Use the `generate_data.py` script to create synthetic clustered data and partition it across multiple clients (If you're using the docker image, this data has already been generated for 2d and 10d data):

```bash
# Generate data for 5 clients with default settings
python generate_data.py --num-clients 3 --samples-per-client 100 --output-dir ./data

# Generate data with custom parameters
python generate_data.py \
  --num-clients 3 \
  --samples-per-client 100 \
  --num-features 2 \
  --num-clusters 3 \
  --cluster-std 1.0 \
  --non-iid-factor 0.5 \
  --overlap-factor 0.0 \
  --density-variation 0.0 \
  --random-state 42 \
  --output-dir ./data
```

### Data Generation Parameters

- `--num-clients`: Number of federated clients (default: 5)
- `--samples-per-client`: Approximate samples per client (default: 100)
- `--num-features`: Number of features/dimensions (default: 2)
- `--num-clusters`: Number of clusters (default: 3)
- `--cluster-std`: Standard deviation of clusters (default: 1.0)
- `--non-iid-factor`: Degree of non-IID distribution, 0.0 = IID, 1.0 = max non-IID (default: 0.5)
- `--overlap-factor`: Degree of cluster overlap, 0.0 = no overlap, 1.0 = high overlap (default: 0.0)
- `--density-variation`: Variation in cluster densities, 0.0 = uniform, 1.0 = max variation (default: 0.0)
- `--random-state`: Random seed for reproducibility (default: 42)
- `--output-dir`: Directory to save generated data files (default: ./data)

### Generated Files

The script creates the following files in the output directory:

- `client_0.npz`, `client_1.npz`, ...: Individual client data partitions
- `metadata.json`: Dataset metadata including configuration and statistics
- `cluster_centers.npy`: True cluster centers for evaluation

### Configure Data Path
Set the `TEST_DATA_PATH` and/or `CLEF_DATA_PATH` to allow the data to load properly. 
Update the `data-source` in `pyproject.toml` to select which data you would like to use. 

```toml
[tool.flwr.app.config]
data-source = "test_data"  # test_data or ALS
log-dir = "./logs"  # Directory for experiment logs and means tracking
```


## Federated K-Means Workflow

The complete federated k-means workflow consists of the following phases, ensuring data remains local while enabling global insights:

1. **Client Initialization & Handshake**
   - Each client loads its local data partition.
   - Clients register with the server, sharing metadata (e.g., number of samples, feature dimensions, partition id) but no raw data.
   - Data compatibility is verified across the federation.

2. **Federated Preprocessing (Global Normalization)**
   - **Local Stats**: Clients compute `sum`, `sum_sq`, and `count` for their local features.
   - **Aggregation**: The server aggregates these to compute global `mean` and `standard deviation`.
   - **Normalization**: Clients receive the global parameters and normalize their data locally. This step is crucial for PCA and K-Means to treat all features fairly.

3. **Federated PCA (Dimensionality Reduction & Visualization)**
   - If the data has more than 2 dimensions, the system performs federated PCA:
     - **Covariance**: Clients compute local covariance matrices of their normalized data.
     - **Global Components**: The server aggregates the matrices and performs Eigen decomposition to find global principal components.
     - **Projection**: Clients project their high-dimensional data into a lower-dimensional subspace (typically 2D) for visualization.
   - Based on: [Federated PCA paper](https://doi.org/10.1093/bioadv/vbac026)

4. **Federated K-Means Clustering**
   - The server initializes `k` global cluster centers (centroids).
   - **Iterative Loop**:
     - **Assignment**: Clients assign local points to the nearest global centroids.
     - **Local Update**: Clients compute new local means based on these assignments.
     - **Privacy Guard**: The "Privacy Threshold" ensures that local means representing too few samples are not shared.
     - **Global Update**: The server aggregates valid local means to update global centroids.
   - **Convergence**: Rounds continue until centroids stabilize or the maximum iteration limit is reached.
   - Based on: [Federated K-Means paper](https://link.springer.com/chapter/10.1007/978-3-031-78166-7_8)

5. **Final Evaluation & Post-processing**
   - **Metrics**: The server computes global metrics like Inertia, Silhouette Score, and Adjusted Rand Index.
   - **Denormalization**: Cluster centroids are transformed back to the original data scale.
   - **Visualization**: An animated video is generated showing the entire clustering process (if enabled).


## Experiment Logging

The federated k-means implementation includes comprehensive logging of cluster means at each step:

### Logging Structure

Each experiment run creates a timestamped directory under the configured `log-dir`:

```
logs/
└── experiment_20251107_160054/
    ├── global_means/           # Global cluster means per round
    │   ├── round_000_global_means.npy
    │   ├── round_000_metadata.json
    │   ├── round_001_global_means.npy
    │   ├── round_001_metadata.json
    │   └── ...
    ├── local_means/            # Local cluster means per client per round
    │   ├── client_<id>/
    │   │   ├── round_001_local_means.npy
    │   │   ├── round_001_sample_counts.npy
    │   │   ├── round_001_metadata.json
    │   │   └── ...
    │   └── ...
    ├── client_logs/            # Client log files
    │   ├── client_<id>.log
    │   └── ...
    ├── server_logs/            # Server log files
    │   └── server.log
    ├── experiment_summary.json # Experiment configuration and results
    └── federated_kmeans_metrics.json
```
The client logs are only stored on the server if the experiment is in simulation mode, otherwise they will be located on the specific client. 

### What Gets Logged

**Global Means (Server):**
- Logged after each round (including initial random initialization at round 0)
- Includes convergence metrics, participating clients, and cluster statistics
- Saved as both `.npy` (numpy array) and `.json` (metadata)

**Local Means (Clients):**
- Logged after each local clustering step
- Includes both pre- and post-privacy filtering results
- Contains sample counts per cluster
- Saved as `.npy` files with accompanying metadata

**Experiment Summary:**
- Configuration parameters (k_global, max_iterations, etc.)
- Final results (total rounds, convergence status, final means shape)
- Convergence history across all rounds

### Analyzing Logged Data

You can load and analyze the logged means using numpy:

```python
import numpy as np
import json

# Load global means from a specific round
global_means = np.load('logs/experiment_20251107_160054/global_means/round_005_global_means.npy')

# Load metadata
with open('logs/experiment_20251107_160054/global_means/round_005_metadata.json') as f:
    metadata = json.load(f)
    print(f"Round {metadata['round_number']}: {metadata['num_clusters']} clusters")
    print(f"Convergence change: {metadata.get('convergence_change', 'N/A')}")

# Load local means from a specific client
client_means = np.load('logs/experiment_20251107_160054/local_means/client_0/round_005_local_means.npy')
sample_counts = np.load('logs/experiment_20251107_160054/local_means/client_0/round_005_sample_counts.npy')
```

When the experiment is ran in simulation mode, or all results have been gathered centrally, a video can be generated how the clustering process unfolds, using `visualize_federated_kmeans.py`.

## Visualization

Create animated videos showing the federated k-means clustering process across all rounds.

### Install Visualization Dependencies

```bash
pip install -e ".[viz]"
```

### Create Visualization Video

After running an experiment, visualize the clustering process:

```bash
# Basic usage - creates video from experiment logs
python visualize_federated_kmeans.py --experiment-dir logs/experiment_20251107_160527

# Custom output filename and frame rate
python visualize_federated_kmeans.py \
    --experiment-dir logs/experiment_20251107_160527 \
    --output my_clustering_video.mp4 \
    --fps 1
```

### Visualization Features

The video shows three types of frames for each round:

1. **Global Means Frame**: Shows each client's data with global cluster means overlaid
2. **Local Clustering Frame**: Shows how each client computes local means from global means
3. **Aggregation Frame**: Shows how local means are aggregated into new global means

Each frame includes:
- Client data points (if available)
- Cluster centers with labels
- Sample counts per cluster
- Convergence metrics
- Movement arrows showing how means evolve

### Visualization Parameters

- `--experiment-dir`: Path to experiment directory (required)
- `--output`: Output video filename (default: `federated_kmeans.mp4`)
- `--fps`: Frames per second (default: 2)
- `--cleanup`: Delete frame files after creating video
- `--frames-only`: Only create frames without generating video

> **Tip:** Your `pyproject.toml` file can define more than just the dependencies of your Flower app. You can also use it to specify hyperparameters for your runs and control which Flower Runtime is used. By default, it uses the Simulation Runtime, but you can switch to the Deployment Runtime when needed.
> Learn more in the [TOML configuration guide](https://flower.ai/docs/framework/how-to-configure-pyproject-toml.html).





## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)


You might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation. Refer to the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide in the documentation for advice on how to optimize your simulations.

You can run Flower on Docker too! Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.