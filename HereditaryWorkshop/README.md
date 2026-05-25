# Hereditary Workshop: NVIDIA FLARE + Flower Integration

This project demonstrates how to run a [Flower](https://flower.ai/) federated learning application natively on the [NVIDIA FLARE](https://nvflare.readthedocs.io/) runtime. It combines Flower's ease of creating federated applications with NVFlare's robust orchestration, job scheduling, and enterprise-grade deployment capabilities.

## Prerequisites

- **Python 3.10**
- **[uv](https://docs.astral.sh/uv/)** (Fast Python package installer and resolver)
- **Docker & Docker Compose** (for simulating a distributed network)

## Project Structure

- **`flwr_pt_tb/`**: The core Flower application.
  - `client.py`: The Flower client code (`ClientApp`). It streams metrics using NVFlare's `SummaryWriter` with a graceful fallback to local TensorBoard.
  - `server.py`: The Flower server code (`ServerApp`), defining the aggregation strategy.
  - `task.py`: The PyTorch neural network definition, dataloaders, and train/test functions.
- **`job.py`**: A build script that uses the NVFlare `FlowerJob` API to package the Flower code into an NVFlare-compatible job.
- **`project.yml`**: Auto-generated from `build_distribution.sh`. The NVFlare network topology (server, client sites, admin).
- **`docker-compose.yml`**: Auto-generated from `build_distribution.sh`. Configures the multi-container Docker environment.
- **`build_distribution.sh`**: The single master script that does everything: generates `project.yml`, provisions cryptographic certificates, builds the Flower job, generates `docker-compose.yml`, and packages everything into `dist/` for deployment.
- **`Dockerfile.nvflare`**: The Docker runtime image used by all server and client containers.
- **`custom4client/`**: Custom security handler components injected into the client and server workspaces during provisioning.
- **`pyproject.toml`**: Defines dependencies and Flower app configurations for the `cifar10_flower` job.
- **`uv.lock`**: Dependency lockfile managed by `uv`.

## Workshop participant

Build and run the NVFlare Docker container directly:
1. Copy the site-X.zip you have received from the workshop organisers to this folder, and then edit the Dockerfile.nvflare to replace site-1.zip with your site-X.zip.

2. **Build the image:**
   ```bash
   docker build -f Dockerfile.nvflare -t hereditary-workshop .
   ```

3. **Run the container:**
   ```bash
   docker run -it --name hereditary-workshop hereditary-workshop bash
   ```
   You can consider adding `--gpus all` if your machine has a GPU that you wish to make available to the container.
   
   If you get an error that the container name is already in use, remove it first:
   ```bash
   docker rm hereditary-workshop
   ```

4. **Inside the container, start NVFlare:**
   ```bash
   cd /workspace 
   cd site-YOUR-SITE-NUMBER
   ./startup/sub_start.sh
   ```

## How to Build and Run

To prepare your network configuration, cryptographic certificates, Flower job package, and Docker configuration, run the distribution builder script:

```bash
./build_distribution.sh
```

You can also customise the run directly with flags:
```bash
./build_distribution.sh --server server1 --sites hospital-a hospital-b hospital-c
```

This script performs the following steps automatically:
1. **Generates `project.yml`** with the server name/IP and site names.
2. **Provisions certificates** — generates unique cryptographic startup kits for each participant.
3. **Builds the Flower NVFlare job** package.
4. **Generates `docker-compose.yml`** dynamically to match the server and site names.
5. **Packages everything** into the `dist/` directory for deployment.

---

## Admin Workflow

Once your distributed nodes (Hospitals/Snellius) are running and "waiting for job," use these commands on your Mac to start the training:

1. **Build the Job Package:**
   ```bash
   uv run python job.py
   ```
2. **"Login" Admin Identity:**
   (Tells your CLI to talk to your server with your certificates)
   ```bash
   uv run nvflare config -d ./nvflare_workspace/provision/hereditary_workshop/prod_00/surf@hereditary.com/startup
   ```
3. **Submit Job:**
   ```bash
   uv run nvflare job submit -j ./nvflare_workspace/jobs/cifar10_flower
   ```

---


### Manual Step-by-Step Execution

If you prefer to run the steps separately:

1. **Build everything (config, certs, job, compose file):**
   ```bash
   ./build_distribution.sh --server server1 --sites site-1 site-2 site-3
   ```

2. **Start the NVFlare Docker network:**
   ```bash
   docker compose up --build -d
   ```

3. **Configure the NVFlare admin client (in a new terminal):**
   ```bash
   uv run nvflare config -d ./nvflare_workspace/provision/hereditary_workshop/prod_00/surf@hereditary.com/startup
   ```

4. **Submit the exported job:**
   ```bash
   uv run nvflare job submit -j ./nvflare_workspace/jobs/cifar10_flower
   ```


## Key Technical Integration Details
- **Native Flower Executor:** The job uses NVFlare's `FlowerExecutor`. This executor builds a direct communication bridge between NVFlare and the Flower SuperNode, completely bypassing the standard `nvflare.client` API. As a result, you do *not* need to run `flare.init()` inside your Flower client code.
- **Metrics Tracking:** The client script attempts to use `NvFlareSummaryWriter` to stream TensorBoard metrics to the NVFlare server. Because it catches `RuntimeError` exceptions, it smoothly falls back to a standard local TensorBoard writer without crashing if the NVFlare client API pipe is unavailable.
- **Unmodified Training Code:** The actual ML training logic in `task.py` and the FL definitions in `client.py` and `server.py` are standard Flower code. NVFlare's `FlowerJob` API effortlessly wraps and deploys it without requiring a major codebase rewrite.

## Distributed Deployment (Physical Machines)

When deploying to real separate machines across the internet:
1. **Run `build_distribution.sh` with your real server public IP and custom site names.** This regenerates everything — `project.yml`, certificates, and the `dist/` zip packages — all in one command:
   ```bash
   ./build_distribution.sh --server 145.38.207.216 --sites hospital-a hospital-b hospital-c
   ```
2. **Send the Files:** Open the `dist/` folder. Send `hospital-a.zip` to Partner Hospital a, `hospital-b.zip` to Partner Hospital b, etc. *(Important: Never share certificates between sites!)*
3. **Partners unzip and run — no Docker needed.** They simply unzip their package and run the startup script directly:
   ```bash
   unzip hospital-a.zip -d .
   cd hospital-a/startup
   ./sub_start.sh
   ```
   Their node will securely connect to your server over the internet. The only requirement is Python 3.10 with `nvflare` and `flwr` installed.

---

## Site Operations & Process Management

Once a site (Server or Client) is started via `./sub_start.sh`, it runs as a background daemon.

### 1. The PID File (`daemon_pid.fl`)
NVFlare creates a `daemon_pid.fl` file in the `startup/` directory to track the running process.
- **To Stop:** Run `./sub_stop.sh` in the same directory.
- **To Restart:** Run `./sub_stop.sh` then `./sub_start.sh`.
- **Troubleshooting:** If the node fails to start and says "already running" but you know it isn't, delete the `daemon_pid.fl` file manually.

### 2. Workspace Folders
On the first run, NVFlare generates folders alongside `startup/`:
- `transfer/`: Contains the job data and code downloaded from the server.
- `local/`: Contains site-specific configuration.
- `log.txt`: The primary log file for debugging connectivity or execution errors.

---

> **Note:** The `docker-compose.yml` is automatically overwritten each time you run `build_distribution.sh` and is only used for the local Docker simulation. It is not needed for real deployments.

## Troubleshooting

- **Clients cannot connect to server (`Cannot connect to host server1:8002`):**
  This happens when you change site names or the server name between runs. The old server container stays alive on the old Docker network while new client containers land on a new network, so Docker DNS can't route between them. **Fix:** Always ensure you run `docker compose down` before running `build_distribution.sh` to tear down stale containers and networks. If you still see this after a manual `docker compose up`, run `docker compose down && docker network prune -f` first.

- **Out Of Memory (OOMKilled) during Local Docker Run:**
  If the training job crashes with `exit code 1` and aborts on the clients, your Docker daemon likely ran out of memory. 4 PyTorch instances (1 server + 3 clients) loading CIFAR-10 concurrently require significant RAM.
  - **Docker Desktop:** Increase your memory limit to at least **8GB** in Settings > Resources.
  - **Colima:** Restart with more resources: `colima stop && colima start --memory 8 --cpu 4`.
