# NVFlare + Flower: Runtime Dependency Installation

This project demonstrates how to use the `allow_runtime_dependency_installation` feature of NVFlare with Flower (FLwr) for federated learning workflows, which allows you to have experiments that require different dependencies without having to restart the federated network. 

## Contents

- `fs26-demo/` - example using NVFlare 2.8.0rc2 and Flower 1.29.0+ with xgboost 3.2.0
- `fs26-demo-old/` - Legacy example for comparison with xgboost 3.1.0
- `job.py` - Script to build NVFlare + Flower jobs
- `requirements.txt` - Python dependencies for setup

## Quick Start

### 1. Setup Virtual Environment

```bash
# Create and activate virtual environment using uv
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Prepare NVFlare POC Environment

```bash
# Prepare POC environment with 3 clients
nvflare poc prepare -n 3

# Start the POC environment
nvflare poc start
```

### 3. Build and Submit Job

**For the example with the old xgboost version (fs26-demo-old):**

```bash
# Build the job
python job.py --flower-dir fs26-demo-old

# Submit the job
nvflare job submit -j ./nvflare_workspace/jobs/fs26-demo-old
```

**For the example with the latest xgboost version (fs26-demo):**

```bash
# Build the job
python job.py --flower-dir fs26-demo

# Submit the job
nvflare job submit -j ./nvflare_workspace/jobs/fs26-demo
```

### 4. Monitor Job Execution

```bash
# Monitor job status
nvflare job monitor <JOB_ID>

# View job logs
nvflare job logs <JOB_ID>
```


## Accessing Runtime Environment

To access the running job environment:

```bash
cd /tmp/nvflare/poc/example_project/prod_00
```

You will see the version information outputted to the job folder on the clients to a file called `MY_LOG.TXT`. Example path:
```bash
/tmp/nvflare/poc/example_project/prod_00/site-1/e4439ed4-4aa1-4796-a451-6e9d9b6ea04c/MY_LOG.TXT
```

