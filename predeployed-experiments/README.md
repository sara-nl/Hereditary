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

# Copy the custom folder to the server to pre-deploy it and adjust default POC permissions
./prepare_poc.sh

# Start the POC environment
nvflare poc start
```

### 3. Build jobs

```bash
# Build job with BYOC permissions
python3 job_byoc.py --flower-dir fs26-demo

# Build job without BYOC permissions
python3 job_no_byoc.py --flower-app-path local/custom/fs26-demo --job-name fs26-demo-no-byoc
```

### 4. Submit jobs

```bash
# Submit job with BYOC permissions
nvflare job submit -j ./nvflare_workspace/jobs/fs26-demo

# Submit job without BYOC permissions, where the code is already on the server
nvflare job submit -j ./nvflare_workspace/jobs/fs26-demo-no-byoc
```
