#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults — must match defaults in build_distribution.sh
PROJECT_NAME="hereditary_workshop"
ADMIN_EMAIL="surf@hereditary.com"

# Parse only the flags we need here; all flags are forwarded to build_distribution.sh too.
ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    arg="${ARGS[$i]}"
    next_i=$((i + 1))
    case "$arg" in
        --name)  PROJECT_NAME="${ARGS[$next_i]}"; i=$((i + 2)) ;;
        --admin) ADMIN_EMAIL="${ARGS[$next_i]}";  i=$((i + 2)) ;;
        --sites) # skip all site names (they don't start with --)
            i=$((i + 1))
            while [[ $i -lt ${#ARGS[@]} ]] && [[ ! "${ARGS[$i]}" == --* ]]; do
                i=$((i + 1))
            done
            ;;
        *) i=$((i + 1)) ;;
    esac
done

PROD_00="${ROOT}/nvflare_workspace/provision/${PROJECT_NAME}/prod_00"
JOB_EXPORT="${ROOT}/nvflare_workspace/jobs/cifar10_flower"

# Forward all arguments to build_distribution.sh (project.yml, certs, job, docker-compose.yml, dist/)
# Tear down any existing containers and networks from a previous run FIRST,
# before build_distribution.sh overwrites docker-compose.yml with new names.
if [[ -f "${ROOT}/docker-compose.yml" ]]; then
    echo "Stopping any existing containers from a previous run..."
    docker compose down --remove-orphans 2>/dev/null || true
fi

echo "Building configuration, provisioning, and generating Docker Compose..."
bash "${ROOT}/build_distribution.sh" "$@"

echo ""
echo "Starting Docker Compose services..."
docker compose up --build -d

echo ""
echo "Waiting for NVFlare server to be ready..."
sleep 15

echo "Configuring NVFlare workspace for admin..."
uv run nvflare config -d "${PROD_00}/${ADMIN_EMAIL}/startup"

echo "Submitting job using NVFlare Admin CLI..."
uv run nvflare job submit -j "${JOB_EXPORT}"

echo ""
echo "Job submitted! Tailing logs (Press Ctrl+C to exit)..."
docker compose logs -f
