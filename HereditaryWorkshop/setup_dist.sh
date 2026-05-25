#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVISION_W="${ROOT}/nvflare_workspace/provision"
PROD_00="${PROVISION_W}/hereditary_workshop/prod_00"
JOB_EXPORT="${ROOT}/nvflare_workspace/jobs"

mkdir -p "${PROVISION_W}" "${JOB_EXPORT}"

# Clean up existing provision to ensure it always generates prod_00
rm -rf "${PROVISION_W}/hereditary_workshop"

echo "Provisioning NVFlare project (project.yml -> ${PROD_00}) ..."
uv run nvflare provision -w "${PROVISION_W}" -p "${ROOT}/project.yml"

echo "Injecting custom security components into node workspaces..."
for node in "${PROD_00}"/*; do
    if [ -d "${node}" ]; then
        node_name=$(basename "${node}")
        if [[ ! "${node_name}" == *"@"* ]]; then
            mkdir -p "${node}/local"
            cp -r "${ROOT}/custom4client/"* "${node}/local/"
        fi
    fi
done

echo "Building Flower NVFlare job -> ${JOB_EXPORT} ..."
echo "  - Packaging cifar10_flower..."
uv run python "${ROOT}/job.py" --export-dir "${JOB_EXPORT}"

export NVFLARE_PROVISION_ROOT="${PROD_00}"
echo ""
echo "Done."
echo "  Provisioned startup kits: ${PROD_00}"
echo "  Exported job folders:     ${JOB_EXPORT}/"
echo "  Docker Compose expects:   NVFLARE_PROVISION_ROOT=${PROD_00}"
echo ""
echo "Next:"
echo "  - Quick local sim:    bash run_local.sh"
echo "  - Docker services:    docker compose up --build"
echo "  - Then submit a job via admin@${PROD_00} (see NVFlare deployment docs)."
echo "    Example: uv run nvflare job submit -j ${JOB_EXPORT}/cifar10_flower"
