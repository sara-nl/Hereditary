#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_ROOT="${ROOT}/nvflare_workspace/jobs"
SIM_WORKSPACE="${ROOT}/nvflare_workspace/sim_workspace"

mkdir -p "${JOB_ROOT}" "${SIM_WORKSPACE}"

echo "Building NVFlare job (Flower payload staged under staging/) ..."
uv run python "${ROOT}/job.py" --export-dir "${JOB_ROOT}"

echo "Running NVFlare simulator: ${JOB_ROOT}/cifar10_flower (3 clients) ..."
uv run nvflare simulator "${JOB_ROOT}/cifar10_flower" -w "${SIM_WORKSPACE}" -n 3 -t 3 -l concise
