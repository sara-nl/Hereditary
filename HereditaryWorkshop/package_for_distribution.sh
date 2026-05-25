#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_00="${ROOT}/nvflare_workspace/provision/hereditary_workshop/prod_00"
DIST_DIR="${ROOT}/dist"

if [ ! -d "${PROD_00}" ]; then
    echo "Error: Startup kits not found. Please run ./setup_dist.sh first!"
    exit 1
fi

echo "Cleaning up old distribution packages..."
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"

echo "Zipping Startup Kits..."
for node in "${PROD_00}"/*; do
    if [ -d "${node}" ]; then
        node_name=$(basename "${node}")
        echo "  Packaging ${node_name}..."
        # Zip quietly (-q) and recursively (-r)
        (cd "${PROD_00}" && zip -qr "${DIST_DIR}/${node_name}.zip" "${node_name}")
    fi
done

echo "Copying Docker Environment..."
cp "${ROOT}/Dockerfile.nvflare" "${DIST_DIR}/"
cp "${ROOT}/pyproject.toml" "${DIST_DIR}/"
cp "${ROOT}/uv.lock" "${DIST_DIR}/"

echo ""
echo "Done! All files ready for distribution are in the 'dist' folder:"
ls -lh "${DIST_DIR}"
