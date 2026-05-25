#!/usr/bin/env bash
set -euo pipefail

# Default values
PROJECT_NAME="hereditary_workshop"
SERVER_IP="server1"
FL_PORT=8002
ADMIN_PORT=8003
ADMIN_EMAIL="surf@hereditary.com"
ORG="nvidia"
OUTPUT_FILE="project.yml"

# Default sites array
SITES=("site-1" "site-2" "site-3")

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --name) PROJECT_NAME="$2"; shift ;;
        --server) SERVER_IP="$2"; shift ;;
        --fl-port) FL_PORT="$2"; shift ;;
        --admin-port) ADMIN_PORT="$2"; shift ;;
        --admin) ADMIN_EMAIL="$2"; shift ;;
        --org) ORG="$2"; shift ;;
        --output) OUTPUT_FILE="$2"; shift ;;
        --sites) 
            shift
            SITES=()
            while [[ "$#" -gt 0 ]] && [[ ! "$1" == --* ]]; do
                SITES+=("$1")
                shift
            done
            continue
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Generating ${OUTPUT_FILE}..."

# Write the top part of the YAML
cat > "${OUTPUT_FILE}" <<EOF
api_version: 3
name: ${PROJECT_NAME}
description: ${PROJECT_NAME} — Flower + NVIDIA FLARE

participants:
  - name: ${SERVER_IP}
    type: server
    org: ${ORG}
    fed_learn_port: ${FL_PORT}
    admin_port: ${ADMIN_PORT}
EOF

# Loop through and append all the client sites
for site in "${SITES[@]}"; do
cat >> "${OUTPUT_FILE}" <<EOF
  - name: ${site}
    type: client
    org: ${ORG}
EOF
done

# Write the bottom part of the YAML
cat >> "${OUTPUT_FILE}" <<EOF
  - name: ${ADMIN_EMAIL}
    type: admin
    org: ${ORG}
    role: project_admin

builders:
  - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
    args:
      template_file:
        - master_template.yml
        - aws_template.yml
        - azure_template.yml
  - path: nvflare.lighter.impl.static_file.StaticFileBuilder
    args:
      config_folder: config
      overseer_agent:
        path: nvflare.ha.dummy_overseer_agent.DummyOverseerAgent
        overseer_exists: false
        args:
          sp_end_point: ${SERVER_IP}:${FL_PORT}:${ADMIN_PORT}
  - path: nvflare.lighter.impl.cert.CertBuilder
  - path: nvflare.lighter.impl.signature.SignatureBuilder
EOF

echo "Successfully generated ${OUTPUT_FILE}"
echo "Server endpoint configured as: ${SERVER_IP}:${FL_PORT}:${ADMIN_PORT}"
echo "Configured ${#SITES[@]} client sites: ${SITES[*]}"
