#!/usr/bin/env bash
# One-command wrapper around the workshop 8 Docker image (macOS / Linux).
#
#   ./ws8.sh                 build the image if needed, then run everything
#   ./ws8.sh predict --site L
#   ./ws8.sh ui              inference UI for the global model: http://localhost:8080
#   ./ws8.sh deploy          Deployment Engine: SuperLink + 2 SuperNodes in containers
#   ./ws8.sh rebuild         rebuild the image (after changing the code)
#   ./ws8.sh help            all commands
#
# Settings: RUN_CONFIG="dataset='datasetA' event='NIV'" ./ws8.sh
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=hereditary-ws8

die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

command -v docker >/dev/null || die "docker is not installed: https://docs.docker.com/get-docker/"
docker info >/dev/null 2>&1 || die "the Docker daemon is not running: start Docker Desktop (or 'colima start') and try again."

mkdir -p outputs   # create it as yourself, so the container can write to it
[[ -d data ]] || die "no ./data folder. Download the iDPP data (see README.md) and extract it so that ./data/ALS_federated/T/... exists."

build() { docker build -t "$IMAGE" .; }
if [[ "${1:-}" == "rebuild" ]]; then build; exit 0; fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || build

export UID GID="$(id -g)"   # docker compose reads these for "user:"
if [[ "${1:-}" == "deploy" ]]; then
    trap 'docker compose --profile deploy down >/dev/null 2>&1' EXIT
    docker compose --profile deploy up --abort-on-container-exit --exit-code-from deploy-run
    exit $?
fi

tty=(); [[ -t 0 && -t 1 ]] && tty=(-it)
ports=()
if [[ "${1:-}" == "ui" ]]; then
    PORT="${PORT:-8080}"
    if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
        die "port $PORT is already in use: run  PORT=8081 ./ws8.sh ui"
    fi
    ports=(-p "127.0.0.1:$PORT:8080" -e "UI_PORT=$PORT")   # this machine only: the UI shows patient data
fi
exec docker run --rm ${tty[@]+"${tty[@]}"} ${ports[@]+"${ports[@]}"} --shm-size=2g \
    --user "$(id -u):$(id -g)" \
    -e RUN_CONFIG="${RUN_CONFIG:-}" \
    -v "$PWD/data:/data:ro" \
    -v "$PWD/outputs:/outputs" \
    "$IMAGE" "$@"
