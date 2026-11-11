#!/usr/bin/env bash
# Entrypoint of the workshop 8 image. Run with "help" for the commands.
set -euo pipefail

APP=/app/fed-survival-flower
DATA_MOUNT=/data
OUT=/outputs

say()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
hereditary-ws8: federated Cox survival model for ALS (Flower)

  docker run --rm --shm-size=2g -v "$PWD/data:/data:ro" -v "$PWD/outputs:/outputs" hereditary-ws8 [COMMAND]

Commands (default: all)
  all            centralised -> federated -> compare -> predict, stops at the first failure
  centralised    pooled baseline (writes outputs/centralised_model.json)
  federated      flwr run . (Simulation Engine; writes outputs/preprocessing.json, model.json)
                 extra args go to flwr run, e.g.  federated --run-config "l2=0.1"
  compare        federated vs centralised
  predict [ARGS] survival curve of one patient, e.g.  predict --site L --split test
  ui             inference UI for the global model on port 8000 (publish with -p 127.0.0.1:8000:8000)
  check          only check the data and the config
  superlink      Deployment Engine: start a SuperLink (port 9093)
  supernode SITE [SUPERLINK_HOST]
                 Deployment Engine: start a SuperNode for SITE (T, L or U)
  deploy-run [SUPERLINK_HOST]
                 Deployment Engine: start the run on a SuperLink and stream its output
  python ... | bash | flwr ...   run any command in the project folder

Settings (environment variables)
  RUN_CONFIG  overrides for [tool.flwr.app.config], applied to the Flower run AND the
              scripts, e.g.  -e RUN_CONFIG="dataset='datasetA' event='NIV'"

Mounts
  /data       the iDPP data: the folder that contains ALS_federated/ (or ALS_federated itself)
  /outputs    results (models, preprocessing artifact, plots)
EOF
}

# ---- 1. Config: start from the pristine pyproject.toml, then apply RUN_CONFIG -----------
cp /app/pyproject.orig.toml "$APP/pyproject.toml"
if [[ -n "${RUN_CONFIG:-}" ]]; then
    python /usr/local/bin/set_config.py "$APP/pyproject.toml" "$RUN_CONFIG" || die "invalid RUN_CONFIG"
fi
cfg() { python -c "from fed_survival import read_config; print(read_config()['$1'])"; }

# ---- 2. Data: find ALS_federated/ wherever it is under /data -----------------------------
find_data() {
    local c
    for c in "$DATA_MOUNT/ALS_federated" "$DATA_MOUNT" "$DATA_MOUNT"/*/ALS_federated "$DATA_MOUNT"/*; do
        if [[ -d "$c/T" && -d "$c/L" ]]; then echo "$c"; return 0; fi
    done
    return 1
}

check_data() {
    local root sites dataset s split kind f missing=()
    if [[ -z "$(ls -A "$DATA_MOUNT" 2>/dev/null)" ]]; then
        die "no data in $DATA_MOUNT.
Download the iDPP data from https://hereditary.dei.unipd.it (Files -> Hereditary/Data/FederatedLearning),
extract it so that you have ./data/ALS_federated/T/..., and mount it:
    docker run ... -v \"\$PWD/data:/data:ro\" ...
(on Windows PowerShell: -v \"\${PWD}/data:/data:ro\")"
    fi
    root=$(find_data) || die "could not find ALS_federated/ (with T/ and L/ inside) in $DATA_MOUNT.
What is there:
$(cd "$DATA_MOUNT" && find . -maxdepth 3 -type d | head -20)"
    mkdir -p /app/data
    ln -sfn "$root" /app/data/ALS_federated
    dataset=$(cfg dataset)
    sites=$(cfg sites)
    for s in ${sites//,/ }; do
        for split in train test; do
            for kind in static-vars visits outcome; do
                f="/app/data/ALS_federated/$s/$dataset/$split/${dataset}_${split}-$kind.csv"
                [[ -r "$f" ]] || missing+=("${f#/app/data/ALS_federated/}")
            done
        done
    done
    if (( ${#missing[@]} )); then
        die "missing data files (dataset $dataset, sites $sites) under $root:
$(printf '  %s\n' "${missing[@]}")"
    fi
    echo "data: ${root} (dataset $dataset, sites $sites)"
}

check_outputs() {
    local dir
    dir=$(cfg output-dir)
    mkdir -p "$dir" 2>/dev/null || true
    if ! touch "$dir/.write-test" 2>/dev/null; then
        die "cannot write to $OUT (uid $(id -u)).
On Linux, run the container as yourself:  docker run --user \"\$(id -u):\$(id -g)\" ...
or make the folder writable:              chmod a+rwx outputs"
    fi
    rm -f "$dir/.write-test"
}

# ---- 3. Steps --------------------------------------------------------------------------
centralised() { say "Centralised baseline"; python centralised.py; }

federated() {
    # flwr run exits 0 even when the run fails: check that model.json was (re)written
    local dir stamp log
    say "Federated run (Flower Simulation Engine)"
    # Simulation maps partition-id 0, 1, 2 to sites T, L, U: start one SuperNode per site
    local sites n
    sites=$(cfg sites)
    n=$(python - "$sites" <<'EOF'
import sys
sites = {s.strip() for s in sys.argv[1].split(",")}
order = ["T", "L", "U"]
n = len(sites)
if sites != set(order[:n]):
    sys.exit(f"in simulation, sites must be T,L or T,L,U (partition-id 0, 1, 2 = T, L, U), not {sys.argv[1]!r}")
print(n)
EOF
) || die "invalid sites for the Simulation Engine"
    dir=$(cfg output-dir)
    stamp=$(mktemp); log=$(mktemp)
    sleep 1   # model.json must be strictly newer than the stamp
    flwr run . --stream --federation-config "num-supernodes=$n" "$@" 2>&1 | tee "$log"
    if [[ ! "$dir/model.json" -nt "$stamp" ]] || grep -qE "SiteError|Traceback \(most recent" "$log"; then
        die "the federated run failed (no new $dir/model.json): see the output above.
If Ray complains about memory or /dev/shm, add --shm-size=2g and give Docker at least 4 GB RAM."
    fi
    rm -f "$stamp" "$log"
}

compare() { say "Federated vs centralised"; python compare.py; }

predict() { say "Prediction for one held-out patient"; python predict.py "$@"; }

need_models() {
    local dir; dir=$(cfg output-dir)
    for m in "$@"; do
        [[ -f "$dir/$m" ]] || die "$dir/$m does not exist yet: run 'all' first (or 'federated' / 'centralised')."
    done
}

# ---- 4. Dispatch -----------------------------------------------------------------------
cd "$APP"
cmd="${1:-all}"; shift || true

case "$cmd" in
    help|-h|--help) usage ;;
    check)       check_data; check_outputs; echo "OK: data and outputs are ready." ;;
    all)         check_data; check_outputs; centralised; federated "$@"; compare; predict
                 say "Done. Results are in the outputs folder you mounted at $OUT." ;;
    centralised) check_data; check_outputs; centralised ;;
    federated)   check_data; check_outputs; federated "$@" ;;
    compare)     need_models model.json centralised_model.json; compare ;;
    predict)     check_data; check_outputs; need_models model.json; predict "$@" ;;
    ui)
        need_models model.json
        (check_data) || echo "No usable site data: the UI works with manual input only."
        say "Inference UI: open http://localhost:${UI_PORT:-8000} in your browser (Ctrl+C to stop)"
        exec python app.py --host 0.0.0.0 --port 8000 ;;
    superlink)
        check_outputs
        exec flower-superlink --insecure --host 0.0.0.0 --port 9093 ;;
    supernode)
        site="${1:?usage: supernode SITE [SUPERLINK_HOST]}"; host="${2:-superlink}"
        check_data
        exec flower-supernode --insecure --superlink "$host:9093" \
            --node-config "site='$site' data-path='/app/data/ALS_federated'" ;;
    deploy-run)
        host="${1:-superlink}"
        # Flower 1.43 reads SuperLink connections from ~/.flwr/config.toml, not from pyproject.toml
        mkdir -p "$HOME/.flwr"
        printf '[superlink]\ndefault = "local"\n\n[superlink.local]\naddress = ":local:"\n\n[superlink.local-superlink]\naddress = "%s:9093"\ninsecure = true\n' \
            "$host" > "$HOME/.flwr/config.toml"
        for i in $(seq 60); do
            (exec 3<>"/dev/tcp/$host/9093") 2>/dev/null && break
            [[ $i == 60 ]] && die "SuperLink $host:9093 is not reachable"
            sleep 1
        done
        exec flwr run . local-superlink --stream ;;
    python|python3|bash|sh|flwr|flower-*) exec "$cmd" "$@" ;;
    *) usage; die "unknown command '$cmd'" ;;
esac
