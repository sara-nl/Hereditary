# Running workshop 8 with Docker

The image contains Python 3.11, every dependency pinned to an exact version and hash (`docker/requirements.lock`), and the code. It does **not** contain the iDPP data: you mount that at run time, so the data never ends up in an image.

You need Docker (Docker Desktop, or colima on macOS) with at least **4 GB of memory**, and the data extracted so that `./data/ALS_federated/T/...` exists (see [README.md](README.md)).

## Quick start (macOS / Linux)

```bash
./ws8.sh                     # builds the image the first time, then runs everything
```

This runs the centralised baseline, the federated Flower run, the comparison and one prediction, and stops at the first failure. Results go to `./outputs`. The run ends with:

```
VERDICT: identical up to rounding - the federated model IS the pooled model.
```

## Inference UI: apply the global model to a patient

```bash
./ws8.sh ui                  # then open http://localhost:8000   (PORT=8001 ./ws8.sh ui if 8000 is taken)
```

Pick a site, split and patient, or edit any schema value, and the page shows:
- P(event) at 12, 24 and 36 months, the predicted median and the hazard ratio, next to the observed outcome;
- the survival curve of the patient (federated model), the same patient through the centralised model (it overlaps), an average patient and the Kaplan-Meier curve of the training population. Hover for H₀(t) and S(t) at any month;
- **how the prediction is computed**, step by step: imputation and z-scoring with the federated means and SDs, each feature's z × β contribution, the risk score x·β, the hazard ratio, and S(t) = exp(−H₀(t)·exp(x·β)) at the horizons;
- checks that the step-by-step numbers equal `features.encode()` and `cox.survival()`, and that the centralised model gives the same risk score;
- where the model comes from: training sites, test C-index per site, Newton iterations and the columns the federated preprocessing dropped.

Every model in `outputs/` can be selected (for example `A_NIV/model.json` from a `RUN_CONFIG` run). Train first (`./ws8.sh`). The UI shows patient-level data, so the port is published on `127.0.0.1` only. Without Docker, run `python app.py` in `fed-survival-flower/`.

More commands:

```bash
./ws8.sh predict --site L                       # one patient of site L
./ws8.sh predict --site U --split train         # centre U, which never took part in training
./ws8.sh centralised | federated | compare      # one step
./ws8.sh ui                                     # inference UI on http://localhost:8000
./ws8.sh deploy                                 # Deployment Engine: SuperLink + 2 SuperNodes in containers
./ws8.sh check                                  # only check the data and the config
./ws8.sh bash                                   # a shell inside the container
./ws8.sh rebuild                                # after you change the code
./ws8.sh help
```

## Changing settings

Use `RUN_CONFIG` instead of editing `pyproject.toml`. It changes the setting for the Flower run **and** for `centralised.py`, `compare.py` and `predict.py`, so they always agree:

```bash
RUN_CONFIG="dataset='datasetA' event='NIV' output-dir='outputs/A_NIV'" ./ws8.sh
RUN_CONFIG="sites='T,L,U'" ./ws8.sh federated       # U as a third site (SuperNodes are added for you)
RUN_CONFIG="min-coverage=0.4" ./ws8.sh              # keeps C9orf72
```

Strings need quotes (`dataset='datasetA'`), numbers do not. A misspelt key or an unquoted string is an error, never a silent default.

## Without the wrapper (Windows, or plain docker)

```bash
docker build -t hereditary-ws8 .
docker run --rm --shm-size=2g -v "$PWD/data:/data:ro" -v "$PWD/outputs:/outputs" hereditary-ws8
docker run --rm --shm-size=2g -v "$PWD/data:/data:ro" -v "$PWD/outputs:/outputs" \
    -e RUN_CONFIG="dataset='datasetA' event='NIV'" hereditary-ws8 federated
```

In PowerShell, write `${PWD}` instead of `$PWD`. On Linux, add `--user "$(id -u):$(id -g)"` so the results in `./outputs` belong to you.

With Docker Compose:

```bash
docker compose run --rm ws8                       # everything, in simulation
docker compose run --rm ws8 predict --site L
docker compose --profile sim up ui                # inference UI on http://localhost:8000
docker compose --profile deploy up --build --abort-on-container-exit --exit-code-from deploy-run
docker compose --profile deploy down
```

## Safety nets built in

- **Data check before anything runs.** The entrypoint finds `ALS_federated/` whether you mount `data/`, `data/ALS_federated` itself or a folder one level up, and lists any missing CSV file for the chosen sites and dataset.
- **`flwr run` failures are caught.** `flwr run` exits with 0 even when the run fails; the container checks that a new `model.json` was written and exits with 1 otherwise.
- **Consistent settings.** `RUN_CONFIG` is validated and applied to the Flower run and to the scripts. Each container starts from the original `pyproject.toml`, so an earlier override never carries over.
- **Simulation sites.** The number of virtual SuperNodes follows `sites` (passed as `--federation-config "num-supernodes=N"`), and site sets that the simulation cannot map (partition-id 0, 1, 2 = T, L, U) are rejected up front.
- **Permissions.** The container runs as a non-root user and works under any `--user` uid. If `./outputs` is not writable, the error says how to fix it.
- **Reproducible install.** Every dependency is pinned with hashes, and `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` stops Flower from installing them again for every run, so runs work offline. To update them, run `uv pip compile fed-survival-flower/pyproject.toml --python-version 3.11 --python-platform linux --generate-hashes -o docker/requirements.lock` and rebuild.

## Troubleshooting

| symptom | fix |
|---|---|
| `the Docker daemon is not running` | start Docker Desktop, or `colima start` |
| `no data in /data` | extract the data into `./data` so that `./data/ALS_federated/T/` exists |
| `cannot write to /outputs` | Linux: use `./ws8.sh`, or add `--user "$(id -u):$(id -g)"` |
| Ray warns that the object store uses `/tmp` instead of `/dev/shm` | harmless for this data size; add `--shm-size=2g` (or more) to silence it |
| Ray is killed, or the run hangs | give Docker at least 4 GB of memory |
| UI: `port 8000 is already in use` | `PORT=8001 ./ws8.sh ui` |
| UI: `No model in outputs` | train first: `./ws8.sh` |
| the image does not have your code changes | `./ws8.sh rebuild` |
| `/var/folders/...` or `/tmp/...` mounts are empty (colima) | colima shares only your home folder: keep the project under `~` |

The workshop SuperLink (`surf-superlink`) is not wired into the image, because its address comes from the group page. Since Flower 1.43 SuperLink connections live in `~/.flwr/config.toml`, not in `pyproject.toml`; in a container shell (`./ws8.sh bash`):

```bash
mkdir -p ~/.flwr && printf '[superlink]\ndefault = "local"\n\n[superlink.local]\naddress = ":local:"\n\n[superlink.surf-superlink]\naddress = "SURF_SUPERLINK_IP:9093"\ninsecure = true\n' > ~/.flwr/config.toml
flwr run . surf-superlink --stream
```
