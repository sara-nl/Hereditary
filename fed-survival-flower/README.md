# fed-survival-flower: Federated Cox survival model for ALS with Flower

This project trains an ALS survival model (time to death) across two sites with the Flower messaging API: **T** (Turin) and **L** (Lisbon, standing in for the Answer ALS site). Patient-level data never leaves a site. The federated model is the model you would get by pooling the data, and it is then used to predict the survival curve of a held-out patient.

The model is a Cox proportional hazards model trained with WebDISCO:
- **WebDISCO: a web service for distributed Cox model learning without patient-level data sharing**, Lu et al., *JAMIA* 2015. [Link to paper](https://doi.org/10.1093/jamia/ocv083)

Headline results (iDPP dataset C, event DEATH, T and L training splits):

| | federated (Flower) | centralised (pooled, scikit-survival) |
|---|---|---|
| training patients / events | T 1199, L 503 / 1474 deaths | 1702 / 1474 deaths |
| Newton iterations | 6 | - |
| max \|difference in coefficients\| | 4.3e-14 | |
| test C-index T (n=362) | 0.7024 | 0.7024 |
| test C-index L (n=121) | 0.7515 | 0.7515 |

The published iDPP results are in the range 0.70-0.75. Each site evaluates the model on its own test split, so there is no "global" C-index.

The code is split into three stages. Only one small JSON file, the *preprocessing artifact*, is passed from stage 1 to stage 2:
1. **Federated preprocessing**, a federated analytics step. It decides which columns to keep and computes the category levels, means and standard deviations. In the workshop exercise you replace this stage with the HEREDITARY federated analytics network.
2. **Federated Cox training** (WebDISCO) on a public monthly time grid.
3. **Evaluation**: each site computes the C-index on its own test split.

The input schema and the choice of model are described in [SCHEMA.md](../SCHEMA.md).


## Install

You need Python 3.11 and the iDPP data (see the [workshop README](../README.md)). Put the data in a `data/` folder in the root of this repository, so that you have `data/ALS_federated/T/...` (the folder is gitignored), or change `data-path` in `pyproject.toml`.

```bash
cd fed-survival-flower
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

If you change the code, `uv pip install -e ".[dev]"` adds black and isort; format with `isort . && black .` (the settings are in `pyproject.toml`).


## Running the experiment

Run all commands from this folder. The scripts read the same `[tool.flwr.app.config]` from `pyproject.toml` as the Flower app does. `--run-config` changes only the Flower run, so edit `pyproject.toml` when you want the scripts to follow a new setting as well.

### 1. The centralised baseline

This is the target the federated run has to reproduce. It pools the training data of T and L, which is possible here only because this is a workshop.

```bash
python centralised.py        # writes outputs/centralised_model.json
```

```
Features (11): age_onset, sex=Male, onset_bulbar, diagnostic_delay, time_since_onset, bmi, alsfrs_r_first, alsfrs_r_last, alsfrs_r_slope, fvc_last, fvc_n
Dropped C9orf72: observed for 44% at L (needs 50% at every site)
Pooled training data: 1702 patients, 1474 DEATH events
...
Test C-index: L 0.751 (n=121), T 0.702 (n=362)
```

### 2. The federated run (Simulation Engine)

```bash
flwr run . --stream          # writes outputs/preprocessing.json and outputs/model.json
```

The simulation runs two virtual SuperNodes (the Flower default): partition-id 0 is site T and partition-id 1 is site L. The run takes about 35 s. Without `--stream`, `flwr run` only starts the run and returns; then `flwr ls` lists the runs and `flwr log <run-id>` shows the output. The server output tells the story (lines prefixed `(ClientAppActor pid=...)` come from the sites):

```
=== fed-survival: datasetC, event DEATH, sites L,T ===

--- Stage 1: federated preprocessing (query.summary) ---
Training patients: L 503, T 1199
Preprocessing: .../fed-survival-flower/outputs/preprocessing.json
Features (11): age_onset, sex=Male, onset_bulbar, diagnostic_delay, time_since_onset, bmi, alsfrs_r_first, alsfrs_r_last, alsfrs_r_slope, fvc_last, fvc_n
Dropped C9orf72: observed for 44% at L (needs 50% at every site)

--- Stage 2: federated Cox training (train.risk_sets, train.hessian) ---
Public time grid: 600 months; 1474 events in 108 of them
iteration    log-likelihood  max |step|
        1      -9697.341130    4.51e-01
        2      -9424.399765    1.49e-01
        3      -9384.744979    2.37e-02
        4      -9384.203770    5.63e-04
        5      -9384.203570    2.47e-07
        6      -9384.203570    4.16e-14

feature                    beta   hazard ratio per SD
age_onset                0.2356                 1.266
...
alsfrs_r_last           -0.2776                 0.758
alsfrs_r_slope          -0.2297                 0.795
...

--- Stage 3: evaluation on each site's test split (evaluate.cindex) ---
Test C-index L: 0.7515 (n=121)
Test C-index T: 0.7024 (n=362)
```

`C9orf72` is in the schema on purpose. Lisbon has it for only 44% of its patients, so the federated coverage rule drops it. This shows why preprocessing has to be a federated decision: no single site could have made it alone.

The first `flwr run` starts a local SuperLink in the background (port 39091, logs in `~/.flwr/local-superlink/`) and later runs reuse it. It keeps using the venv it was started from, so if you switch venvs or Flower versions, stop it first: `pkill -f "flower-superlink --insecure --simulation"`.

### 3. Compare centralised and federated model

```bash
python compare.py
```

```
max |delta beta|               4.3e-14
max |delta cumulative hazard|  9.6e-13   (H0 ranges 0 to 7.83)

test C-index    federated  centralised
L                  0.7515       0.7515
T                  0.7024       0.7024

VERDICT: identical up to rounding - the federated model IS the pooled model.
```

### 4. Live demo: predict a held-out patient

A site applies the trained model to one of its own patients. The script prints the patient's schema values, P(death) at 12, 24 and 36 months, the predicted median survival and the observed outcome. It also saves a plot to `outputs/prediction_<site>_<id>.png` (`--model outputs/centralised_model.json` uses the centralised model instead) with the federated curve, the centralised curve (which should overlap it) and the federated Kaplan-Meier curve of the training population.

```bash
python predict.py                           # first Turin test patient
python predict.py --site U --split train    # centre U, which never took part in training
python predict.py --site L --patient 0xca23e24556475bbc742f37df9b4dacfc   # any patient
```

```
Patient 0xca2a048cc828f5ebd2000a554301d13f (site T, test split)
  ...
  C9orf72               normal   not used: dropped by the federated preprocessing
  fvc_last                   -   missing: imputed with the federated mean
  ...
Risk score x.beta = -0.638: hazard 0.53 x that of a patient with average values
  P(DEATH by 12 months) = 0.08
  P(DEATH by 24 months) = 0.31
  P(DEATH by 36 months) = 0.48
  Predicted median time to DEATH: 38.0 months
  Observed: DEATH at 24.0 months
```

For the patient from centre U (`0x11574b16...`) the model gives P(death) = 0.08, 0.33 and 0.51 at 12, 24 and 36 months and a predicted median of 35.0 months. The observed outcome is death at 39.4 months. The model works for a centre that had no part in training because U builds the same schema table and encodes it with the same artifact.


### 5. Inference UI: the global model, step by step

```bash
python app.py                                # http://127.0.0.1:8000
```

A local web page that applies any model in `outputs/` to a patient you pick (or values you type) and shows the whole calculation: imputation and z-scoring with the federated artifact, the contribution of every feature to x·β, the hazard ratio, the Breslow baseline hazard and the survival curve, with checks against `encode()`, `survival()` and the centralised model. It only binds to 127.0.0.1, because it shows patient-level data. In Docker: `./ws8.sh ui` (see [DOCKER.md](../DOCKER.md)).


## Running with the Deployment Engine
In a deployment every site runs its own SuperNode and says in `--node-config` which site it is and where its data is (give `data-path` as an absolute path). In simulation the mapping is {0: "T", 1: "L", 2: "U"}.

Open four terminals. In each one: `cd fed-survival-flower`, activate the venv and, on a shared node, `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. Then:

`flwr run` finds a SuperLink by a connection name in `~/.flwr/config.toml` (Flower 1.43 no longer reads them from `pyproject.toml`). Add these once to the end of that file; it is created by the first `flwr run`:

```toml
[superlink.local-superlink]
address = "127.0.0.1:9093"
insecure = true

[superlink.surf-superlink]
address = "SURF_SUPERLINK_IP:9093"   # see the group page
insecure = true
```

`flwr config list` shows the connections. Then:

```bash
# 1. The server (SuperLink). Start it in this folder: the ServerApp writes outputs/ relative to it
flower-superlink --insecure --port 9093                    # add --host 0.0.0.0 for remote SuperNodes

# 2. One SuperNode per site
flower-supernode --insecure --superlink SUPERLINK_IP:9093 --port 9094 \
    --node-config "site='T' data-path='/abs/path/to/ALS_federated'"
flower-supernode --insecure --superlink SUPERLINK_IP:9093 --port 9095 \
    --node-config "site='L' data-path='/abs/path/to/ALS_federated'"

# 3. Start the run
flwr run . local-superlink --stream
```

- On one machine, `SUPERLINK_IP` is `127.0.0.1`. Since Flower 1.43 the SuperLink serves the SuperNodes and `flwr run` over one HTTP port (by default 8000 on 127.0.0.1; here 9093, because `app.py` uses 8000). A remote SuperLink needs `--host 0.0.0.0` and that port open for incoming TCP connections. Each SuperNode on a machine needs its own `--port` (local only). If a port is already in use (check with `ss -ltnp` or `lsof -iTCP -sTCP:LISTEN`), choose another.
- The run takes about 90 s instead of 35 s, because every message starts a fresh ClientApp process at the site (15 rounds of about 6 s). The `[site T] ...` lines appear in the SuperNode terminals, not in the `flwr run` output.
- The result is the same `outputs/model.json` as in simulation, so `python compare.py` gives the same verdict.
- During the workshop the HEREDITARY server is the SuperLink: use the `surf-superlink` connection (`flwr run . surf-superlink --stream`). The IP for the HEREDITARY server can be found on the group page. The outputs are then written on the server.
- The SuperNodes must be exactly the sites in `sites`. The ServerApp waits up to 300 s for them, checks the site names they report, and stops with an error naming the problem if a site is missing or its `data-path` is wrong.
- Without `--stream`: `flwr ls local-superlink` lists the runs, `flwr log <run-id> local-superlink` shows the output and `flwr stop <run-id> local-superlink` stops a run.
- `--insecure` means no TLS and no SuperNode authentication, which is fine for a localhost demo only. For a real deployment see the Flower guides under Resources, and `generating_keys/` in workshop 3.


## How it works

Every site runs `features.load_site`. It builds the 12-column schema table from its own BRAINTEASER CSVs, using only data up to month 6 (the landmark window). This table never leaves the site.

**Stage 1: preprocessing as federated analytics** (`analytics.py`). Each site answers one query with counts, sums and sums of squares (`site_summary`). The server combines them with plain sums (`build_artifact`):
- A column is kept only if at least `min-coverage` of the patients have it at *every* site.
- Categorical columns become 0/1 dummies `col=level` against the first level.
- The means and population standard deviations come from the summed moments.

The result is written to `outputs/preprocessing.json`, the only thing stage 2 receives. Sites use it to impute missing values with the mean and to z-score, so every site encodes its data identically.

**Stage 2: WebDISCO Cox on a public monthly grid** (`cox.py`). A Cox model cannot be averaged, because each patient's risk set spans all sites. But the gradient and Hessian of the Breslow partial likelihood are sums over patients. So in each Newton iteration the sites return two kinds of sums, and the server takes the exact pooled Newton step:
- round A: event counts and risk-set sums per grid month;
- round B: one p x p matrix.

All sites use the same public monthly time grid (1, 2, ..., 600 months), so no site reveals its exact event times. Round A at beta = 0 also gives the at-risk and event counts: the Kaplan-Meier curve of the training population, free of charge. After convergence the server computes the Breslow baseline hazard from round A at the final beta. There is no Strategy and no FedAvg.

**Stage 3: evaluation**. Each site computes Harrell's C-index of the final model on its own test split, using the exact (not month-rounded) times.

| message type | server -> site | site -> server |
|---|---|---|
| `query.summary` (stage 1, once) | nothing | per column: number observed, sum, sum of squares, or counts per level (JSON) |
| `train.risk_sets` (round A) | artifact, `beta` (p) | `d` (K), `S0` (K), `S1` (K x p), `x_events` (p) |
| `train.hessian` (round B) | artifact, `beta` (p), `c` (K) | `M` (p x p) |
| `evaluate.cindex` (stage 3, once) | artifact, `beta` (p) | `c_index`, `n` |

Here K = 600 grid months and p = 11 features. For dataset C the run has 15 rounds: 1 summary, 7 round A, 6 round B and 1 C-index. Every reply also carries the site's name, and the server sums the replies in a fixed order (sorted by site name), so runs are reproducible.

### What leaves a site

No per-patient rows leave a site, but the aggregates are not anonymous by themselves. On the exact event-time grid, the differences between consecutive risk-set sums would reveal individual patients' covariates for 38% of Turin and 69% of Lisbon training patients. The public monthly grid reduces this to the patients who are alone in their grid month: 38 (T) and 30 (L), or 17 and 26 if the sites only sent sums for months with a death. At the tail of follow-up a site's risk set can be a handful of patients, and small level counts in stage 1 can also single out a patient. This code applies no minimum-count rules and no secure aggregation. It is fine for the public iDPP data but not for live clinical data. See [SCHEMA.md](../SCHEMA.md), "What leaves a site", for the analysis and the mitigations.


## Main exercise: replace stage 1 with the HEREDITARY federated analytics network

Training only needs `preprocessing.json`, so the Flower preprocessing stage can be replaced by any system that produces the same file.

**1. The queries.** Run these on the schema table of each site's *training* split, as built by `features.load_site`. Each site answers with one summary in exactly the `site_summary` format:

```json
{"n": 1199,
 "numeric": {"age_onset": {"n_observed": 1199, "sum": 76720.1, "sum_sq": 5054321.9}, "...": {}},
 "categorical": {"sex": {"Female": 574, "Male": 625}, "C9orf72": {"expansion": 12, "normal": 400}}}
```

(The numbers in this example are made up.)

| field | per site |
|---|---|
| `n` | the number of training patients |
| `numeric.<col>`, for `age_onset`, `onset_bulbar`, `diagnostic_delay`, `time_since_onset`, `bmi`, `alsfrs_r_first`, `alsfrs_r_last`, `alsfrs_r_slope`, `fvc_last`, `fvc_n` | `n_observed` (the number of non-missing values), `sum`, `sum_sq` |
| `categorical.<col>`, for `sex`, `C9orf72` | the count of each observed level (missing values are not counted) |

To see what each site's answer must look like, compute the reference answers locally (after a first run, so that `outputs/` exists):

```python
import json
from fed_survival import read_config
from fed_survival.analytics import site_summary
from fed_survival.features import load_site

c = read_config()
for s in ["T", "L"]:
    table, _, _ = load_site(c["data-path"], s, c["dataset"], "train", c["window"], c["event"])
    json.dump(site_summary(table), open(f"outputs/fa_summary_{s}.json", "w"), indent=1)
```

**2. Build the artifact.** Put the site summaries (from the FA network, or the reference answers above) through `build_artifact`, the same function the Flower server uses:

```python
import json
from fed_survival.analytics import build_artifact, save_artifact

summaries = {s: json.load(open(f"outputs/fa_summary_{s}.json")) for s in ["T", "L"]}
save_artifact(build_artifact(summaries, min_coverage=0.5), "outputs/fa_preprocessing.json")
```

You can also let the FA network write the artifact directly. The training stage needs:
- `"format": "fed-survival-preprocessing/1"`;
- `features`, the ordered feature names, with `col=level` for dummies;
- `mean` and `std`, lists aligned with `features`;
- optionally `dropped`, a dict of column to reason, which is only printed.

**3. Train on it** (stage 1 is then skipped):

```bash
flwr run . --stream --run-config "preprocessing='file' preprocessing-file='outputs/fa_preprocessing.json'"
```

**4. Check it.** If the FA network produced the same artifact, you get exactly the same model: the same features, the same coefficients and the same C-index. Compare your file with `outputs/preprocessing.json` from a Flower run, and run `python compare.py`.


## Further exercises

- **NMES (Naive Multiple Event Survival): one standard Cox model per event.** Here any other outcome counts as censored. Run dataset A with the event NIV:
  ```bash
  flwr run . --stream --run-config "dataset='datasetA' event='NIV' output-dir='outputs/A_NIV'"
  ```
  This gives 665 NIV events, 8 iterations and a test C-index of T 0.6717 and L 0.7257. Then run dataset A with `event='DEATH'` (death as the *first* event, with NIV counted as censoring: only 38 deaths in L's training data), and dataset B with `event='PEG'`. A dataset/event combination that never occurs, such as dataset C with NIV, fails loudly.
- **Add U as a third site.** In simulation: `flwr run . --stream --federation-config "num-supernodes=3" --run-config "sites='T,L,U'"`. In deployment, add a third SuperNode with `site='U'`. U has only 14 training patients: look at which columns survive the coverage rule and at U's C-index.
- **Change the coverage rule.** With `min-coverage=0.4`, `C9orf72` is kept (44% at L). Or add a column to `SCHEMA` and `load_site` in `features.py`, and see whether every site can provide it.
- **Change the time grid.** Try `time-step=3.0` (edit `pyproject.toml` so that `centralised.py` uses it too). What happens to the coefficients, the C-index, the federated == pooled check and the number of grid months with a single patient leaving the risk set?
- **Secure aggregation.** Hide each site's contribution to the sums as in workshop 5 (`flower-secure-aggregation`).

## Configuration

All settings are under `[tool.flwr.app.config]` in `pyproject.toml`. Override them for one Flower run with `flwr run . --run-config "key=value ..."`.

| key | default | meaning |
|---|---|---|
| `data-path` | `"../data/ALS_federated"` | folder with `T/`, `L/`, `U/`, relative to this folder (the repository's `data/`). A site can override it with node-config `data-path` |
| `sites` | `"T,L"` | training sites. The server waits for this many and checks their names |
| `dataset` | `"datasetC"` | `datasetA` (NIV or death), `datasetB` (PEG or death), `datasetC` (death only) |
| `event` | `"DEATH"` | event = `Occured == 1` and `Type == event`; every other outcome is censored |
| `window` | `6.0` | features use data up to this month only |
| `time-step` | `1.0` | public time grid of the Cox model, in months |
| `time-max` | `600.0` | end of the grid, in months. A later time is an error |
| `preprocessing` | `"flower"` | `"flower"`: stage 1 over Flower. `"file"`: load `preprocessing-file` |
| `preprocessing-file` | `""` | the preprocessing artifact to load, for example from the FA network |
| `min-coverage` | `0.5` | keep a column only if it is observed for at least this fraction at every site |
| `l2` | `0.0` | ridge penalty (0 = plain Cox) |
| `max-iter` | `25` | maximum number of Newton iterations (stops when max \|step\| < 1e-10) |
| `output-dir` | `"outputs"` | where `preprocessing.json`, `model.json` and the plots are written |

> **Tip:** `pyproject.toml` holds the app's dependencies and its run config. Which SuperLink a run goes to (the built-in `local` simulation, or a SuperLink of the Deployment Engine) is set in the Flower configuration file `~/.flwr/config.toml`, and the simulation size with `--federation-config` or `flwr federation simulation-config`.
> Learn more in the [Flower configuration guide](https://flower.ai/docs/framework/ref-flower-configuration.html).


## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) and [How to run Flower with the Deployment Engine](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html)
- [Secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)
