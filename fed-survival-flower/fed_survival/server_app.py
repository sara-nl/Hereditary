"""Flower ServerApp: the coordinator. It never sees patient data, only per-site sums.

    stage 1  run_preprocessing  query.summary -> build_artifact -> outputs/preprocessing.json
             (skipped with preprocessing = "file": the artifact then comes from
             preprocessing-file, e.g. one produced by the HEREDITARY federated analytics network)
    stage 2  run_training       Newton-Raphson: train.risk_sets (round A), train.hessian (round B)
    stage 3  run_evaluation     evaluate.cindex: each site's C-index on its own test split
    then     outputs/model.json

The only thing stage 2 receives from stage 1 is the preprocessing JSON file. There is no
Strategy and no FedAvg: the server sums the sites' replies and takes the Newton step itself.
"""

import json
import time as clock
from pathlib import Path

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.serverapp import Grid, ServerApp

from fed_survival import cox
from fed_survival.analytics import build_artifact, save_artifact
from fed_survival.features import load_artifact

app = ServerApp()
TIMEOUT = 600.0  # seconds to wait for the replies to one message


def log(*args):
    print(*args, flush=True)  # flush: Flower may block-buffer the ServerApp's output


class SiteError(Exception):
    """A site is missing or failed. Not a RuntimeError: a deployed Flower 1.25 ServerApp
    reports a RuntimeError as "Failed to start run" and hides its message."""


def connect(grid: Grid, n: int, timeout: float = 300.0) -> list[int]:
    """Wait until n SuperNodes (sites) have connected; returns their node ids."""
    start = clock.time()
    while len(node_ids := sorted(grid.get_node_ids())) < n:
        if clock.time() - start > timeout:
            raise SiteError(f"only {len(node_ids)} of {n} sites connected")
        clock.sleep(1.0)
    return node_ids


def ask(grid: Grid, node_ids, sites, message_type, **records) -> dict:
    """Send one message to every site and wait for all replies: {site: reply content}.

    A missing reply or an error is fatal: a sum over some of the sites is not the
    federated estimator."""
    messages = [Message(RecordDict(records), message_type=message_type, dst_node_id=node) for node in node_ids]
    replies = list(grid.send_and_receive(messages, timeout=TIMEOUT))
    errors = [f"node {r.metadata.src_node_id}: {r.error.reason}" for r in replies if r.has_error()]
    if errors or len(replies) < len(node_ids):
        raise SiteError(f"{message_type}: {len(replies)} of {len(node_ids)} replies, {errors}")
    contents = {str(r.content["config"]["site"]): r.content for r in replies}
    if sorted(contents) != sites:
        raise SiteError(f"expected sites {sites}, but got replies from {sorted(contents)}")
    return {s: contents[s] for s in sites}  # fixed order: sorted by site name


def arrays(content) -> dict:
    return {key: array.numpy() for key, array in content["arrays"].items()}


def run_preprocessing(grid, node_ids, sites, cfg, path):
    """Stage 1: one round of federated analytics. Writes the preprocessing artifact to path."""
    replies = ask(grid, node_ids, sites, "query.summary", config=ConfigRecord())
    summaries = {s: json.loads(r["config"]["summary"]) for s, r in replies.items()}
    log("Training patients: " + ", ".join(f"{s} {m['n']}" for s, m in summaries.items()))
    save_artifact(build_artifact(summaries, float(cfg["min-coverage"])), path)


def run_training(grid, node_ids, sites, cfg, artifact):
    """Stage 2: federated Newton-Raphson on the Breslow partial likelihood (WebDISCO)."""
    l2, max_iter = float(cfg["l2"]), int(cfg["max-iter"])
    time_grid = cox.public_grid(float(cfg["time-step"]), float(cfg["time-max"]))
    preprocessing = ConfigRecord({"artifact": json.dumps(artifact)})

    def round_a(beta):  # {site: {"d", "S0", "S1", "x_events"}}
        replies = ask(
            grid, node_ids, sites, "train.risk_sets", config=preprocessing, arrays=ArrayRecord({"beta": Array(beta)})
        )
        return {s: arrays(r) for s, r in replies.items()}

    def round_b(beta, c):  # the summed p x p matrix M
        replies = ask(
            grid,
            node_ids,
            sites,
            "train.hessian",
            config=preprocessing,
            arrays=ArrayRecord({"beta": Array(beta), "c": Array(c)}),
        )
        return cox.total({s: arrays(r)["M"] for s, r in replies.items()})

    # 1. Round A at beta = 0 also gives the number at risk and the number of events per month:
    #    a federated Kaplan-Meier curve, and (S0 of the first month) each site's n.
    beta = np.zeros(len(artifact["features"]))
    first = round_a(beta)
    n_train = {s: int(round(a["S0"][0])) for s, a in first.items()}
    A0 = A = cox.total(first)
    log(
        f"Public time grid: {len(time_grid)} months; {int(A['d'].sum())} events in "
        f"{int((A['d'] > 0).sum())} of them"
    )

    # 2. Newton iterations: round A -> loglik, gradient, c; round B -> M; step
    log(f"{'iteration':>9}{'log-likelihood':>18}{'max |step|':>12}")
    loglik = []
    for iteration in range(1, max_iter + 1):
        ll, gradient, c = cox.score(A, beta, l2)
        loglik.append(ll)
        new_beta = cox.newton_step(A, round_b(beta, c), beta, gradient, l2)
        step, beta = np.max(np.abs(new_beta - beta)), new_beta
        A = cox.total(round_a(beta))  # round A at the new beta: next iteration, or the final
        log(f"{iteration:>9}{ll:>18.6f}{step:>12.2e}")
        if step < 1e-10:
            break
    else:
        log(f"WARNING: not converged after {max_iter} iterations")
    loglik.append(cox.score(A, beta, l2)[0])

    # 3. Breslow baseline hazard at the FINAL beta (the last round A), KM at beta = 0
    baseline_time, cumulative_hazard = cox.baseline_hazard(A, time_grid)
    km_time, km_survival = cox.kaplan_meier(A0, time_grid)
    return {
        "beta": beta,
        "iterations": iteration,
        "loglik": loglik,
        "n_train": n_train,
        "baseline": {"time": baseline_time, "cumulative_hazard": cumulative_hazard},
        "kaplan_meier": {"time": km_time, "survival": km_survival},
    }


def run_evaluation(grid, node_ids, sites, artifact, beta):
    """Stage 3: every site evaluates the final model on its own test split."""
    replies = ask(
        grid,
        node_ids,
        sites,
        "evaluate.cindex",
        config=ConfigRecord({"artifact": json.dumps(artifact)}),
        arrays=ArrayRecord({"beta": Array(beta)}),
    )
    for s, r in replies.items():
        log(f"Test C-index {s}: {r['metrics']['c_index']:.4f} (n={r['metrics']['n']})")
    return {s: float(r["metrics"]["c_index"]) for s, r in replies.items()}


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    sites = sorted(s.strip() for s in str(cfg["sites"]).split(","))
    output = Path(str(cfg["output-dir"])).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log(f"\n=== fed-survival: {cfg['dataset']}, event {cfg['event']}, sites " f"{','.join(sites)} ===")
    node_ids = connect(grid, len(sites))
    start = clock.time()

    # Stage 1: preprocessing -> a JSON file, the only thing stage 2 receives
    if cfg["preprocessing"] == "flower":
        log("\n--- Stage 1: federated preprocessing (query.summary) ---")
        path = output / "preprocessing.json"
        run_preprocessing(grid, node_ids, sites, cfg, path)
    elif cfg["preprocessing"] == "file":
        log("\n--- Stage 1: skipped, preprocessing from file ---")
        path = Path(str(cfg["preprocessing-file"])).resolve()
    else:
        raise ValueError(f"preprocessing must be 'flower' or 'file', not {cfg['preprocessing']!r}")
    artifact = load_artifact(path)
    log(f"Preprocessing: {path}\nFeatures ({len(artifact['features'])}): " + ", ".join(artifact["features"]))
    for col, reason in artifact.get("dropped", {}).items():
        log(f"Dropped {col}: {reason}")

    log("\n--- Stage 2: federated Cox training (train.risk_sets, train.hessian) ---")
    model = run_training(grid, node_ids, sites, cfg, artifact)
    log(f"\n{'feature':<22}{'beta':>9}{'hazard ratio per SD':>22}")
    for name, b in zip(artifact["features"], model["beta"]):
        log(f"{name:<22}{b:>9.4f}{np.exp(b):>22.3f}")

    log("\n--- Stage 3: evaluation on each site's test split (evaluate.cindex) ---")
    test_cindex = run_evaluation(grid, node_ids, sites, artifact, model["beta"])

    cox.save_model(
        {
            "dataset": str(cfg["dataset"]),
            "event": str(cfg["event"]),
            "window": float(cfg["window"]),
            "time_step": float(cfg["time-step"]),
            "time_max": float(cfg["time-max"]),
            "l2": float(cfg["l2"]),
            "trained_with": f"federated: Flower, WebDISCO Newton over sites {','.join(sites)}",
            "preprocessing": artifact,
            "features": artifact["features"],
            "test_cindex": test_cindex,
            **model,
        },
        output / "model.json",
    )
    log(f"\nSaved {output / 'model.json'} ({clock.time() - start:.1f} s)")
