"""Flower ClientApp: one site. Thin handlers around features.py, analytics.py and cox.py.

Each handler reads the site's own data, computes an aggregate and replies with it: nothing
with one row per patient leaves this file. Every message is self-contained (the server sends
beta, c and the preprocessing artifact each time), because in a deployment every message is
handled by a fresh process.

    query.summary     stage 1: preprocessing summary (counts, sums, sums of squares)
    train.risk_sets   stage 2, round A: d, S0, S1, x_events at the current beta
    train.hessian     stage 2, round B: the matrix M at the current beta, given c
    evaluate.cindex   stage 3: C-index on the site's own test split
"""

import json
from functools import lru_cache

from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fed_survival import cox
from fed_survival.analytics import site_summary
from fed_survival.features import encode, load_site

app = ClientApp()
cached_load_site = lru_cache(load_site)  # a process may handle several messages


def site_name(context: Context) -> str:
    """node-config "site" in a deployment; partition-id 0, 1, 2 = T, L, U in simulation."""
    if "site" in context.node_config:
        return str(context.node_config["site"])
    return ["T", "L", "U"][int(context.node_config["partition-id"])]


def load(context: Context, split: str):
    """This site's (table, time, event) for the run's dataset, event and window."""
    cfg = context.run_config
    data_path = context.node_config.get("data-path", cfg["data-path"])  # a site may override
    return cached_load_site(
        str(data_path), site_name(context), str(cfg["dataset"]), split, float(cfg["window"]), str(cfg["event"])
    )


def encoded(msg: Message, context: Context, split: str):
    """This site's data, encoded with the preprocessing artifact sent with the message."""
    table, time, event = load(context, split)
    return encode(table, json.loads(msg.content["config"]["artifact"])), time, event


def public_grid(context: Context):
    return cox.public_grid(float(context.run_config["time-step"]), float(context.run_config["time-max"]))


def reply(msg: Message, context: Context, **records) -> Message:
    """The reply to msg. It always says which site it comes from."""
    records.setdefault("config", ConfigRecord())["site"] = site_name(context)
    return Message(RecordDict(records), reply_to=msg)


@app.query("summary")
def summary(msg: Message, context: Context) -> Message:
    """Stage 1: the preprocessing summary of this site's training data (as JSON)."""
    table, _, _ = load(context, "train")
    print(f"[site {site_name(context)}] preprocessing summary of {len(table)} patients")
    return reply(msg, context, config=ConfigRecord({"summary": json.dumps(site_summary(table))}))


@app.train("risk_sets")
def risk_sets(msg: Message, context: Context) -> Message:
    """Stage 2, round A: the risk-set sums at the current beta."""
    X, time, event = encoded(msg, context, "train")
    beta = msg.content["arrays"]["beta"].numpy()
    sums = cox.risk_set_sums(X, time, event, beta, public_grid(context))
    return reply(msg, context, arrays=ArrayRecord({k: Array(v) for k, v in sums.items()}))


@app.train("hessian")
def hessian(msg: Message, context: Context) -> Message:
    """Stage 2, round B: M = sum_j w_j a_j x_j x_j^T at the current beta, given c."""
    X, time, _ = encoded(msg, context, "train")
    beta, c = msg.content["arrays"]["beta"].numpy(), msg.content["arrays"]["c"].numpy()
    M = cox.hessian_sum(X, time, beta, public_grid(context), c)
    return reply(msg, context, arrays=ArrayRecord({"M": Array(M)}))


@app.evaluate("cindex")
def cindex(msg: Message, context: Context) -> Message:
    """Stage 3: Harrell's C-index of the final model on this site's own test split."""
    X, time, event = encoded(msg, context, "test")
    c, n = cox.c_index(X, time, event, msg.content["arrays"]["beta"].numpy())
    print(f"[site {site_name(context)}] test C-index {c:.4f} (n={n})")
    return reply(msg, context, metrics=MetricRecord({"c_index": c, "n": n}))
