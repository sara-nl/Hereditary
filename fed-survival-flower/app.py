"""Inference UI: apply the global (federated) model to one patient and show every step.

    python app.py                 # http://127.0.0.1:8080
    python app.py --host 0.0.0.0  # inside Docker (publish the port on 127.0.0.1 only)

A small standard-library web server around the same functions predict.py uses
(features.encode, cox.survival). For every prediction it returns the full working:

    schema value -> impute with the federated mean -> z-score with the federated std
    -> x_i * beta_i -> risk score x.beta -> hazard ratio exp(x.beta)
    -> S(t) = exp(-H0(t) * exp(x.beta)) with the Breslow baseline hazard H0 of the model

and checks the step-by-step numbers against encode() and survival(), so the page shows
what the model really computes. Patient data is read from this site's own files and never
leaves this machine. Research use only. Not a medical device.
"""

import argparse
import json
import math
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from fed_survival import read_config
from fed_survival.cox import MODEL_FORMAT, load_model, step, survival
from fed_survival.features import SCHEMA, encode, load_site

CONFIG = read_config()
OUTPUT = Path(CONFIG["output-dir"])
DATA = Path(CONFIG["data-path"])
PAGE = Path(__file__).resolve().parent / "webui" / "index.html"
SITES = ["T", "L", "U"]
SPLITS = ["test", "train"]
HORIZONS = [12.0, 24.0, 36.0]
CURVE_MAX = 120.0  # months shown in the plot


# ---- models -----------------------------------------------------------------------------


def list_models():
    """Every model JSON under the output folder: {id: path}, the federated model first."""
    found = {}
    for path in sorted(OUTPUT.rglob("*.json")):
        try:
            with open(path) as f:
                if json.load(f).get("format") == MODEL_FORMAT:
                    found[str(path.relative_to(OUTPUT))] = path
        except (OSError, ValueError, AttributeError):
            continue
    return dict(sorted(found.items(), key=lambda kv: (kv[0].endswith("centralised_model.json"), kv[0])))


@lru_cache(maxsize=32)
def _load(path, mtime):
    return load_model(path)


def get_model(model_id):
    models = list_models()
    if model_id not in models:
        raise LookupError(f"unknown model {model_id!r}; available: {', '.join(models) or 'none'}")
    path = models[model_id]
    return _load(str(path), path.stat().st_mtime)


def twin(model_id):
    """The other model trained with the same settings (federated <-> centralised), if any."""
    models, folder = list_models(), str(Path(model_id).parent)
    name = "centralised_model.json" if model_id.endswith("/model.json") or model_id == "model.json" else "model.json"
    other = name if folder == "." else f"{folder}/{name}"
    if other not in models:
        return None, None
    m, o = get_model(model_id), get_model(other)
    if all(m[k] == o[k] for k in ("dataset", "event", "window", "time_step", "features")):
        return other, o
    return None, None


def model_summary(model_id):
    m = get_model(model_id)
    art = m["preprocessing"]
    other_id, _ = twin(model_id)
    return {
        "id": model_id,
        "kind": m["trained_with"].split(":")[0],
        "trained_with": m["trained_with"],
        "twin": other_id,
        **{k: m[k] for k in ("dataset", "event", "window", "time_step", "time_max", "l2", "iterations", "loglik")},
        "n_train": m["n_train"],
        "test_cindex": m["test_cindex"],
        "features": m["features"],
        "beta": m["beta"],
        "mean": art["mean"],
        "std": art["std"],
        "dropped": art.get("dropped", {}),
        "levels": art.get("levels", {}),
        "min_coverage": art.get("min_coverage"),
        "schema": SCHEMA,
        "baseline": m["baseline"],
        "kaplan_meier": m.get("kaplan_meier"),
        "sites": available_sites(m["dataset"]),
    }


# ---- site data ----------------------------------------------------------------------------


def available_sites(dataset):
    return [s for s in SITES if all((DATA / s / dataset / sp).is_dir() for sp in SPLITS)]


@lru_cache(maxsize=24)
def site_data(site, dataset, split, window, event):
    if site not in available_sites(dataset) or split not in SPLITS:
        raise LookupError(f"no data for site {site!r}, {dataset} {split} split under {DATA}")
    return load_site(str(DATA), site, dataset, split, window, event)


def data_for(m, site, split):
    return site_data(site, m["dataset"], split, float(m["window"]), m["event"])


def patients(model_id, site, split):
    m = get_model(model_id)
    table, time, event = data_for(m, site, split)
    return [{"id": pid, "time": float(t), "event": bool(e)} for pid, t, e in zip(table.index, time, event)]


def patient(model_id, site, split, pid):
    m = get_model(model_id)
    table, time, event = data_for(m, site, split)
    if pid not in table.index:
        raise LookupError(f"{pid} is not in the {split} split of site {site}")
    i = table.index.get_loc(pid)
    return {
        "id": pid,
        "site": site,
        "split": split,
        "values": {c: py(v) for c, v in table.iloc[i].items()},
        "observed": {"time": float(time[i]), "event": bool(event[i]), "name": m["event"]},
    }


# ---- the prediction, step by step ---------------------------------------------------------


def schema_row(values):
    """A one-row schema table from the form values (None / "" = missing)."""
    row = {}
    for col, kind in SCHEMA.items():
        v = values.get(col)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            row[col] = np.nan if kind == "numeric" else None
        elif kind == "numeric":
            row[col] = float(v)
        else:
            row[col] = str(v)
    table = pd.DataFrame([row], index=["input"])
    for col, kind in SCHEMA.items():
        table[col] = table[col].astype(float) if kind == "numeric" else table[col].astype(object)
    return table


def predict(model_id, values, site=None, split=None):
    m = get_model(model_id)
    art = m["preprocessing"]
    beta = np.asarray(m["beta"], np.float64)
    table = schema_row(values)

    # 1-2. Encoding, one feature at a time (written out, so it can be shown)
    steps = []
    for name, b, mu, sd in zip(m["features"], beta, art["mean"], art["std"]):
        if "=" in name:
            col, level = name.split("=", 1)
            raw = table[col].iloc[0]
            x = None if raw is None or pd.isna(raw) else float(str(raw) == level)
            kind = "dummy"
        else:
            col, level, kind = name, None, "numeric"
            raw = table[col].iloc[0]
            x = None if pd.isna(raw) else float(raw)
        used = mu if x is None else x
        z = (used - mu) / sd
        steps.append(
            {
                "feature": name,
                "column": col,
                "kind": kind,
                "level": level,
                "raw": py(raw),
                "x": x,
                "missing": x is None,
                "used": used,
                "mean": mu,
                "std": sd,
                "z": z,
                "beta": float(b),
                "contribution": z * float(b),
            }
        )
    z = np.array([s["z"] for s in steps])
    lp = float(sum(s["contribution"] for s in steps))

    # Check against the production code path
    x_encoded = encode(table, art)[0]
    t = np.concatenate([[0.0], [u for u in m["baseline"]["time"] if u <= CURVE_MAX]])
    if t[-1] < CURVE_MAX:
        t = np.append(t, CURVE_MAX)
    S_official = survival(m, x_encoded[None, :], t)[0]
    H0 = step(m["baseline"]["time"], m["baseline"]["cumulative_hazard"], t, 0.0)
    S_manual = np.exp(-H0 * math.exp(lp))

    # 3-4. Hazard ratio, baseline hazard, survival at the horizons
    hr = math.exp(lp)
    horizons = []
    for h in HORIZONS:
        H0_h = float(step(m["baseline"]["time"], m["baseline"]["cumulative_hazard"], [h], 0.0)[0])
        S_h = math.exp(-H0_h * hr)
        horizons.append({"t": h, "H0": H0_h, "H": H0_h * hr, "S": S_h, "P": 1 - S_h})
    all_t = np.concatenate([[0.0], m["baseline"]["time"]])
    S_all = survival(m, x_encoded[None, :], all_t)[0]
    below = np.nonzero(S_all <= 0.5)[0]
    median = float(all_t[below[0]]) if below.size else None

    out = {
        "model": model_id,
        "event": m["event"],
        "steps": steps,
        "lp": lp,
        "hazard_ratio": hr,
        "horizons": horizons,
        "median": median,
        "median_limit": float(all_t[-1]),
        "dropped": [
            {"column": c, "reason": r, "raw": py(table[c].iloc[0]) if c in table else None}
            for c, r in art.get("dropped", {}).items()
        ],
        "curve": {"t": t, "S": S_official, "H0": H0, "S_average": np.exp(-H0)},
        "check": {
            "encode_max_diff": float(np.max(np.abs(x_encoded - z))),
            "survival_max_diff": float(np.max(np.abs(S_official - S_manual))),
            "lp_encode": float(x_encoded @ beta),
        },
        "window": m["window"],
        "kaplan_meier": m.get("kaplan_meier"),
    }

    # The same patient through the twin model (federated <-> centralised): should overlap
    other_id, other = twin(model_id)
    if other is not None:
        xo = encode(table, other["preprocessing"])
        out["twin"] = {
            "model": other_id,
            "kind": other["trained_with"].split(":")[0],
            "lp": float((xo @ np.asarray(other["beta"]))[0]),
            "S": survival(other, xo, t)[0],
        }

    # Where this patient sits among the site's patients (same split)
    if site and split:
        try:
            tab, _, _ = data_for(m, site, split)
            lps = encode(tab, art) @ beta
            out["population"] = {
                "site": site,
                "split": split,
                "n": int(len(lps)),
                "percentile": float((lps < lp).mean() * 100),
                "lp": sorted(float(v) for v in lps),
            }
        except LookupError:
            pass
    return out


# ---- JSON + HTTP ---------------------------------------------------------------------------


def py(v):
    """numpy / pandas scalars and arrays -> JSON-safe Python (NaN -> None)."""
    if isinstance(v, dict):
        return {k: py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [py(x) for x in v]
    if v is None or v is pd.NA:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) or math.isinf(v) else float(v)
    return str(v)


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(py(body), allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def handle_api(self, fn):
        try:
            self.send(200, fn())
        except LookupError as e:
            self.send(404, {"error": str(e).strip("'\"")})
        except (ValueError, TypeError, KeyError) as e:
            self.send(400, {"error": f"{type(e).__name__}: {e}"})
        except Exception as e:  # keep the server alive, show the reason in the page
            self.send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        routes = {
            "/api/models": lambda: {
                "models": [model_summary(i) | {"baseline": None, "kaplan_meier": None} for i in list_models()],
                "output": str(OUTPUT),
            },
            "/api/model": lambda: model_summary(q["model"]),
            "/api/patients": lambda: patients(q["model"], q["site"], q["split"]),
            "/api/patient": lambda: patient(q["model"], q["site"], q["split"], q["id"]),
        }
        if url.path in ("/", "/index.html"):
            self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path in routes:
            self.handle_api(routes[url.path])
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            return self.send(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", 0))
        if length > 100_000:
            return self.send(413, {"error": "request too large"})
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self.send(400, {"error": "invalid JSON"})
        self.handle_api(lambda: predict(body["model"], body.get("values", {}), body.get("site"), body.get("split")))

    def log_message(self, fmt, *args):  # quiet: only errors
        if args and str(args[1])[0] in "45":
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description="Inference UI for the federated survival model.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    models = list_models()
    if not models:
        raise SystemExit(f"No model in {OUTPUT}: run the federated training first (flwr run . / ./ws8.sh).")
    print(f"Models: {', '.join(models)}")
    print(f"Data:   {DATA} ({'found' if DATA.is_dir() else 'not found: manual input only'})")
    print(f"Inference UI on http://{'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}")
    print("Research use only. Not a medical device. Ctrl+C to stop.", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
