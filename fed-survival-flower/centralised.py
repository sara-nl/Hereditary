"""Centralised baseline: the model you would get by pooling the sites' training data.

This is the target the federated run has to reproduce. It uses the same schema table, the
same preprocessing artifact (build_artifact over the per-site summaries) and the same
model file format as the federated run, but pools the encoded training data and fits the
Cox model independently with scikit-survival. Pooling is only possible because this is a
workshop: in the project the patient-level data never leaves the sites.

    python centralised.py            # writes outputs/centralised_model.json
"""

from pathlib import Path

import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator

from fed_survival import read_config
from fed_survival.analytics import build_artifact, site_summary
from fed_survival.cox import c_index, grid_index, public_grid, save_model
from fed_survival.features import encode, load_site

config = read_config()
sites = sorted(s.strip() for s in config["sites"].split(","))


def load(site, split):
    return load_site(config["data-path"], site, config["dataset"], split, config["window"], config["event"])


# 1. Every site builds its schema table (train and test split)
train = {s: load(s, "train") for s in sites}
test = {s: load(s, "test") for s in sites}

# 2. The same preprocessing artifact as the federated run
artifact = build_artifact({s: site_summary(train[s][0]) for s in sites}, config["min-coverage"])
print(f"Features ({len(artifact['features'])}): {', '.join(artifact['features'])}")
for col, reason in artifact["dropped"].items():
    print(f"Dropped {col}: {reason}")

# 3. Pool the encoded training data of all sites
X = np.vstack([encode(train[s][0], artifact) for s in sites])
time = np.concatenate([train[s][1] for s in sites])
event = np.concatenate([train[s][2] for s in sites])
if not event.any():
    raise ValueError(f"no {config['event']} events in {config['dataset']}: check dataset/event")
print(f"Pooled training data: {len(time)} patients, {event.sum()} {config['event']} events")

# 4. Fit a Breslow Cox model on the same public monthly grid as the federated run: every time
#    is rounded up to its grid month. scikit-survival minimises -loglik/n + alpha*|beta|^2/(2n),
#    so its alpha is exactly our l2 (we maximise loglik - l2*|beta|^2/2).
grid = public_grid(config["time-step"], config["time-max"])
time = grid[grid_index(time, grid)]
y = np.array(list(zip(event, time)), dtype=[("event", bool), ("time", float)])
fit = CoxPHSurvivalAnalysis(alpha=config["l2"], ties="breslow", tol=1e-12, n_iter=100).fit(X, y)
beta = fit.coef_

# 5. Evaluate on each site's own test split (exact times)
cindex = {s: c_index(encode(test[s][0], artifact), test[s][1], test[s][2], beta)[0] for s in sites}

print(f"\n{'feature':<22}{'beta':>9}{'hazard ratio per SD':>22}")
for name, b in zip(artifact["features"], beta):
    print(f"{name:<22}{b:>9.4f}{np.exp(b):>22.3f}")
print("\nTest C-index: " + ", ".join(f"{s} {c:.3f} (n={len(test[s][1])})" for s, c in cindex.items()))

# 6. Save in the same format as the federated model
km_time, km_survival = kaplan_meier_estimator(event, time)
output = Path(config["output-dir"])
output.mkdir(parents=True, exist_ok=True)
save_model(
    {
        "dataset": config["dataset"],
        "event": config["event"],
        "window": config["window"],
        "time_step": config["time-step"],
        "time_max": config["time-max"],
        "l2": config["l2"],
        "trained_with": "centralised: scikit-survival CoxPHSurvivalAnalysis on pooled " + ",".join(sites),
        "preprocessing": artifact,
        "features": artifact["features"],
        "beta": beta,
        "baseline": {"time": fit.cum_baseline_hazard_.x, "cumulative_hazard": fit.cum_baseline_hazard_.y},
        "kaplan_meier": {"time": km_time, "survival": km_survival},
        "test_cindex": cindex,
        "n_train": {s: len(train[s][1]) for s in sites},
        "iterations": None,
        "loglik": [],  # scikit-survival does not report these
    },
    output / "centralised_model.json",
)
print(f"Saved {output / 'centralised_model.json'}")
