"""Stage 2, federated Cox model: WebDISCO (Lu et al., JAMIA 2015).

A Cox model cannot be trained by averaging site models: every patient's risk set spans
all sites. But the gradient and Hessian of the Breslow partial likelihood are SUMS over
patients, so the server can take the exact pooled Newton step from per-site sums:

    grid     a PUBLIC time grid t_1..t_K (whole months) that every site knows in advance;
             each time T_j is rounded up to its grid month.  w_j = exp(x_j . beta)
    round A  site:    d_k = number of events in month t_k,  S0_k = sum_{T_j >= t_k} w_j,
                      S1_k = sum_{T_j >= t_k} w_j x_j,  x_events = sum of x over its events
             server:  sums them; loglik, gradient g and c_k = d_k / S0_k
    round B  site:    M = sum_j w_j a_j x_j x_j^T  with  a_j = sum_{t_k <= T_j} c_k  (p x p)
             server:  I = sum(M) - sum_k d_k m_k m_k^T + l2 * Id  (m_k = S1_k / S0_k);
                      beta <- beta + solve(I, g)

The result equals a pooled Breslow Cox fit on the same month-rounded times to rounding error.
Ties are handled with Breslow (Efron's correction does not split into per-site sums).
Nothing a site returns has one row per patient, and because the grid is public no site
reveals its exact event times.
"""

import json

import numpy as np

MODEL_FORMAT = "fed-survival-cox/1"
MODEL_KEYS = [
    "dataset",
    "event",
    "window",
    "time_step",
    "time_max",
    "l2",
    "trained_with",
    "preprocessing",
    "features",
    "beta",
    "baseline",
    "kaplan_meier",
    "test_cindex",
    "n_train",
    "iterations",
    "loglik",
]


# ---- site side -------------------------------------------------------------------------


def public_grid(time_step, time_max):
    """The public time grid: time_step, 2 * time_step, ... up to time_max (months)."""
    return np.arange(1, int(round(time_max / time_step)) + 1) * float(time_step)


def grid_index(time, grid):
    """Index of each patient's grid month: the first grid point >= time (rounded up)."""
    if time.max() > grid[-1]:
        raise ValueError(f"a time of {time.max():.1f} months is beyond the grid (time-max)")
    return np.searchsorted(grid, time, side="left")


def risk_set_sums(X, time, event, beta, grid):
    """Site, round A: {"d": (K,), "S0": (K,), "S1": (K, p), "x_events": (p,)}, all float64."""
    k = grid_index(time, grid)
    w = np.exp(X @ beta)
    at_risk = (k[None, :] >= np.arange(len(grid))[:, None]).astype(np.float64)  # K x n
    return {
        "d": np.bincount(k[event], minlength=len(grid)).astype(np.float64),
        "S0": at_risk @ w,
        "S1": at_risk @ (w[:, None] * X),
        "x_events": X[event].sum(axis=0),
    }


def hessian_sum(X, time, beta, grid, c):
    """Site, round B: M = sum_j w_j a_j x_j x_j^T, a (p, p) float64 matrix."""
    w = np.exp(X @ beta)
    a = np.cumsum(c)[grid_index(time, grid)]  # a_j = sum_{t_k <= T_j} c_k
    return X.T @ (X * (w * a)[:, None])


def c_index(X, time, event, beta):
    """Site: Harrell's C-index of risk = X beta on its own data. Returns (c, n).

    Uses the exact (not month-rounded) times. c is NaN when it is undefined (fewer than two
    patients, or no comparable pairs).
    """
    from sksurv.metrics import concordance_index_censored  # lazy: a slow import

    try:
        with np.errstate(invalid="ignore"):
            c = concordance_index_censored(event, time, X @ beta)[0]
    except ValueError:
        c = np.nan
    return float(c), len(time)


# ---- server side -----------------------------------------------------------------------


def total(replies):
    """Server: sum {site: reply} in a fixed order (sorted by name), so runs are reproducible.
    A reply is an array or a dict of arrays."""
    sites = sorted(replies)
    if isinstance(replies[sites[0]], dict):
        return {k: sum(replies[s][k] for s in sites) for k in replies[sites[0]]}
    return sum(replies[s] for s in sites)


def hazard_steps(A):
    """Server: c_k = d_k / S0_k, 0 in months without events (where S0_k may also be 0)."""
    c = np.zeros_like(A["d"])
    e = A["d"] > 0  # S0_k > 0 there: the patient with the event is at risk
    c[e] = A["d"][e] / A["S0"][e]
    return c


def score(A, beta, l2):
    """Server: (loglik, gradient, c) from the summed round A. c = d / S0 is sent out for round B."""
    e = A["d"] > 0
    if not e.any():
        raise ValueError("the chosen event never occurs in the training data: check dataset/event")
    c = hazard_steps(A)
    loglik = beta @ A["x_events"] - A["d"][e] @ np.log(A["S0"][e]) - 0.5 * l2 * beta @ beta
    gradient = A["x_events"] - c @ A["S1"] - l2 * beta  # c_k S1_k = d_k m_k
    return float(loglik), gradient, c


def newton_step(A, M, beta, gradient, l2):
    """Server: the next beta, from the summed round A and summed round B."""
    e = A["d"] > 0
    m = A["S1"][e] / A["S0"][e, None]
    information = M - (m * A["d"][e, None]).T @ m + l2 * np.eye(len(beta))
    return beta + np.linalg.solve(information, gradient)


def baseline_hazard(A, grid):
    """Server: Breslow baseline cumulative hazard H0(t_k) = sum_{t_i <= t_k} d_i / S0_i, at the
    months with events: (time, H0). Use the round A computed at the FINAL beta."""
    e = A["d"] > 0
    return grid[e], np.cumsum(hazard_steps(A))[e]


def kaplan_meier(A0, grid):
    """Server: Kaplan-Meier survival of the training population, (time, survival).
    Needs the round A at beta = 0, where S0 is the number at risk and d the number of events."""
    e = A0["d"] > 0
    return grid[e], np.cumprod(1.0 - hazard_steps(A0))[e]


# ---- prediction and model file ---------------------------------------------------------


def step(knots, values, t, before):
    """Right-continuous step function: values[i] on [knots[i], knots[i+1]), `before` left of
    the first knot, held flat after the last one."""
    i = np.searchsorted(knots, t, side="right") - 1
    return np.where(i >= 0, np.asarray(values, np.float64)[np.maximum(i, 0)], before)


def survival(model, X, times):
    """S(t | x) = exp(-H0(t) * exp(x . beta)) for each row of the encoded X: (n, len(times))."""
    H0 = step(model["baseline"]["time"], model["baseline"]["cumulative_hazard"], times, 0.0)
    return np.exp(-np.outer(np.exp(X @ np.asarray(model["beta"])), H0))


def save_model(model, path):
    """Write the model JSON. Both the federated run and centralised.py use this, same keys."""
    missing = [k for k in MODEL_KEYS if k not in model]
    if missing:
        raise ValueError(f"model is missing {missing}")
    with open(path, "w") as f:
        json.dump({"format": MODEL_FORMAT, **model}, f, indent=2, default=lambda o: o.tolist())


def load_model(path):
    """Read a model JSON written by save_model()."""
    with open(path) as f:
        model = json.load(f)
    if model.get("format") != MODEL_FORMAT:
        raise ValueError(f"{path}: expected format {MODEL_FORMAT!r}, got {model.get('format')!r}")
    return model
