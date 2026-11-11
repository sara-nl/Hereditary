"""Stage 1, federated analytics: decide the preprocessing without pooling any data.

Which columns to keep, which category levels exist, and the means and stds used for
imputation and scaling are all computed from ONE round of aggregate queries:

    site_summary(table)             at each site: counts, sums and sums of squares
    build_artifact(summaries, ...)  at the server: plain sums, a per-site coverage check
                                    and a union of level names

The result, the "preprocessing artifact", is a small JSON file and the only thing stage 2
(training) receives. This file is the part the HEREDITARY federated analytics (FA) network
replaces in the workshop exercise: it only has to produce the same JSON.

Privacy note: a level count of 1 or 2 at a small site can single out a patient. A
production FA network applies a minimum cell size; this workshop code does not.
"""

import json
import math

from fed_survival.features import ARTIFACT_FORMAT, SCHEMA


def site_summary(table):
    """One site's answer to the preprocessing query (JSON-serialisable, no per-patient rows).

    {"n": 1199,
     "numeric":     {"age_onset": {"n_observed": 1199, "sum": ..., "sum_sq": ...}, ...},
     "categorical": {"sex": {"Female": 574, "Male": 625}, "C9orf72": {...}}}
    """
    summary = {"n": int(len(table)), "numeric": {}, "categorical": {}}
    for col, kind in SCHEMA.items():
        x = table[col].dropna()
        if kind == "numeric":
            x = x.to_numpy(float)
            summary["numeric"][col] = {"n_observed": int(x.size), "sum": float(x.sum()), "sum_sq": float((x * x).sum())}
        else:
            summary["categorical"][col] = {str(k): int(v) for k, v in x.value_counts().items()}
    return summary


def build_artifact(summaries, min_coverage=0.5):
    """Combine {site: summary} into the preprocessing artifact (a JSON-serialisable dict)."""
    sites = sorted(summaries)  # fixed order -> reproducible sums
    features, mean, std, levels, dropped = [], [], [], {}, {}

    for col, kind in SCHEMA.items():
        parts = [summaries[s][kind][col] for s in sites]

        # 1. Coverage: keep a column only if EVERY site observes it for enough patients
        n_obs = [p["n_observed"] if kind == "numeric" else sum(p.values()) for p in parts]
        coverage = {s: n / summaries[s]["n"] for s, n in zip(sites, n_obs)}
        low = {s: f for s, f in coverage.items() if f < min_coverage}
        if low:
            where = ", ".join(f"{f:.0%} at {s}" for s, f in low.items())
            dropped[col] = f"observed for {where} (needs {min_coverage:.0%} at every site)"
            continue

        # 2. Categorical: reference = first level, one 0/1 dummy "col=level" per other level
        if kind == "categorical":
            levels[col] = sorted(set().union(*parts))
            if len(levels[col]) < 2:
                dropped[col] = f"only one level observed ({levels[col][0]})"
                continue
            for level in levels[col][1:]:
                m = sum(p.get(level, 0) for p in parts) / sum(n_obs)
                features.append(f"{col}={level}")
                mean.append(m)
                std.append(math.sqrt(m * (1 - m)))
            continue

        # 3. Numeric: pooled mean and (population) std from the summed moments
        n = sum(n_obs)
        m = sum(p["sum"] for p in parts) / n
        var = sum(p["sum_sq"] for p in parts) / n - m * m
        if var <= 1e-12 * (1 + m * m):  # zero up to rounding: a constant column
            dropped[col] = f"constant (value {m:g} at every site)"
            continue
        features.append(col)
        mean.append(m)
        std.append(math.sqrt(var))

    return {
        "format": ARTIFACT_FORMAT,
        "features": features,
        "mean": mean,
        "std": std,
        "levels": levels,
        "dropped": dropped,
        "min_coverage": min_coverage,
        "n_patients": {s: summaries[s]["n"] for s in sites},
    }


def save_artifact(artifact, path):
    """Write the artifact as JSON (floats round-trip exactly)."""
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)
