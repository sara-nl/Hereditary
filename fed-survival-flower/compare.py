"""Federated vs centralised: is the federated model the pooled model?

Reads outputs/model.json (the Flower run) and outputs/centralised_model.json
(python centralised.py) and prints the coefficients side by side, the largest differences
and the test C-index per site.

    python compare.py
"""

from pathlib import Path

import numpy as np

from fed_survival import read_config
from fed_survival.cox import load_model, step

output = Path(read_config()["output-dir"])
fed = load_model(output / "model.json")
cen = load_model(output / "centralised_model.json")
for key in ("dataset", "event", "window", "time_step", "time_max", "l2", "features"):
    if fed[key] != cen[key]:
        raise SystemExit(
            f"The two models differ in {key}: {fed[key]} vs {cen[key]}. " f"Rerun both with the same run config."
        )

# 1. Coefficients side by side
print(f"{'feature':<22}{'federated':>13}{'centralised':>13}{'difference':>12}")
for name, b_fed, b_cen in zip(fed["features"], fed["beta"], cen["beta"]):
    print(f"{name:<22}{b_fed:>13.6f}{b_cen:>13.6f}{b_fed - b_cen:>12.1e}")
d_beta = np.max(np.abs(np.subtract(fed["beta"], cen["beta"])))

# 2. Baseline cumulative hazard, on the union of both models' time points
t = np.union1d(fed["baseline"]["time"], cen["baseline"]["time"])
H_fed, H_cen = (step(m["baseline"]["time"], m["baseline"]["cumulative_hazard"], t, 0.0) for m in (fed, cen))
d_H = np.max(np.abs(H_fed - H_cen))

print(f"\nmax |delta beta|               {d_beta:.1e}")
print(f"max |delta cumulative hazard|  {d_H:.1e}   (H0 ranges 0 to {H_cen[-1]:.2f})")

# 3. Test C-index, per site (each site evaluates on its own test split)
print(f"\n{'test C-index':<14}{'federated':>11}{'centralised':>13}")
for s in sorted(fed["test_cindex"]):
    print(f"{s:<14}{fed['test_cindex'][s]:>11.4f}{cen['test_cindex'].get(s, np.nan):>13.4f}")

# 4. Verdict
if d_beta < 1e-6 and d_H < 1e-6:
    print("\nVERDICT: identical up to rounding - the federated model IS the pooled model.")
else:
    print("\nVERDICT: the models differ - check the run config and preprocessing of both runs.")
