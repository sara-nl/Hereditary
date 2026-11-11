"""Live demo: predict the survival curve of one held-out patient.

A site applies the trained model to one of its own test patients. The patient's schema
values are encoded with the preprocessing artifact stored inside the model, and the Cox
model gives S(t | x) = exp(-H0(t) * exp(x . beta)).

    python predict.py                                # first patient of site T's test split
    python predict.py --site L --patient 0x...       # any patient
    python predict.py --model outputs/centralised_model.json

Prints the patient's values, P(event by 12/24/36 months), the predicted median and the
observed outcome, and saves outputs/prediction_<site>_<id>.png.
Research use only: not a medical device.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fed_survival import read_config
from fed_survival.cox import load_model, survival
from fed_survival.features import encode, load_site

HORIZONS = [12.0, 24.0, 36.0]  # months from Time 0
X_MAX = 120.0  # plot 10 years; few patients are at risk later

config = read_config()
output = Path(config["output-dir"])
parser = argparse.ArgumentParser(description="Predict the survival curve of one patient.")
parser.add_argument("--site", default="T")
parser.add_argument("--split", default="test")
parser.add_argument("--patient", help="PatientID (default: the first patient in the file)")
parser.add_argument("--model", default=str(output / "model.json"))
args = parser.parse_args()

# 1. The model and the site's own data (same dataset, event and window as in training)
model = load_model(args.model)
event_name = model["event"]
table, time, event = load_site(
    config["data-path"], args.site, model["dataset"], args.split, model["window"], event_name
)
patient = args.patient or table.index[0]
if patient not in table.index:
    raise SystemExit(f"{patient} is not in the {args.split} split of site {args.site}")
i = table.index.get_loc(patient)
row = table.iloc[[i]]

print(f"Patient {patient} (site {args.site}, {args.split} split)")
dropped = model["preprocessing"].get("dropped", {})
for col, value in row.iloc[0].items():
    note = (
        "not used: dropped by the federated preprocessing"
        if col in dropped
        else "missing: imputed with the federated mean" if pd.isna(value) else ""
    )
    shown = (
        "-"
        if pd.isna(value)
        else value if not isinstance(value, float) else f"{value:.0f}" if value.is_integer() else f"{value:.2f}"
    )
    print(f"  {col:<18}{shown:>10}   {note}")

# 2. Risk score and survival curve, from the model's own preprocessing and coefficients
x = encode(row, model["preprocessing"])
lp = float((x @ np.asarray(model["beta"]))[0])
t = np.concatenate([[0.0], model["baseline"]["time"]])  # the curve's step points
S = survival(model, x, t)[0]
S_h = survival(model, x, HORIZONS)[0]
print(f"\nModel: {args.model}\n  ({model['trained_with']})")
print(f"Risk score x.beta = {lp:+.3f}: hazard {np.exp(lp):.2f} x that of a patient " f"with average values")
for h, s_h in zip(HORIZONS, S_h):
    print(f"  P({event_name} by {h:.0f} months) = {1 - s_h:.2f}")
below = np.nonzero(S <= 0.5)[0]
print(
    f"  Predicted median time to {event_name}: "
    + (f"{t[below[0]]:.1f} months" if below.size else f"not reached by {t[-1]:.0f} months")
)
observed = f"{event_name if event[i] else 'censored'} at {time[i]:.1f} months"
print(f"  Observed: {observed}")

# 3. The figure: this patient, the centralised model (should overlap) and the population
what = "survival" if event_name == "DEATH" else f"{event_name}-free survival"
kind = model["trained_with"].split(":")[0]  # "federated" or "centralised"
fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
ax.axvspan(0, model["window"], color="#f0efec", lw=0, label=f"feature window (months 0-{model['window']:g})")
km = model.get("kaplan_meier")
if km:
    ax.step(
        np.concatenate([[0.0], km["time"]]),
        np.concatenate([[1.0], km["survival"]]),
        where="post",
        color="#8a8984",
        lw=1.5,
        label=f"training population ({kind} Kaplan-Meier)",
    )
ax.step(t, S, where="post", color="#2a78d6", lw=2.2, label=f"this patient: {kind} Cox model")

centralised = output / "centralised_model.json"
if centralised.exists() and Path(args.model).resolve() != centralised.resolve():
    cen = load_model(centralised)
    if all(cen[k] == model[k] for k in ("dataset", "event", "window", "time_step", "features")):
        S_cen = survival(cen, encode(row, cen["preprocessing"]), t)[0]
        ax.step(
            t, S_cen, where="post", color="#eb6834", lw=1.5, ls=(0, (4, 3)), label="this patient: centralised Cox model"
        )

for h, s_h in zip(HORIZONS, S_h):
    ax.plot(h, s_h, "o", color="#2a78d6", ms=6, zorder=5)
    ax.annotate(
        f"P({event_name.replace('DEATH', 'death')} by {h:.0f} m) = {1 - s_h:.2f}",
        (h, s_h),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.2", "fc": "#fcfcfb", "ec": "none", "alpha": 0.85},
    )
ax.plot(
    time[i],
    survival(model, x, [time[i]])[0][0],
    "X" if event[i] else "o",
    ms=10,
    mfc="#0b0b0b" if event[i] else "#fcfcfb",
    mec="#0b0b0b",
    ls="none",
    zorder=6,
    label=f"observed: {observed}",
)

ax.set(
    xlim=(0, min(X_MAX, t[-1])),
    ylim=(0, 1.02),
    ylabel=f"{what} probability",
    xlabel="months since Time 0 (first ALSFRS-R)",
)
ax.set_title(
    f"Predicted {what} for patient {patient[:8]}... " f"(site {args.site}, {args.split} split)", fontsize=11, loc="left"
)
ax.grid(color="#e5e4e0", lw=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper right", fontsize=9, frameon=False)
fig.text(0.01, 0.01, "Research use only. Not a medical device. HEREDITARY workshop 8.", fontsize=8, color="#52514e")
fig.tight_layout(rect=(0, 0.03, 1, 1))
suffix = "" if kind == "federated" else f"_{kind}"
path = output / f"prediction_{args.site}_{patient[:8]}{suffix}.png"
fig.savefig(path, dpi=150)
print(f"Saved {path}")
