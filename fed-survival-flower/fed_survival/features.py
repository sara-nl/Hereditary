"""Site-local features: raw BRAINTEASER CSVs -> the common input schema -> model matrix.

Everything in this file runs inside a site and returns one row per patient, so none of it
is ever sent anywhere. Two steps:

    load_site()  reads one site's CSVs and builds the schema table (see SCHEMA.md)
    encode()     turns a schema table into the float64 matrix the Cox model uses, with the
                 feature list, means and stds of the preprocessing artifact
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# The common input schema: the 12 columns every site builds, and their kind.
# SCHEMA.md documents each column (source, unit, coding) for Turin and Answer ALS.
SCHEMA = {
    "age_onset": "numeric",  # years
    "sex": "categorical",  # Female / Male
    "onset_bulbar": "numeric",  # 0 / 1
    "diagnostic_delay": "numeric",  # months from onset to diagnosis
    "time_since_onset": "numeric",  # months from onset to Time 0
    "bmi": "numeric",  # kg / m2
    "C9orf72": "categorical",  # expansion / normal
    "alsfrs_r_first": "numeric",  # ALSFRS-R total score (0-48), first in the window
    "alsfrs_r_last": "numeric",  # ... and last in the window
    "alsfrs_r_slope": "numeric",  # points / month between those two
    "fvc_last": "numeric",  # FVC, % predicted, last in the window
    "fvc_n": "numeric",  # number of FVC measurements in the window
}

ARTIFACT_FORMAT = "fed-survival-preprocessing/1"


def load_site(data_path, site, dataset, split, window, event):
    """Read one site's split. Returns (table, time, event).

    table: DataFrame indexed by PatientID with exactly the SCHEMA columns (NaN = missing)
    time:  float64 array, months from Time 0 to the event or censoring
    event: bool array, True if the chosen event (e.g. "DEATH") occurred; any other outcome
           (no event, or a competing event such as NIV) counts as censored
    """
    base = Path(data_path) / site / dataset / split / f"{dataset}_{split}"
    static = pd.read_csv(f"{base}-static-vars.csv", low_memory=False).set_index("PatientID")
    visits = pd.read_csv(f"{base}-visits.csv")
    outcome = pd.read_csv(f"{base}-outcome.csv").set_index("PatientID").loc[static.index]

    # 1. Static variables. The BRAINTEASER dates are already floats: months relative to Time 0.
    table = pd.DataFrame(index=static.index)
    table["age_onset"] = static["age_onset"]
    table["sex"] = static["sex"]
    table["onset_bulbar"] = static["onset_bulbar"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    diagnosed = static["diagnosisDate"].where(static["diagnosisDate"] <= window)  # else unknown yet
    table["diagnostic_delay"] = diagnosed - static["onsetDate"]
    table["time_since_onset"] = -static["onsetDate"]
    table["bmi"] = static["weight"] / static["height"] ** 2  # height is in metres
    table["C9orf72"] = static["C9orf72"]

    # 2. ALSFRS-R questionnaires up to the end of the window (month 6)
    q = visits[(visits["date_alsfrs_r"] <= window) & visits["alsfrs_r_tot_score"].notna()]
    q = q.sort_values("date_alsfrs_r", kind="stable").groupby("PatientID")
    first, last = q["alsfrs_r_tot_score"].first(), q["alsfrs_r_tot_score"].last()
    t_first, t_last = q["date_alsfrs_r"].first(), q["date_alsfrs_r"].last()
    table["alsfrs_r_first"] = first
    table["alsfrs_r_last"] = last
    table["alsfrs_r_slope"] = ((last - first) / (t_last - t_first)).where(t_last > t_first)

    # 3. Spirometry up to the end of the window (pre-baseline measurements count too)
    s = visits[(visits["date_spiro"] <= window) & visits["fvcValue"].notna()]
    s = s.sort_values("date_spiro", kind="stable").groupby("PatientID")
    table["fvc_last"] = s["fvcValue"].last()
    table["fvc_n"] = s.size().reindex(table.index, fill_value=0).astype(float)

    # 4. Outcome for the chosen event
    time = outcome["Time"].to_numpy(np.float64)
    occurred = ((outcome["Occured"] == 1) & (outcome["Type"] == event)).to_numpy(bool)
    return table[list(SCHEMA)], time, occurred


def encode(table, artifact):
    """Schema table -> float64 matrix (patients x artifact features).

    A feature "col=level" is a 0/1 dummy (NaN when col is missing; values are compared as
    strings, as site_summary counts them); every other feature is a numeric column.
    Missing values get the artifact mean, then everything is z-scored with the artifact
    mean and std, so all sites encode identically.
    """
    columns = []
    for name in artifact["features"]:
        if "=" in name:
            col, level = name.split("=", 1)
            x = (table[col].astype(str) == level).astype(float).where(table[col].notna())
        else:
            x = table[name].astype(float)
        columns.append(x.to_numpy(np.float64))
    X = np.column_stack(columns)
    mean = np.asarray(artifact["mean"], np.float64)
    std = np.asarray(artifact["std"], np.float64)
    X = np.where(np.isnan(X), mean, X)
    return (X - mean) / std


def load_artifact(path):
    """Read a preprocessing artifact (from stage 1, or from the FA network)."""
    with open(path) as f:
        artifact = json.load(f)
    if artifact.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"{path}: not a {ARTIFACT_FORMAT} file")
    if not len(artifact["features"]) == len(artifact["mean"]) == len(artifact["std"]):
        raise ValueError(f"{path}: features, mean and std must have the same length")
    return artifact
