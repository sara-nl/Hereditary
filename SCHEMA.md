# Input schema and first model: ALS survival

This document proposes the input schema every site maps its data to, and the first model to train on
it, for the University of Turin, the Answer ALS mapping and the discussion in Padua. It is based on the
BRAINTEASER / iDPP 2023 ALS data split by centre (`T` Turin, `L` Lisbon, `U` a small unknown centre);
Lisbon stands in for the Answer ALS site until that data is available.

**In short.** Endpoint: time to death (dataset C) from the first ALSFRS-R questionnaire, for patients
still alive and in follow-up at month 6. Model: Cox proportional hazards fitted with WebDISCO on a
public monthly time grid, so the federated fit is the pooled fit. Input: the 12 schema columns of
section 3, built locally at each site. Preprocessing is decided by one round of federated analytics
(section 5). First request to Turin and Answer ALS: the checklist in section 8. We do not need data yet.

## 1. Which model first, and why

**Start with a Cox proportional-hazards model trained with WebDISCO** (Lu et al., JAMIA 2015):
Newton-Raphson on the Breslow partial likelihood, where every site contribution is a sum, so the
server obtains exactly the gradient and information matrix of the pooled data. All sites use the
same public time grid (whole months 1 to 600, each time rounded up), so no site reveals its exact
event times.

- **Exact.** Federated coefficients equal those of a pooled fit on the same month-rounded times; the
  workshop checks them against an independent scikit-survival fit (largest difference 4e-14), which
  is a stronger claim than two curves that look similar.
- **Interpretable.** One hazard ratio per feature (per standard deviation, as features are z-scored).
- **A survival curve per patient.** S(t | x) = exp(-H0(t) exp(x·β)) gives the event probability at any
  horizon (the demo uses 12, 24 and 36 months).
- **Accurate enough.** For death with month-6 data, the organisers' published Cox baseline scored 0.706.
  The best iDPP 2022 submissions reached just over 0.725, and the published band is 0.70 to 0.75.
  The federated model (T+L training data, the 11 features that survive preprocessing, monthly grid)
  reaches a test Harrell's C of **0.702 (T)** and **0.751 (L)** (0.7024, 0.7515; versus exact times the
  grid moves coefficients by at most 0.008 and the C-index by at most 0.0002). Our internal
  271-feature model reaches 0.715 and 0.749, so the small schema gives up very little.
- **One member of NMES** ("Naive Multiple Event Survival", Pancotti et al., iDPP 2022): one standard
  survival model per event, other events treated as censoring, base estimators from scikit-survival.
  The same code trains the per-event Cox model for NIV (`dataset = "datasetA"`, `event = "NIV"`: test
  C 0.672 T, 0.726 L) or PEG (`datasetB`, `"PEG"`).

| candidate | how it federates | equals pooled? | role |
|---|---|---|---|
| **Cox PH (Breslow), WebDISCO** | the server sums site statistics and takes the Newton step | **yes** | **start here** |
| Ridge-penalised Cox | same code, `l2 > 0` (lasso or elastic-net would need another solver) | yes | option |
| Random survival forest | local trees; the server selects and combines them (FedSurF-style) | no, approximate | next step |
| XGBoost AFT (`survival:aft`) | Flower bagging (trees concatenated); the AFT loss is per-patient | no, approximate | next step |
| Deep Survival Machines (auton-survival) | no federated implementation exists | n/a | not a starting point |
| Fine-Gray (competing risks) | needs a global censoring distribution; no federated implementation found | n/a | only if needed |

**Why not Deep Survival Machines first.** `auton-survival` (no commit since 2023-10-16) pins `torch<2`,
`numpy<2`, `pandas<2` and `scikit-survival` 0.21, so it cannot share an environment with scikit-survival
0.28. No federated implementation exists, and Pancotti et al. found no clear advantage over NMES on this
data. If wanted, run it centrally as a reference.

**Competing events.** Dataset C records death only: no competing event. For NIV (A) or PEG (B), NMES
treats death as censoring, which gives a *cause-specific* hazard; one minus its "survival" is not the
probability of NIV by time t (the cumulative incidence), since patients who die can no longer get NIV.
Fine-Gray models the cumulative incidence directly but needs the censoring distribution of all sites
together: one extra federated-analytics step (the counts of a federated Kaplan-Meier curve).

## 2. Time conventions (agree these first)

| concept | convention | check on the data |
|---|---|---|
| Time 0 | date of the **first ALSFRS-R questionnaire** | every patient has exactly one ALSFRS-R at `date_alsfrs_r = 0` |
| unit | **months = days / 30** | every time value × 30 is a whole number; × 30.44 is not |
| dates in static-vars | `onsetDate`, `diagnosisDate` are months **relative to Time 0** (negative = before) | T: onset -165 to +14, median -9.9 |
| landmark | only patients with more than 6 months of follow-up after Time 0 are included (event or censored) | minimum `Time`: T 6.03, L 6.07, U 13.4 |
| `Time` in outcome | measured **from Time 0**, not from the landmark, so `Time > 6` | as above |
| feature window | observations with date ≤ 6.0 months; earlier (pre-baseline) observations count | visit dates end at 6.0; spirometry from month -89 |
| censoring | `Occured = 0`, `Type = NONE`, at last known contact | |
| model time grid | public, set in the run config: months 1, 2, ..., 600 (`time-step`, `time-max`); the Cox code rounds each `Time` **up** to its grid month; the C-index uses the exact times | deaths fall in months 7 to 196; latest `Time` 222.1 (L) |

A site must not restart the clock at the landmark, and must exclude patients with 6 months of
follow-up or less; otherwise the site populations are not comparable. Sites deliver `Time`
unrounded: the rounding to the grid happens inside the model code, identically everywhere.

## 3. Input tables and the schema columns

Three tables per site, joined on a pseudonymous `PatientID` that never leaves it. The model reads only:

| table | grain | columns used |
|---|---|---|
| `outcome` | one row per patient | `PatientID`, `Occured` (0/1), `Type` (`DEATH`/`NONE`; `NIV`/`PEG` in A/B), `Time` (months from Time 0) |
| `static-vars` | one row per patient | `sex`, `age_onset`, `onsetDate`, `diagnosisDate`, `height`, `weight`, `onset_bulbar`, `C9orf72` |
| `visits` | one row per assessment | `date_alsfrs_r`, `alsfrs_r_tot_score`, `date_spiro`, `fvcValue` |

The event is `(Occured == 1) & (Type == event)` (default `"DEATH"`); every other outcome is censored.

From these, each site builds the **12 schema columns** (names are fixed; W = 6.0 months):

| column | type, unit | derivation |
|---|---|---|
| `age_onset` | numeric, years | static `age_onset` |
| `sex` | categorical {`Female`, `Male`} | static `sex` |
| `onset_bulbar` | numeric 0/1 | static `onset_bulbar` (`True`/`False`, possibly as strings) |
| `diagnostic_delay` | numeric, months | `diagnosisDate - onsetDate` (both already months relative to Time 0); **missing if `diagnosisDate` > W**: a diagnosis after the landmark is future information |
| `time_since_onset` | numeric, months | `-onsetDate` (disease duration at Time 0) |
| `bmi` | numeric, kg/m² | `weight / height**2` (height in metres) |
| `C9orf72` | categorical {`expansion`, `normal`} | static `C9orf72` |
| `alsfrs_r_first` | numeric, points 0-48 | first `alsfrs_r_tot_score` with `date_alsfrs_r <= W` |
| `alsfrs_r_last` | numeric, points 0-48 | last `alsfrs_r_tot_score` with `date_alsfrs_r <= W` |
| `alsfrs_r_slope` | numeric, points/month | `(last - first) / (t_last - t_first)`; missing if the patient has fewer than 2 questionnaires (at different dates) in the window |
| `fvc_last` | numeric, % predicted | last `fvcValue` with `date_spiro <= W` (pre-baseline values count) |
| `fvc_n` | numeric, count | number of `fvcValue` measurements with `date_spiro <= W` (0 if none) |

Never use `alive` (section 6). The environmental file is not used: few patients are linked to it, and
it did not help in iDPP 2023. `fvc_n = 0` also works as a missing indicator for `fvc_last`.

## 4. Coverage per site (dataset C, training split)

Percentage of patients with the column observed, as `features.load_site` computes it.

| | T (Turin) | L (Lisbon) | U |
|---|---|---|---|
| patients (train / test) | 1199 / 362 | 503 / 121 | 14 / 3 |
| deaths / censored (train) | 1049 / 150 | 425 / 78 | 12 / 2 |
| deaths / censored (test) | 313 / 49 | 101 / 20 | 3 / 0 |
| `age_onset`, `sex`, `onset_bulbar`, `time_since_onset`, `alsfrs_r_first`, `alsfrs_r_last`, `fvc_n` | 100 | 100 | 100 |
| `diagnostic_delay` | 99.7 | 100 | 100 |
| `bmi` | 92.7 | 89.7 | 92.9 |
| `C9orf72` | 73.6 | **44.3** | **0.0** |
| `alsfrs_r_slope` (≥ 2 questionnaires) | 79.6 | 90.5 | 85.7 |
| `fvc_last` (= patients with `fvc_n` ≥ 1) | 51.8 | 88.1 | 85.7 |

U is too small to be a training site. Turin quirks: one patient has `onsetDate` after Time 0 (negative
`time_since_onset`); four were diagnosed after month 6 (`diagnosisDate` 6.6 to 17.9), not yet known at the
landmark, so their `diagnostic_delay` is missing and gets the mean (C-index unchanged to three decimals).

## 5. What a site does locally, and what is agreed federatedly

| step | where | how |
|---|---|---|
| the schema: names, units, derivations | on paper | this document |
| build the schema table from local tables | site | section 3, 6-month window |
| which columns to keep | federated (preprocessing) | keep a column only if **every** site observes it for ≥ 50% of patients (`min-coverage = 0.5`) |
| category levels | federated | sorted union of the observed levels; the first level is the reference; one 0/1 dummy `col=level` per other level (a column with one level is dropped) |
| means and standard deviations | federated | from summed counts, sums and sums of squares (population std); a constant column is dropped |
| encode: dummies, impute missing with the mean, z-score | site | with the agreed artifact |
| time grid | on paper (run config) | public whole months 1 to 600, known in advance; no site sends event times |
| coefficients, baseline hazard | federated (training) | the server sums the sites' round A and B replies and takes the Newton step |
| test C-index | site | own test split only; no "global" C-index |

Only one small JSON file (the *preprocessing artifact*, format `fed-survival-preprocessing/1`) passes
from preprocessing to training, which reads only its ordered features, means and standard deviations
(it also records the levels, dropped columns with a reason, and patients per site). The workshop
computes it over Flower; in the exercise the HEREDITARY federated analytics network produces the same
file (`preprocessing = "file"`, `preprocessing-file = ...`).

**Example: `C9orf72` is dropped.** It is observed for 73.6% of Turin but only 44.3% of Lisbon patients.
A pooled rule would have kept it (1106 of 1702 = 65%) and Lisbon's values would then be mostly imputed;
only per-site coverage shows this. The model then uses 11 features: `age_onset`, `sex=Male`,
`onset_bulbar`, `diagnostic_delay`, `time_since_onset`, `bmi`, `alsfrs_r_first`, `alsfrs_r_last`,
`alsfrs_r_slope`, `fvc_last`, `fvc_n`. The same rule applies to Answer ALS.

Global (not per-site) means and standard deviations are deliberate: with site-specific scaling
the same patient is encoded differently at each site, and federated would no longer equal pooled.

## 6. Mapping pitfalls

| item | BRAINTEASER convention | what goes wrong |
|---|---|---|
| `onsetDate`, `diagnosisDate` | floats: months relative to Time 0 | parsing them as dates gives `diagnostic_delay = 0` for everyone, and the std-0 rule then drops the column without warning |
| months | days / 30 | days / 30.44 shifts every time by 1.5%; small, but all sites must use the same divisor |
| `height` | **metres** (Turin 1.36-1.98) | height in cm makes the BMI 10 000 times too small |
| `weight` | kg, at baseline (not `weight_before_onset`) | |
| `fvcValue` | **% of predicted** (Turin 9-154, Lisbon 16-195) | values in litres, or sitting and supine mixed. Converting litres to % predicted needs a stated reference equation (GLI-2012 and older ECSC equations give different percentages) |
| ALSFRS-R | revised 12-item scale, 0-48, 48 = normal | the original 10-item ALSFRS (0-40) is not comparable |
| `alive` | vital status at the **end** of follow-up | it is the outcome. In Turin training data, all 1013 patients with `alive = False` died. It must never be a feature, and it is unknown at prediction time |
| genetics | `C9orf72` ∈ {`expansion`, `normal`}; empty = not tested or not recorded | "not tested" coded as `normal`. The coding already differs between centres: Turin stores `SOD1`/`TARDBP`/`FUS` as True/False, Lisbon mixes `'False'` strings with free-text variant descriptions |
| `onset_bulbar` | boolean | arrives as the strings `"True"`/`"False"` |

## 7. What leaves a site

No row-level data, no `PatientID` and no exact event or censoring time leaves a site. These
aggregates do (dataset C, T+L, p = 11 features, K = 600 grid months); the server sends β, the pooled
`c_k = d_k / S0_k` and the preprocessing artifact:

| step (message) | sent by each site | size |
|---|---|---|
| preprocessing, once (`query.summary`) | patient count; per numeric column: number observed, sum, sum of squares; per categorical column: count per level | 35 numbers |
| round A (`train.risk_sets`), at β = 0 and after every Newton step: 7 times | per grid month: event count `d_k`, risk-set sums `S0_k` (1 number) and `S1_k` (p numbers); one p-vector: the sum of x over its events | 7 811 numbers |
| round B (`train.hessian`), each iteration: 6 times | one p × p matrix | 121 numbers |
| evaluation, once (`evaluate.cindex`) | test C-index and number of test patients | 2 numbers |

**Disclosure caveats.** Round A is more than a gradient; the WebDISCO authors state that the method
implements no differential privacy.

- Round A at β = 0 (where `S0_k` is the at-risk count) reveals each site's event and at-risk counts
  per month: its Kaplan-Meier curve at monthly resolution, and the month of its last follow-up.
- **Differencing.** `S0_k - S0_{k+1}` and `S1_k - S1_{k+1}` are sums over the patients whose time falls
  in grid month k. If that is a single patient, `ΔS1 / ΔS0` is that patient's encoded covariate vector.
  The artifact's means and standard deviations turn it back into raw values, and `d_k` tells whether
  the patient died. With the exact event times as grid this would apply to **460 of 1199 Turin (38%)
  and 346 of 503 Lisbon (69%)** training patients, from the first round; on the public monthly grid
  to **38 Turin (3.2%) and 30 Lisbon (6.0%)** patients.
- **The tail.** At the last five months with a death (141, 149, 156, 158, 196) Lisbon has 4, 3, 2, 2 and
  1 patients at risk, and its sums continue after the last death (the grid runs to month 600). In all,
  T has 33 and L 89 grid months with 1 to 4 patients at risk.

**Mitigations** (combine them):

1. **A public time grid** (done in the code). A coarser one isolates fewer patients: `time-step = 3.0`
   leaves 13 (T) and 12 (L), at a test C-index of 0.7015 (T) / 0.7515 (L), with coefficients moving by
   up to 0.04 from the exact-time fit. Federated still equals pooled on the rounded times.
2. **Minimum-count rules.** Send sums only at months with a death (still exact: months without one merge
   into the previous interval), which leaves 17 (T) and 26 (L). Merge further until each site has at
   least k patients leaving every interval, and end the grid (administrative censoring at a public
   horizon) before any site's risk set falls below k. Apply the same minimum cell size to the
   preprocessing counts: a level count of 1 or 2 at a small site can single out a patient.
3. **Secure aggregation** (workshop 5) hides which site contributed what, but not the pooled sums:
   on the monthly grid they still isolate 35 patients (450 on the exact grid).

The workshop code uses the public monthly grid but no minimum-count rules and no secure aggregation,
to keep it small. That is acceptable on the public iDPP data, not on live clinical data.

## 8. Questions for Turin and for the Answer ALS mapping

First step: a column-level availability matrix (available / partial / unavailable, with units and
coding) for the columns of section 3, plus answers to:

- [ ] **Dated vital status.** Is a death date, or a date of last contact, available? Without it there is
      no death endpoint. For Answer ALS this is the blocker.
- [ ] **Time origin and landmark.** Can Time 0 be the first ALSFRS-R questionnaire (days / 30), and can
      patients with at most 6 months of follow-up be excluded with time still measured (unrounded) from
      Time 0? For Answer ALS: when is the first ALSFRS-R relative to enrolment?
- [ ] **Onset and diagnosis dates**, so that `diagnostic_delay` and `time_since_onset` can be derived.
- [ ] **FVC:** % predicted or litres, which reference equation, and sitting or supine.
- [ ] **ALSFRS-R:** the revised 12-item scale, or the original ALSFRS?
- [ ] **Height and weight units**, and which weight is recorded.
- [ ] **Genetics coding**, in particular how "not tested" is distinguished from "negative".
- [ ] **Counts after landmarking:** patients, deaths and censored patients (compare section 4).

**Optional extra columns for later** (*not used by the workshop model*; only if cheap to provide):

- onset site detail (`onset_limbs`, `onset_axial`, `onset_generalized`, `onset_limb_type`);
  `weight_before_onset` (weight loss); `ALS_familiar_history`; `smoking`
- motor neuron predominance (`prevalentUMN`, `prevalentLMN`, `mixedMN`); `SOD1`, `TARDBP`, `FUS` mutations
- ALSFRS-R sub-scores (bulbar, motor, respiratory) and items `q1`-`q12`; the challenge's own slope,
  `(48 - ALSFRS-R at Time 0) / (Time 0 - onset)`
- comorbidity flags (hypertension, diabetes, dyslipidaemia, thyroid disorder, autoimmune disease, stroke,
  cardiac disease, primary neoplasm); baseline labs (CK, albumin, creatinine, cholesterol,
  triglycerides) in one agreed unit
