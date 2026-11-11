# Hereditary workshop 8

**Research use only. Not a medical device.** This software was developed in the HEREDITARY project as a research prototype. It has not been assessed or certified under the EU Medical Device Regulation (EU) 2017/745. It must not be used to diagnose, treat, monitor or make any other decision about individual patients.

Workshop 8 is about federated survival analysis for ALS on the BRAINTEASER / iDPP data. We train a Cox model for time to death across two sites with Flower: Turin, and Lisbon, which stands in for the Answer ALS site. The federated model turns out to be exactly the model you would get by pooling the data, and we use it to predict the survival curve of a held-out patient. The preprocessing (which columns to keep, category levels, means and standard deviations) is itself a federated analytics step. In the main exercise you replace it with the HEREDITARY federated analytics network.

**Docker:** to run this workshop in a container without installing anything, see [DOCKER.md](DOCKER.md) (`./ws8.sh` runs everything).

all workshops use UV to install dependencies (see this page on how to install: https://docs.astral.sh/uv/getting-started/installation/)

## Workshop Contents

### fed-survival-flower
A Flower app (messaging API) for a federated Cox model (WebDISCO), trained on the sites' own data. The folder also has a centralised baseline, a comparison of the federated and pooled models, and a live prediction demo for one patient. See its README for setup, the commands and the exercises.

### SCHEMA.md
The common input schema for Turin and Answer ALS: the columns, their sources and units, and how they are coded. It also explains which model we train first and why, and what leaves a site. This is the document for the discussion with Padua.

## Data
The data is the iDPP 2023 challenge data, partitioned by centre. Download it from the HEREDITARY collaborative platform at https://hereditary.dei.unipd.it, under Files -> Hereditary/Data/FederatedLearning. In case you are signed in, you can directly download the data by clicking [here](http://hereditary.dei.unipd.it/groupoffice/#file/1640). The code expects this layout:

```
ALS_federated/
├── T/                       Turin
│   ├── datasetA/            NIV or death
│   │   ├── train/datasetA_train-{static-vars,visits,outcome,environmental}.csv
│   │   └── test/datasetA_test-{static-vars,visits,outcome,environmental}.csv
│   ├── datasetB/            PEG or death
│   └── datasetC/            death only
├── L/                       Lisbon
└── U/                       a small centre, not a training site by default
```

Extract it into a `data/` folder in the root of this repository, so that you have `data/ALS_federated/T/...`: that is the default `data-path` in `fed-survival-flower/pyproject.toml`, and `data/` is gitignored. If your data is somewhere else, change `data-path`.

## License and intended use

This code is released under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE) for attribution.
It is educational workshop material for federated learning infrastructure in the HEREDITARY project (Horizon Europe, GA 101137074).

Report misuse or security issues to douwe.vanderwal@surf.nl.
