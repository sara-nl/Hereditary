# FLWDocker

Docker with the example for the federated learning example.

Download [data.zip](https://mega.nz/file/05wQlACa#IWL1MDrAQF1MJTBtSZLpn_ziE7e5EPbW81RUz2j4LoQ) into the current directory before starting.

## GPU Build

```bash
docker build -t flworkshop .
```

Creating the container:

```bash
docker container create -i -t --gpus=all --name FLW flworkshop 
```

## CPU Build

```bash
docker build -t flworkshop .
```

Creating the container:

```bash
docker container create -i -t --name FLW flworkshop 
```

## Run the container (bash)

```bash
docker container start  --attach -i FLW
```

## Run example from bash

To run the example from bash for the dataset C with two partitions (which is included in the image building):

```bash
cd /hereditary/Hereditary/xgboost-CLEF
flwr run .
```

To run with other datasets/partitions, the dataset must be divided and the environment variable `CLEF_DATA_PATH` set as explained in [example documentation](https://github.com/sara-nl/Hereditary/tree/third_workshop/xgboost-CLEF).

