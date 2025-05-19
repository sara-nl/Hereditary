# FLWDocker

Docker with the example for the federated learning example.

Download [data.zip](https://mega.nz/file/05wQlACa#IWL1MDrAQF1MJTBtSZLpn_ziE7e5EPbW81RUz2j4LoQ) into the current directory before starting.

## GPU Build

```bash
docker build -t flworkshop_4 .
```

Creating the container:

```bash
docker container create -i -t --gpus=all --name FLW flworkshop_4 
```

## CPU Build

```bash
docker build -t flworkshop_4 .
```

Creating the container:

```bash
docker container create -i -t --name FLW flworkshop_4 
```

## Run the container (bash)

```bash
docker container start  --attach -i FLW
```

When starting the container to run a flwr superlink (the server), then we need to make sure the appropriate ports are opened. 
```bash
docker container create -i -t --publish 9091:9091 --publish 9092:9092 --publish 9093:9093 --name FLW flworkshop_4
```

## Run example from bash

To run the example from bash for the dataset C with two partitions (which is included in the image building):

```bash
cd /hereditary/Hereditary/CLEF_vertical
flwr run .
```

To run with other datasets/partitions, the dataset must be divided and the environment variable `CLEF_DATA_PATH` set as explained in [example documentation](https://github.com/sara-nl/Hereditary/tree/third_workshop/xgboost-CLEF).
