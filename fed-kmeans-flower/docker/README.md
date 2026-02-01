# FLWDocker

Docker with the example for the federated learning example.

First sign into the collaborative platform [here](https://hereditary.dei.unipd.it/groupoffice/) and then download this [file](http://hereditary.dei.unipd.it/groupoffice/index.php?r=files/file/download&id=1640&security_token=HM8UXL67wVRCWfmhZdk5) in the same browser and place it into the current directory before starting.

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

When starting the container to run a flwr superlink (the server), then we need to make sure the appropriate ports are opened. 
```bash
docker container create -i -t --publish 9091:9091 --publish 9092:9092 --publish 9093:9093 --name FLW flworkshop
```

## Run example from bash

To run the example using the test data, use the following command inside the Docker image:

```bash
cd /hereditary/Hereditary/fed-kmeans-flower
flwr run .
```

To run the experiment with CLEF data, change the datasource from `test_data` to `ALS`. 

