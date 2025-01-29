# FETS-swin-unet: 
During this part of the workshop we will be using the [Swin UNETR](https://github.com/LeonidAlekseev/Swin-UNETR) model to train on the FETS dataset. The code is based on a [tutorial](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb) from Monai.

## Install dependencies and project

```bash
# Create a virtual environment if you wish
python -m venv FETS_venv
source FETS_venv/bin/activate

# Install the dependencies
pip install -e .
```

## Run the experiment
Before we can run the experiment, we will need to set the environment variable indicating the path to the FETS data.
```bash
export FETS_DATA_DIR=/path/to/FETS/data
# e.g.
export FETS_DATA_DIR=/Users/d0uwe/hereditary_data/FETS_data/MICCAI_FeTS2022_TrainingData/
```
Keep in mind that the SuperNodes in this experiment will need to have access to a GPU with at least 20GB of memory.

Start the superlink:
```bash
flower-superlink \
    --ssl-ca-certfile certificates/ca.crt \
    --ssl-certfile certificates/server.pem \
    --ssl-keyfile certificates/server.key \
    --auth-list-public-keys keys/client_public_keys.csv \
    --auth-superlink-private-key keys/server_credentials \
    --auth-superlink-public-key keys/server_credentials.pub
```

Start the supernodes:
```bash
flower-supernode \
    --root-certificates certificates/ca.crt \
    --superlink IP_HERE:9092 \
    --clientappio-api-address 0.0.0.0:9094 \
    --node-config="partition-id=1" \
    --auth-supernode-private-key keys/client_credentials_1 \
    --auth-supernode-public-key keys/client_credentials_1.pub
```
```bash
flower-supernode \
    --root-certificates certificates/ca.crt \
    --superlink IP_HERE:9092 \
    --clientappio-api-address 0.0.0.0:9095 \
    --node-config="partition-id=2" \
    --auth-supernode-private-key keys/client_credentials_2 \
    --auth-supernode-public-key keys/client_credentials_2.pub
```

Then, if the default federation is set to `surfsuperlink`, you can start the experiment by running:
```bash
flwr run .
```
