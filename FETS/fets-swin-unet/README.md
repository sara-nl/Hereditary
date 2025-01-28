# FETS-swin-unet: 
During this part of the workshop we will be using the [Swin UNETR](https://github.com/LeonidAlekseev/Swin-UNETR) model to train on the FETS dataset. The code is based on a [tutorial](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb) from Monai.

## Install dependencies and project

```bash
pip install -e .
```

## Run the experiment
Before we can run the experiment, we will need to set the environment variable indicating the path to the FETS data.
```bash
export FETS_DATA_DIR=/path/to/FETS/data
```

Start the supernodes:
```bash
flower-supernode \
    --root-certificates certificates/ca.crt \
    --superlink hereditary.soil.surf.nl:9092 \
    --clientappio-api-address 0.0.0.0:9094 \
    --node-config="partition-id=1" \
    --auth-supernode-private-key keys/client_credentials_1 \
    --auth-supernode-public-key keys/client_credentials_1.pub
```
```bash
flower-supernode \
    --root-certificates certificates/ca.crt \
    --superlink hereditary.soil.surf.nl:9092 \
    --clientappio-api-address 0.0.0.0:9095 \
    --node-config="partition-id=2" \
    --auth-supernode-private-key keys/client_credentials_2 \
    --auth-supernode-public-key keys/client_credentials_2.pub
```

