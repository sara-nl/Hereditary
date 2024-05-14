# Federated learning workshop


## Software installation
It would be helpful if you could install the following software before the workshop. If you have any trouble, we can help you during the workshop.

### Using a Docker image
This is the easiest way to get started. You can run the following command to pull the Docker image and run a container. The main benefit of this method is that it hides all the painful (Nvidia and PyTorch) installation details from you. Since this is relatively large, please let me know if you need help downloading it.


### Building your local container
If you prefer to build your container, use the following Dockerfile. This will install all the required software in a container.

```Dockerfile
# Description: Dockerfile for the hereditary project
#Start from the base pytorch image
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:23.12-py3
FROM ${PYTORCH_IMAGE}

#Set the working directory
WORKDIR /mnt/workspace/

RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools

# RUN python3 -m pip install torch torchvision torchaudio
#Install the required packages
#General python packages

RUN python3 -m pip install matplotlib transformers evaluate datasets scikit-learn tqdm pillow pytorch_lightning jupyter notebook
#FL frameworks
##NVFlare
# RUN git clone https://github.com/NVIDIA/NVFlare.git --branch ${NVF_BRANCH} --single-branch NVFlare
# RUN cd NVFlare/
# RUN python3 setup.py install
# RUN cd -
RUN python3 -m pip install nvflare
##Flower
RUN python3 -m pip install flwr flwr_datasets
##Pysyft
# RUN git clone https://github.com/OpenMined/PySyft.git
# RUN cd PySyft
# RUN python3 setup.py install
# RUN python3 setup.py test
# RUN cd -
RUN python3 -m pip install syft
```

You can build the container using the following command

```bash
docker build -t fedai .
```

You can run the container using the following command

```bash
docker run -it --rm --gpus all -v $(pwd):/mnt/workspace/ fedai
```

### Using a virtual environment
If you prefer to install the software on your local machine, follow the instructions below. This method is more flexible and allows you to customize the software installation. It also relies on you setting up all your (GPU) drivers and  libraries correctly.

First, create a virtual environment.

```bash
python3 -m venv fedai
```

Then, activate the virtual environment.

```bash
source fedai/bin/activate
```

Then install the required software

```bash
python3 -m pip install -U pip
python3 -m pip install -U setuptools
python3 -m pip install matplotlib scikit-learn torch torchvision torchaudio transformers evaluate datasets tqdm pillow pytorch_lightning jupyter notebook
python3 -m pip install nvflare
python3 -m pip install flwr flwr_datasets
python3 -m pip install syft
```
