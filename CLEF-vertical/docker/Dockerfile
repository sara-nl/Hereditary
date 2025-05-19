FROM ubuntu:noble

#Getting updated packages 
RUN apt-get update 
RUN apt-get upgrade -y
RUN apt-get install -y curl 
RUN apt-get install -y git
RUN apt-get install -y zip

#COPY pyproject.toml /root
#Installing Python 3.11
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.11 python3.11-dev python3.11-distutils python3.11-venv
RUN apt-get install -y python3-pip

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install build dependencies
RUN apt-get install -y build-essential make
RUN apt-get remove -y python3-cryptography || true

RUN mkdir /hereditary
WORKDIR /hereditary

RUN git clone https://github.com/sara-nl/Hereditary.git
WORKDIR /hereditary/Hereditary
RUN git checkout fourth_workshop

COPY data.zip /hereditary/Hereditary/CLEF_vertical

WORKDIR /hereditary/Hereditary/CLEF_vertical

RUN pip install -e .
RUN unzip data.zip
RUN rm data.zip

ENV CLEF_DATA_PATH=/hereditary/Hereditary/CLEF_vertical/data/CLEF/BRAINTEASER_ALS_MS_datasets/retrospective/ALS/CSV/data/datasetC/

WORKDIR /hereditary
VOLUME /hereditary
VOLUME /root
