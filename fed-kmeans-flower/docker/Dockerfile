FROM ubuntu:noble

#Getting updated packages 
RUN apt-get update 
RUN apt-get upgrade -y
RUN apt-get install -y curl 
RUN apt-get install -y git
RUN apt-get install -y zip

#Installing Python 3.11
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3.11-distutils python3.11-venv python3-pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 vim nano -y

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install build dependencies
RUN apt-get install -y build-essential make
RUN apt-get remove -y python3-cryptography || true

RUN mkdir /hereditary
WORKDIR /hereditary
# RUN git clone https://github.com/sara-nl/Hereditary.git
RUN git clone https://github.com/sara-nl/Hereditary.git Hereditary
WORKDIR /hereditary/Hereditary
RUN git checkout sixth_workshop
RUN mkdir -p /hereditary/Hereditary/data/

COPY ALS.zip /hereditary/Hereditary/data
WORKDIR /hereditary/Hereditary/data
RUN unzip ALS.zip && rm ALS.zip

WORKDIR /hereditary/Hereditary/fed-kmeans-flower
RUN pip install -e .
RUN pip install -e ".[viz]"

ENV CLEF_DATA_PATH=/hereditary/Hereditary/data/ALS/
ENV TEST_DATA_PATH=/hereditary/Hereditary/data/test_data/
RUN python3 prep_unique_column_values.py
RUN python generate_test_data.py --num-clients 3 --num-features 2 --samples-per-client 100 --output-dir /hereditary/Hereditary/data/test_data/
RUN python generate_test_data.py --num-clients 3 --num-features 10 --samples-per-client 100 --output-dir /hereditary/Hereditary/data/test_data_10d/
VOLUME /hereditary
VOLUME /root