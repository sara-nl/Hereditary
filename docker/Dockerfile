FROM ubuntu:oracular

#Getting updated packages 
RUN apt-get update 
RUN apt-get upgrade -y
RUN apt-get install -y curl 
RUN apt-get install -y git
RUN apt-get install -y zip

#COPY pyproject.toml /root
#Getting Miniconda 
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
RUN rm -f Miniconda3-latest-Linux-x86_64.sh

#Conda init
ENV PATH="/root/miniconda3/bin":$PATH
RUN conda init bash
RUN echo 'conda activate base' >> ~/.bashrc

#Getting python 
RUN conda install -y python=3.11 make conda-forge::cxx-compiler conda-forge::jupyterlab conda-forge::seaborn conda-forge::pandas conda-forge::scikit-learn
RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
#RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch

RUN mkdir /hereditary
WORKDIR /hereditary

RUN git clone https://github.com/sara-nl/Hereditary.git
WORKDIR /hereditary/Hereditary
RUN git checkout third_workshop

COPY data.zip /hereditary/Hereditary/xgboost-CLEF
 
WORKDIR /hereditary/Hereditary/xgboost-CLEF
RUN pip install -e .
RUN unzip data.zip
RUN rm data.zip
RUN python partition_clef.py --data_path data/CLEF/BRAINTEASER_ALS_MS_datasets/retrospective/ALS/CSV/data/datasetC/ --partitions 2

ENV CLEF_DATA_PATH=/hereditary/Hereditary/xgboost-CLEF/data/CLEF/BRAINTEASER_ALS_MS_datasets/retrospective/ALS/CSV/data/datasetC/


WORKDIR /hereditary
VOLUME /hereditary
VOLUME /root

