ARG IMAGE=nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

FROM $IMAGE

ARG MINICONDA_VERSION=latest

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y wget git

# Anaconda for BioConda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda create -n metawrap -c conda-forge -c bioconda -c ursky metawrap-mg=1.3.2 && echo "conda activate metawrap" >> ~/.bashrc
RUN conda install -y -c bioconda graftm=0.13.1

RUN git clone -b v0.1.0 https://github.com/np-core/druid && pip install druid/

