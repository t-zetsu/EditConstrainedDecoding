ARG BASE_IMAGE=nvidia/cuda:11.3.1-devel-ubuntu20.04
FROM ${BASE_IMAGE}

ENV TZ=Asia/Tokyo 
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update
RUN apt-get install -y software-properties-common tzdata
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.7-dev python3.7-distutils python3-pip
RUN apt-get -y install git

WORKDIR /work
COPY requirements.txt /work

RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
ENV PYTHONIOENCODING utf-8
ENV CUDA_HOME /usr/local/cuda-11.3

