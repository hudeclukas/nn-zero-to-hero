FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    libgl1 \
    libglib2.0-0

COPY ./requirements.txt /nn-repo/requirements.txt

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /nn-repo/requirements.txt

WORKDIR /nn-repo