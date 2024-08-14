# Note: You can use any Debian/Ubuntu based image you want.
# nvcr.io/nvidia/pytorch:23.05-py3 - CUDA 12.1 / Python 3.10 / PyTorch 2.0.0
#  - https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-05.html
# nvcr.io/nvidia/pytorch:23.11-py3 - CUDA 12.3 / Python 3.10 / PyTorch 2.2.0
#  - https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-11.html
# nvcr.io/nvidia/pytorch:24.05-py3 - CUDA 12.4.1 / Python 3.10 / PyTorch 2.4.0
#  - https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-05.html
FROM nvcr.io/nvidia/pytorch:24.05-py3 as base

ENV DEBIAN_FRONTEND=noninteractive
# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && apt-get -y install curl nano jq git wget watch

FROM base AS dev
COPY configs/accelerate.yaml /root/.cache/huggingface/accelerate/
WORKDIR /workspace
COPY . .
RUN pip install --upgrade pip && \
    pip install setuptools setuptools_scm && \
    pip install -e '.[model]'
