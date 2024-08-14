#!/bin/bash

# configure git
git config --global --add safe.directory /workspace
git config --global user.email \"${GITHUB_EMAIL}\" && git config --global user.name \"${GITHUB_USERNAME}\"

# configure pre-commit hooks
pip install --upgrade pip pre-commit
pre-commit autoupdate && pre-commit run --all

# install cursivetransformer module
pip install -e '.[model]'

# setup accelerate
echo "Setting up accelerate config"
mkdir -p ~/.cache/huggingface/accelerate
cp configs/accelerate.yaml ~/.cache/huggingface/accelerate/default_config.yaml
echo "Test accelerate setup"
accelerate test
