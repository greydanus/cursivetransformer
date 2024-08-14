#!/bin/bash

# configure git
git config --global --add safe.directory /opt/artisight/llm_lab
git config --global user.email \"${GITLAB_EMAIL}\" && git config --global user.name \"${GITLAB_USERNAME}\"

# configure pre-commit hooks
pip install --upgrade pip pre-commit
pre-commit autoupdate && pre-commit run --all

# install llm_lab module
pip install -e '.[model]'

# setup accelerate
echo "Setting up accelerate config"
mkdir -p ~/.cache/huggingface/accelerate
cp configs/accelerate.yaml ~/.cache/huggingface/accelerate/default_config.yaml
echo "Test accelerate setup"
accelerate test
