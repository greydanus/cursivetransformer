#!/bin/bash

# configure git
git config --global --add safe.directory /workspace
git config --global user.email \"${GITHUB_EMAIL}\" && git config --global user.name \"${GITHUB_USERNAME}\"

# configure pre-commit hooks
pip install --upgrade pip pre-commit
pre-commit autoupdate && pre-commit run --all

# install cursivetransformer module
pip install -e '.[dataset]'
