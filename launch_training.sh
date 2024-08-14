#!/bin/bash

# Configure git
git config --global --add safe.directory /opt/artisight/llm_lab

# Set environment variables to help with timeout issues
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=23
export NCCL_SOCKET_TIMEOUT=23

# Increase the timeout for monitored barrier
export TORCHELASTIC_MAX_SECONDS=3600

# Setup accelerate
echo "Setting up accelerate config"
accelerate test --config_file "${CURSIVE_TRANSFORMER_ACCELERATE_CONF}"

if [ $? -eq 0 ]; then
    echo "Accelerate config is valid. Launching training..."
    accelerate launch --config_file "${CURSIVE_TRANSFORMER_ACCELERATE_CONF}" \
    -m cursivetransformer --config "${CURSIVE_TRANSFORMER_TRAIN_CONF}"
else
    echo "Accelerate config test failed. Please check your configuration."
    exit 1
fi
