#!/bin/bash

cd /workspace/finetuning
/usr/bin/python3 \
    ./neurons/validator.py \
    --wallet.name cfusion \
    --wallet.hotkey v1 \
    --subtensor.network finney