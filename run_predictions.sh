#!/bin/bash

# Activate the environment
source activate abisr_test

# Run predictions
python predict.py --model_path='models/PansharpeningCNN_12_layers_256_filters/' \
                   --input_path='.' \
                   --output_path='.' \
                   --timestring='s20221251520205' \
                   --gpu_index=0 \
                   --chunk_size=1024 \
                   --overlap=32 \
                   --domain='FD_Example4' \
                   --n_parallel_out=8


