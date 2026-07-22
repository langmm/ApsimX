#!/bin/bash
set -e
# source /opt/conda/etc/profile.d/conda.sh
# conda activate env
fastapi run --host 0.0.0.0 --port $PORT main.py
