#!/bin/bash
# Install dependencies for Kaggle environment
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements.txt"
