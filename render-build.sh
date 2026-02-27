#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Run training scripts to generate model artifacts and stats
# This ensures deployment always has the latest logic/data processed
python data/generate_data.py
python analyze_datasets.py
python models/train_forecaster.py
python models/train_suggester.py

echo "Build process completed successfully."
