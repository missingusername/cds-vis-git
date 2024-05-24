#!/bin/bash

# Activate the virtual environment
source ./env/bin/activate

# Run the simple search
python src/simple_search.py "$@"

# Run the KNN search
python src/knn_search.py "$@"

# Deactivate the virtual environment
deactivate

