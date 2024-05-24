#!/bin/bash

# Activate the virtual environment
source ./env/bin/activate

# Run the logreg.py file
python src/logreg.py

# Run the mlp.py file
python src/mlp.py

# Deactivate the virtual environment
deactivate

