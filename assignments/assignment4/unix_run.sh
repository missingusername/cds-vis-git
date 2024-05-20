#!/bin/bash

# Activate the virtual environment
source ./env/bin/activate

# Run the face_detection.py file, using optional comman line arguments
python src/face_detection.py "$@"

# Run the plot.py file afterwards
python src/plot.py

# Deactivate the virtual environment
deactivate
