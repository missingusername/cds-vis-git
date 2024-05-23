#!/bin/bash

# Create a Python virtual environment named "env"
python -m venv env

# Activate the virtual environment
source ./env/bin/activate

#update pip
pip install --upgrade pip

# Install required libraries from requirements.txt
pip install -r requirements.txt

# Deactivate the virtual environment
deactivate

#bash unix_setup.sh
