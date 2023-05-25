#!/bin/bash

# Creating virtual environment
python3 -m venv inference_env

# Activating
source inference_env/bin/activate

pip install --upgrade pip

# installing packages
pip install torch

pip install monai==1.0.1 pydicom nibabel connected-components-3d opencv-python matplotlib

echo "Done! Your directory is ready for inference"
