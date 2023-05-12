# Inference with nnUNet

### Create a virtual environment
```
# Creating virtual environment
python3 -m venv nnunet_env

# Activating
source nnunet_env/bin/activate

pip install --upgrade pip
```

### Dependencies
To install all required dependencies, please follow the offical [instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) from nnUNet repo.

### Extract Prediction
To run the inference, please use the following command:
```bash
export RESULTS_FOLDER="nnUNet_trained_models"

nnUNet_predict -i nnUNet_raw_data_base/nnUNet_raw_data/Task501_NLST/imagesTs -o ~/results_nnUnet -t 0 -m 3d_fullres
```
Your result will be saved to a folder `results_nnUnet` that is automatically created in your running directory.
However, you can also edit the target directory where to save the prediscted volume by changing the path in tag `-o`.

