# Heart Segmentation
Estimating size of the heart by solving heart segmentation task.

### Dependencies
Minimum Python 3.7 is required to install the dependecies below. 

To install all required dependencies, please run
```bash
source install_packages.sh 
```
This code creates a local virtual environment and installs needed packages within it.

### Extract Prediction
To run the inference, please use the following command:
```bash
python run.py --scan_path <PATH_TO_DICOM_SERIES>
```
Your result will be saved to a folder `inference_results` that is automatically created in your running directory.
However, you can also specify the target directory where to save the prediscted volume:
```bash
python run.py --scan_path <PATH_TO_DICOM_SERIES> --result_save <PATH_TO_SAVING>
```
The code by design is using CPU for inference. 
If you need to change the device and have access to the GPU, add `--device 'cuda:0'` to the running command above.