# kidney-sonography
## Introduction
kidney function classification and prediction through ultrasound-based kidney imaging: from deep learning to mass screening of chronic kidney disease

## Prediction module
Ensembles 10 trained models to predict estimated glomerular filtration rate (eGFR).
### Tutorial
1. Run get_models.sh to retrieve 10 trained model weights. (Warning! The file size is about 1.7 GB.)
2. Execute ensemble_predict.py
3. Input file path to a cropped kidney sonography.

Use --help to see usage of ensemble_predict.py
'''
usage: ensemble_predict.py [-h] [-g]

optional arguments:
-h, --help    show help message and exit
-g, --gpu_id  assign GPU ID, default 0
'''
