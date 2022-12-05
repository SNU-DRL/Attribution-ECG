# Evaluating Feature Attribution Methods for Electrocardiogram
This repository contains the code for experiments in our paper, *EVALUATING FEATURE ATTRIBUTION METHODS FOR ELECTROCARIDOGRAM*.

**NOTICE: Our paper is under review as of now, and we are planning to include more details here when our paper is published so other researchers can reproduce our experiments and use our implementation code.**


## Dataset
Subset of Icentia11K used for beat classification task in our experiment (sample size = 12000): https://drive.google.com/drive/folders/1UJsW6iW13ONoGy6WCdACdG5O_xoswie1?usp=sharing

Original source: [Icentia11k: An Unsupervised ECG Representation Learning Dataset for Arrhythmia Subtype Discovery](https://academictorrents.com/details/af04abfe9a3c96b30e5dd029eb185e19a7055272)


## How to use
### Requirements
- PyTorch==1.8.1
```
    pip install -r requirements.txt
```


### Training a model
```
    python train.py
```
You will get a trained model in the directory specified in `--results_path` argument.

### Run feature attribution methods
```
    python run_attribution.py
```
You need to specify the path of a trained model in line 53 (will be fixed).
You will get the matrix of feature attribution result in the directory specified in `--results_path`.

### Evaluating feature attribution methods
```
    python evaluate_attributions.py --gpu $gpu --attr_method $method --model_path $model_path --results_path results_dir 
```
You need to pass the path of a trained model with `--model_path` argument.
You will get result files(`attr_eval_all.json`) in the directory specified in `--results_path`.