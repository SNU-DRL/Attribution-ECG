# Evaluating Feature Attribution Methods for Electrocardiogram
This repository contains the code for experiments in our paper, [*EVALUATING FEATURE ATTRIBUTION METHODS FOR ELECTROCARIDOGRAM*](https://arxiv.org/abs/2211.12702).

**NOTICE: Our paper is under review as of now, and we are planning to include more details here when our paper is published so other researchers can reproduce our experiments and use our implementation code.**


## Dataset
Subset of Icentia11K used for beat classification task in our experiment (sample size = 12000): https://drive.google.com/drive/folders/1UJsW6iW13ONoGy6WCdACdG5O_xoswie1?usp=sharing

Original source: [Icentia11k: An Unsupervised ECG Representation Learning Dataset for Arrhythmia Subtype Discovery](https://academictorrents.com/details/af04abfe9a3c96b30e5dd029eb185e19a7055272)


## How to use
### Requirements
Tested on
- Python 3.10.10
- PyTorch 2.0.0
```
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
- Other packages
```
    pip install -r requirements.txt
```


### Training a model
```
    python train.py
```
You will get a trained model in the directory specified in `--results_path` argument.

### Evaluating feature attribution methods
```
    python evaluate_attributions.py --gpu $gpu --attr_method $method --model_path $model_path --results_path results_dir 
```
You need to pass the path of a trained model with `--model_path` argument.
You will get result files(`attr_eval_all.json`) in the directory specified in `--results_path`.
- Avaliable attribution methods:, "saliency", "input_gradient", "guided_backprop", "integrated_gradients", "deep_lift", "deep_shap", "lrp", "lime", "kernel_shap", "gradcam", "guided_gradcam"