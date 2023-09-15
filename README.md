# Evaluating Feature Attribution Methods for Electrocardiogram
### Under construction!

---

This repository contains the code for experiments in our paper, [*EVALUATING FEATURE ATTRIBUTION METHODS FOR ELECTROCARIDOGRAM*](https://arxiv.org/abs/2211.12702).

**NOTICE: Our paper is under review as of now, and we are planning to include more details here when our paper is published so other researchers can reproduce our experiments and use our implementation code.**

## How to use
### Requirements
Tested on
- Python 3.10
- PyTorch 2.0.0
```
    conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
- Other packages
```
    pip install -r requirements.txt
```
### Building datasets
Please refer to [this](dataset/README.md).

### Training a model
```
    python train.py
```
- Result: a trained model in the directory specified in `--result_dir` argument.

### Compute attribution
```
    python compute_attribution.py
```
- Avaliable attribution methods:, "saliency", "input_gradient", "guided_backprop", "integrated_gradients", "deep_lift", "deep_shap", "lrp", "lime", "kernel_shap", "gradcam", "guided_gradcam"
- Result: feature attribution values of test samples in the directory specified in `--result_dir` argument.

### Evaluating feature attribution methods
```
    python evaluate_attributions.py
```
- Available evaluation metrics: "attribution_localization", "auc", "pointing_game", "relevance_mass_accuracy", "relevance_rank_accuracy", "top_k_intersection"
- Result: evaluation results of feature attributions in test set in the directory specified in `--result_dir` argument.