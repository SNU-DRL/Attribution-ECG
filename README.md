# Evaluating Feature Attribution Methods for Electrocardiogram
### Under construction!

---

This repository contains the code for experiments in our paper, [*EVALUATING FEATURE ATTRIBUTION METHODS FOR ELECTROCARIDOGRAM*](https://arxiv.org/abs/2211.12702).

**NOTICE: Our paper is under review as of now, and we are planning to include more details here when our paper is published so other researchers can reproduce our experiments and use our implementation code.**



## Dataset
1. Icentia11K
Subset of Icentia11K used for beat classification task in our experiment (sample size = 12000): https://drive.google.com/drive/folders/1UJsW6iW13ONoGy6WCdACdG5O_xoswie1?usp=sharing
Original source: [Icentia11k: An Unsupervised ECG Representation Learning Dataset for Arrhythmia Subtype Discovery](https://academictorrents.com/details/af04abfe9a3c96b30e5dd029eb185e19a7055272)

```
    cd dataset/data
    mv 12000_btype_new.pkl icentia11k.pkl
```


2. MIT-BIH Arrhythmia Database
Original source: [MIT-BIH Arrhythmia Database](https://www.physionet.org/content/mitdb/1.0.0/)
```
    cd dataset
    wget https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip
    unzip mit-bih-arrhythmia-database-1.0.0.zip
    python build_mitbih_dataset.py
```

3. St Petersburg INCART 12-lead Arrhythmia Database
Original source: [St Petersburg INCART 12-lead Arrhythmia Database](https://physionet.org/content/incartdb/1.0.0/)
```
    cd dataset
    wget https://physionet.org/static/published-projects/incartdb/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0.zip
    unzip st-petersburg-incart-12-lead-arrhythmia-database-1.0.0.zip -d st-petersburg-incart-12-lead-arrhythmia-database-1.0.0
    python build_st-petersburg_dataset.py
```

4. MIT-BIH Supraventricular Arrhythmia Database
Original source: [MIT-BIH Supraventricular Arrhythmia Database](https://physionet.org/content/svdb/1.0.0/)
```
    cd dataset
    wget https://physionet.org/static/published-projects/svdb/mit-bih-supraventricular-arrhythmia-database-1.0.0.zip
    unzip mit-bih-supraventricular-arrhythmia-database-1.0.0.zip
    python build_mitbih_svdb_dataset.py

```


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