# Comprehensive Evaluation of Feature Attribution Methods in Heart Anomaly Detection
This repository contains the code for evaluating feature attribution methods in heart anomaly detection tasks using ECG datasets.
Our paper is currently under review, and we will update the links and experimental scripts once the paper is published.
Our preliminary study is on arXiv, titled [*EVALUATING FEATURE ATTRIBUTION METHODS FOR ELECTROCARIDOGRAM*](https://arxiv.org/abs/2211.12702).

## Requirements
This experiment was implemented using the following libraries:

- Python 3.10
- PyTorch 2.0.0
- Please refer to `requirements.txt` for other libraries.
```
    pip install -r requirements.txt
```

## Getting started

### Building datasets
Please refer to [this](dataset/README.md).
- Available datasets: `mitdb`, `svdb`, `incartdb`, `icentia11k`, `ptbxl`

### Training a model
```
    python train.py
        --dataset       icentia11k
        --dataset_path  ./dataset/data/icentia11k.pkl
        --result_dir    ./result_train
```
- Result: A trained model is saved in the directory specified in `--result_dir` argument.


### Computing attribution values
```
    python compute_attribution.py
        --dataset       icentia11k
        --dataset_path  ./dataset/data/icentia11k.pkl
        --model_path    ./result_train/model_last.pt
        --attr_method   gradcam
        --result_dir    ./result_attr
```
- Avaliable feature attribution methods:, `saliency`, `input_gradient`, `guided_backprop`, `integrated_gradients`, `deep_lift`, `deep_shap`, `lrp`, `lime`, `kernel_shap`, `gradcam`, `guided_gradcam`
- Result: Attribution values of test samples are saved in the directory specified in `--result_dir` argument.
    - `eval_attr_data.pkl`: Contains test samples used for evaluating feature attribution methods.
    - `attr_list.pkl`: Stores attribution values of samples found in eval_attr_data.pkl.


### Evaluating feature attribution methods
```
    python evaluate_attribution.py
        --attr_dir      ./result_attr
        --model_path    ./result_train/model_last.pt
        --eval_metric   attribution_localization
        --result_dir    ./result_eval
```
- Available evaluation metrics: `attribution_localization`, `auc`, `pointing_game`, `relevance_mass_accuracy`, `relevance_rank_accuracy`, `top_k_intersection`, `region_perturbation`, `faithfulness_correlation`
- Result: Evaluation results of feature attributions using test samples are saved in the directory specified in `--result_dir` argument.


## Conducting experiments using provided scripts
The scripts for our experiments are located in the `script` directory.
Please execute the scripts in the following order:
- `scripts/train/train_*.sh`
- `scripts/compute_attribution/compute_attribution_*.sh`
- `scripts/evaluate_attribution/evaluate_attribution_*.sh`

The scripts should be executed in the root of this repository.
```
    sh scripts/train/train_icentia11k.sh
    sh scripts/compute_attribution/compute_attribution_icentia11k.sh
    sh scripts/evaluate_attribution/evaluate_attribution_icentia11k.sh
```

Please note that the values of the results may vary slightly if the experiments are run on different machines or different versions of libraries.
