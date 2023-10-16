# Evaluating Feature Attribution Methods for Electrocardiogram
### Under construction!

---

This repository contains the code for experiments in our paper, [*EVALUATING FEATURE ATTRIBUTION METHODS FOR ELECTROCARIDOGRAM*](https://arxiv.org/abs/2211.12702).

**NOTICE: Our paper is under review as of now, and we are planning to include more details here when our paper is published so other researchers can reproduce our experiments and use our implementation code.**


## How to use
### Requirements
This experiment was tested using the following libraries:

- Python 3.10
- PyTorch 2.0.0
- Please refer to `requirements.txt` for other libraries.
```
    pip install -r requirements.txt
```


### Building datasets
Please refer to [this](dataset/README.md).


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
- Avaliable feature attribution methods:, "saliency", "input_gradient", "guided_backprop", "integrated_gradients", "deep_lift", "deep_shap", "lrp", "lime", "kernel_shap", "gradcam", "guided_gradcam"
- Result: Attribution values of test samples are saved in the directory specified in `--result_dir` argument.
    - `eval_attr_data.pkl`: Contains test samples used for evaluating feature attribution methods.
    - `attr_list.pkl`: Stores attribution values of samples found in eval_attr_data.pkl.


### Evaluating feature attribution methods
```
    python evaluate_attributions.py
        --attr_dir      ./result_attr
        --model_path    ./result_train/model_last.pt
        --eval_metric   attribution_localization
        --result_dir    ./result_eval
```
- Available evaluation metrics: "attribution_localization", "auc", "pointing_game", "relevance_mass_accuracy", "relevance_rank_accuracy", "top_k_intersection", "region_perturbation"
- Result: Evaluation results of feature attributions using test samples are saved in the directory specified in `--result_dir` argument.