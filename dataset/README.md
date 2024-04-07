## Dataset
Download each data source to the `/dataset/data` directory, and build datasets from the sources.

### 1. MIT-BIH Arrhythmia Database (MITDB)

```
    mkdir source
    wget https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip -P source
    unzip source/mit-bih-arrhythmia-database-1.0.0.zip -d source/
    python build_mitdb_dataset.py
```

*Original source: [MIT-BIH Arrhythmia Database](https://www.physionet.org/content/mitdb/1.0.0/)*


###  2. MIT-BIH Supraventricular Arrhythmia Database (SVDB)

```
    mkdir source
    wget https://physionet.org/static/published-projects/svdb/mit-bih-supraventricular-arrhythmia-database-1.0.0.zip -P source
    unzip source/mit-bih-supraventricular-arrhythmia-database-1.0.0.zip -d source/
    python build_svdb_dataset.py
```


*Original source: [MIT-BIH Supraventricular Arrhythmia Database](https://physionet.org/content/svdb/1.0.0/)*

### 3. St Petersburg INCART 12-lead Arrhythmia Database (INCARTDB)

```
    mkdir source
    wget https://physionet.org/static/published-projects/incartdb/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0.zip -P source
    unzip source/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0.zip -d source/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0
    python build_incartdb_dataset.py
```

*Original source: [St Petersburg INCART 12-lead Arrhythmia Database](https://physionet.org/content/incartdb/1.0.0/)*


### 4. Icentia11K (ICENTIA11K)

Download `icentia11k.pkl` from the link below and move it into `/dataset/data`.

Subset of Icentia11K used for beat classification task in our experiment (sample size = 12000): https://drive.google.com/drive/folders/1UJsW6iW13ONoGy6WCdACdG5O_xoswie1?usp=sharing

*Original source: [Icentia11k: An Unsupervised ECG Representation Learning Dataset for Arrhythmia Subtype Discovery](https://academictorrents.com/details/af04abfe9a3c96b30e5dd029eb185e19a7055272)*


### 5. PTB-XL

Download a zip file from https://www.kaggle.com/datasets/bjoernjostein/ptbxl-electrocardiography-database/data.

```
    mkdir source
    // move the downloaded `archive.zip` into the `source/`
    unzip source/archive.zip -d source/
    python split_train_test.py
    python build_ptbxl_dataset.py
```

When building the PTB-XL dataset, we used 22 labels utilized as target labels in the [PhysioNet/CinC Challenge 2021](https://github.com/physionetchallenges/evaluation-2021/blob/main/dx_mapping_scored.csv).
The following pairs of labels were regarded as equivalent, as suggested by the challenge.

| Label 1                            | Label 2                          |
|------------------------------------|----------------------------------|
| complete left bundle branch block  | left bundle branch block         |
| complete right bundle branch block | right bundle branch block        |
| premature atrial contraction       | supraventricular premature beats |
| premature ventricular contractions | ventricular premature beats      |


*Original source: [PTB-XL, a large publicly available electrocardiography dataset](https://physionet.org/content/ptb-xl/1.0.3/)*
