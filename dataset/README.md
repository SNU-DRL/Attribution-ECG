## Dataset
Download each data source to the `/dataset/data` directory, and build datasets from the sources.

1. MIT-BIH Arrhythmia Database (MITDB)

Original source: [MIT-BIH Arrhythmia Database](https://www.physionet.org/content/mitdb/1.0.0/)
```
    mkdir source
    wget https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip -P source
    unzip source/mit-bih-arrhythmia-database-1.0.0.zip -d source/
    python build_mitdb_dataset.py
```

2. MIT-BIH Supraventricular Arrhythmia Database (SVDB)

Original source: [MIT-BIH Supraventricular Arrhythmia Database](https://physionet.org/content/svdb/1.0.0/)
```
    mkdir source
    wget https://physionet.org/static/published-projects/svdb/mit-bih-supraventricular-arrhythmia-database-1.0.0.zip -P source
    unzip source/mit-bih-supraventricular-arrhythmia-database-1.0.0.zip -d source/
    python build_svdb_dataset.py
```

3. St Petersburg INCART 12-lead Arrhythmia Database (INCARTDB)

Original source: [St Petersburg INCART 12-lead Arrhythmia Database](https://physionet.org/content/incartdb/1.0.0/)
```
    mkdir source
    wget https://physionet.org/static/published-projects/incartdb/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0.zip -P source
    unzip source/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0.zip -d source/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0
    python build_incartdb_dataset.py
```


4. Icentia11K (ICENTIA11K)

Subset of Icentia11K used for beat classification task in our experiment (sample size = 12000): https://drive.google.com/drive/folders/1UJsW6iW13ONoGy6WCdACdG5O_xoswie1?usp=sharing

Original source: [Icentia11k: An Unsupervised ECG Representation Learning Dataset for Arrhythmia Subtype Discovery](https://academictorrents.com/details/af04abfe9a3c96b30e5dd029eb185e19a7055272)

Download `icentia11k.pkl` and move it into `/dataset/data`.
