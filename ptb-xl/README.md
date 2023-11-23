https://www.kaggle.com/datasets/bjoernjostein/ptbxl-electrocardiography-database/data
여기서 zip파일 download

1. train / test 분리
```
    $ python split_train_test.py split_train_test.yaml
```

2. 2 leads 추출
```
    $ sh extract_leadsets.sh
```

