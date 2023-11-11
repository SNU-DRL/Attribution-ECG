from glob import glob
import os
import pickle
import gzip
import numpy as np
from collections import Counter

BASE_PATH = "results_for_paper/results_attribution/st-petersburg_resnet18_7_bs32_lr1e-3_wd1e-4_ep20"

directories = glob(f"{BASE_PATH}_seed*")

count_result_dict = {0: 0, 1: 0, 2: 0, 3: 0}

for directory in sorted(directories):
    eval_attr_data = os.path.join(directory, "random_baseline", "eval_attr_data.pkl")
    data_dict = pickle.load(gzip.GzipFile(eval_attr_data, "rb"))    
    y_test = data_dict['y']
    count_result = Counter(y_test)
    for key in count_result:
        count_result_dict[key] += count_result[key]
        
        
for key in count_result_dict:
    print(key, count_result_dict[key] / len(directories))

print(sum(count_result_dict.values()) / len(directories))
