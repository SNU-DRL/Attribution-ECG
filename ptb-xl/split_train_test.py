import yaml
import pandas as pd
import numpy as np
import glob
import os
import shutil
import sys
import random
import math
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
pd.options.mode.chained_assignment = None  # default='warn'

def set_random_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)

def get_is_scored(header_file_name, scored_set):
    fid = open(header_file_name, 'r')
    readall = fid.readlines()
    fid.close()
    for each in readall:
        if each[:5]=='#Dx: ':
            dx_set = set(each[5:].replace('\n','').split(','))
    if len(dx_set.intersection(scored_set)) > 0:
        is_scored = True
    else:
        is_scored = False
    return is_scored

# Get labels from header file.
def get_labels(header_file_name):
    labels = list()
    with open(header_file_name, 'r') as f:
        header = f.read()
        for l in header.split('\n'):
            if l.startswith('#Dx'):
                try:
                    entries = l.split(': ')[1].split(',')
                    for entry in entries:
                        labels.append(entry.strip())
                except:
                    pass
    return labels

def split_train_test(yaml_file):
    config = yaml.load(open(yaml_file,'r'),Loader=yaml.FullLoader)

    dx_mapping_scored = config['dx_mapping_scored']
    source_path = config['source_path']
    result_path = config['result_path']
    train_path = os.path.join(result_path, "train")
    test_path = os.path.join(result_path, "test")
    split_ratio = config['split_ratio']

    dx_mapping_scored = pd.read_csv(dx_mapping_scored, dtype=str)
    scored_labels = [str(each) for each in dx_mapping_scored['SNOMEDCTCode'].tolist()]
    scored_set = set(scored_labels)

    train_label_df = dx_mapping_scored.iloc[:, 0:3]
    train_label_df["ptbxl"] = 0
    test_label_df = dx_mapping_scored.iloc[:, 0:3]
    test_label_df["ptbxl"] = 0

    header_files = glob.glob(f'{source_path}/*.hea')
    header_files.sort()
    num_samples = len(header_files)

    # label in a scored label set?
    is_scored_list = [get_is_scored(header_file, scored_set) for header_file in header_files]
    df = pd.DataFrame({'header': header_files, 'is_scored': is_scored_list})

    indices_is_scored = list(df[df.is_scored == True].index)
    random.shuffle(indices_is_scored)
    print(f'# of scored samples: {len(indices_is_scored)}')
    
    num_test_samples = math.floor(len(indices_is_scored) * split_ratio)
    num_train_samples = num_samples - num_test_samples
    test_indices = indices_is_scored[:num_test_samples]

    df['is_test'] = [True if t in test_indices else False for t in range(num_samples)]
    assert df['is_test'].sum() == num_test_samples
    print(f'# of training samples: {num_train_samples}, # of test samples: {num_test_samples}')

    os.makedirs(train_path, exist_ok=False) # overwriting is not recommended; set to False
    os.makedirs(test_path, exist_ok=False)

    train_header_list = df.query('not is_test')['header'].tolist()
    test_header_list = df.query('is_test')['header'].tolist()

    print("Copying files...")
    for header_file in train_header_list:
        base_name = os.path.basename(header_file).replace('.hea', '')
        dir_name = os.path.dirname(header_file)
        shutil.copy('{0}/{1}.hea'.format(dir_name, base_name), \
            '{0}/{1}.hea'.format(train_path, base_name))
        shutil.copy('{0}/{1}.mat'.format(dir_name, base_name), \
            '{0}/{1}.mat'.format(train_path, base_name))

    for header_file in test_header_list:
        base_name = os.path.basename(header_file).replace('.hea', '')
        dir_name = os.path.dirname(header_file)
        shutil.copy('{0}/{1}.hea'.format(dir_name, base_name), \
            '{0}/{1}.hea'.format(test_path, base_name))
        shutil.copy('{0}/{1}.mat'.format(dir_name, base_name), \
            '{0}/{1}.mat'.format(test_path, base_name))

    # Label distribution
    print("Calculating label distributions...")
    train_df = df[df.is_test == False]
    for idx, row in train_df.iterrows():
        labels = get_labels(row['header'])
        for label in labels:
            if label in scored_labels:
                train_label_df.loc[train_label_df.SNOMEDCTCode == label, 'ptbxl']+= 1

    test_df = df[df.is_test == True]
    for idx, row in test_df.iterrows():
        labels = get_labels(row['header'])
        for label in labels:
            if label in scored_labels:
                test_label_df.loc[test_label_df.SNOMEDCTCode == label, 'ptbxl']+= 1

    train_label_df.to_csv(os.path.join(result_path, "train_label_dist.csv"))
    test_label_df.to_csv(os.path.join(result_path, "test_label_dist.csv"))

if __name__ == "__main__":
    # Parse arguments.
    if len(sys.argv) != 2:
        raise Exception(
            "Include the yaml file as arguments, e.g., python split_train_test.py split_train_test.yaml"
        )
    set_random_seed(42)
    split_train_test(sys.argv[1])


