import glob
import math
import os
import random
import shutil

import pandas as pd

from utils import get_is_scored, get_labels_from_filename, set_random_seed

DX_MAPPING_PATH = "./ptbxl_labels/dx_mapping_targeted.csv"
SPLIT_RATIO = 0.2
SOURCE_PATH = "./source/WFDB"
RESULT_PATH = "./data/ptbxl"

def main():
    train_path = os.path.join(RESULT_PATH, "train")
    test_path = os.path.join(RESULT_PATH, "test")
    os.makedirs(train_path, exist_ok=False) # overwriting is not recommended; set to False
    os.makedirs(test_path, exist_ok=False)
    
    dx_mapping_scored = pd.read_csv(DX_MAPPING_PATH, dtype=str)
    scored_labels = [str(each) for each in dx_mapping_scored['SNOMEDCTCode'].tolist()]
    scored_set = set(scored_labels)

    train_label_df = dx_mapping_scored.iloc[:, 0:3]
    train_label_df = train_label_df.assign(ptbxl=[0] * len(train_label_df))
    test_label_df = dx_mapping_scored.iloc[:, 0:3]
    test_label_df = test_label_df.assign(ptbxl=[0] * len(test_label_df))

    header_files = glob.glob(f'{SOURCE_PATH}/*.hea')
    header_files.sort()
    num_samples = len(header_files)

    # label in a scored label set?
    is_scored_list = [get_is_scored(header_file, scored_set) for header_file in header_files]
    df = pd.DataFrame({'header': header_files, 'is_scored': is_scored_list})

    indices_is_scored = list(df[df.is_scored == True].index)
    random.shuffle(indices_is_scored)
    print(f'# of scored samples: {len(indices_is_scored)}')
    
    num_test_samples = math.floor(len(indices_is_scored) * SPLIT_RATIO)
    num_train_samples = num_samples - num_test_samples
    test_indices = indices_is_scored[:num_test_samples]

    df['is_test'] = [True if t in test_indices else False for t in range(num_samples)]
    assert df['is_test'].sum() == num_test_samples
    print(f'# of training samples: {num_train_samples}, # of test samples: {num_test_samples}')

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
        labels = get_labels_from_filename(row['header'])
        for label in labels:
            if label in scored_labels:
                train_label_df.loc[train_label_df.SNOMEDCTCode == label, 'ptbxl'] += 1

    test_df = df[df.is_test == True]
    for idx, row in test_df.iterrows():
        labels = get_labels_from_filename(row['header'])
        for label in labels:
            if label in scored_labels:
                test_label_df.loc[test_label_df.SNOMEDCTCode == label, 'ptbxl'] += 1

    train_label_df.to_csv(os.path.join(RESULT_PATH, "train_label_dist.csv"))
    test_label_df.to_csv(os.path.join(RESULT_PATH, "test_label_dist.csv"))

if __name__ == "__main__":
    set_random_seed(42)
    main()
