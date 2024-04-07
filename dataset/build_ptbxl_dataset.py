import gzip
import os
import pickle

import numpy as np
import pandas as pd
from utils import (find_challenge_files, get_labels, load_header,
                         load_recording, get_id)

DATA_DIR = "./data/ptbxl"
DATA_DIR_TRAIN = os.path.join(DATA_DIR, "train")
DATA_DIR_TEST = os.path.join(DATA_DIR, "test")
RESULT_DIR = "./data"
RESULT_FILEPATH = f"{RESULT_DIR}/ptbxl_12leads.pkl"
LABEL_MAPPING_PATH = "./ptb-xl_labels/label_mapping.csv"
os.makedirs(RESULT_DIR, exist_ok=True)

label_mapping_df = pd.read_csv(LABEL_MAPPING_PATH, index_col=0)
classes = label_mapping_df["SNOMEDCTCode"].tolist()
classes = list(map(str, classes))

result_dict = {
    "train": {
        "header_files": [],
        "recording_files": []
    },
    "test": {
        "header_files": [],
        "recording_files": []
    },
}

def main():
    result_dict["train"]["header_files"], result_dict["train"]["recording_files"] = find_challenge_files(DATA_DIR_TRAIN)
    result_dict["test"]["header_files"], result_dict["test"]["recording_files"] = find_challenge_files(DATA_DIR_TEST)

    for set_key, set_dict in result_dict.items():
        ids = []
        X = []
        y = []
        for idx in range(len(set_dict["header_files"])):
            header_file = set_dict["header_files"][idx]
            recording_file = set_dict["recording_files"][idx]
            
            header = load_header(header_file)
            sample_id = get_id(header)
            
            label = [0] * len(classes)
            for l in (header_label := get_labels(header)):
                if l != '': # remove empty class
                    if l == "63593006":
                        l = "284470004" # replace equivalent class
                    try:
                        i = classes.index(l)
                        label[i] = 1
                    except ValueError: # Not in scored class list
                        pass

            if sum(label) == 0:
                print(f"Sample without target labels detected: {sample_id}")
                continue # skip this sample

            recording = load_recording(recording_file)
            
            ids.append(sample_id)
            X.append(recording)
            y.append(label)
        
        result_dict[set_key]["ids"] = ids
        result_dict[set_key]["X"] = np.array(X)
        result_dict[set_key]["y"] = y
    
    with gzip.open(RESULT_FILEPATH, 'wb') as f:
        pickle.dump(result_dict, f)

if __name__ == "__main__":
    main()