#!/usr/bin/env sh

dir="./data/data_ptbxl"
src_train=$dir"/train"
src_test=$dir"/test"

# echo "Extracting L2 Train..."
# python ./extract_leads_wfdb.py -i $src_train -o $dir"/train_2leads" -l I II

# echo "Extracting L2 Test..."
# python ./extract_leads_wfdb.py -i $src_test -o $dir"/test_2leads" -l I II

echo "Extracting L4 Train..."
python ./extract_leads_wfdb.py -i $src_train -o $dir"/train_4leads" -l I II V1 aVF

echo "Extracting L4 Test..."
python ./extract_leads_wfdb.py -i $src_test -o $dir"/test_4leads" -l I II V1 aVF
