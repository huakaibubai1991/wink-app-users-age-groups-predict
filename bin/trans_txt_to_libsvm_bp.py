#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import dump_svmlight_file
import pandas as pd
import sys

def txt_to_libsvm(input_data, libsvm_data):
    print('----------Format Converting begin----------')
    v_columns = [f'v{i}' for i in range(1, 2181)]
    input_data_columns = ["label", "gid"] + v_columns
    input_data.columns = input_data_columns
    print(input_data)
    label = input_data["label"]
    gid = input_data["gid"]
    features = input_data[v_columns]
    features_array = features.values

    # transform txt file to libsvm file
    dump_svmlight_file(features_array, label, libsvm_data, zero_based=False)


def main():
    data_path = sys.argv[1]
    input_data = pd.read_table(data_path, sep='\t', header=None)
    input_data_libsvm = sys.argv[2]
    txt_to_libsvm(input_data, input_data_libsvm)


if __name__ == '__main__':
    main()
