#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import dump_svmlight_file
import pandas as pd
import numpy as np
import sys


def txt_to_libsvm(input_data_path, libsvm_data_path):
    print('----------Format Converting begin----------')

    func = lambda col: 'float32' if col.startswith('v') else 'object' if col == 'gid' else 'int8'
    v_columns = [f'v{i}' for i in range(1, 2181)]
    columns = ["label", "gid"] + v_columns
    dype = dict([(col, func(col)) for col in columns]) 
    input_data = pd.read_table(input_data_path, sep='\t', header=None, names=columns, dtype=dype)
    
    print(input_data)
    label = input_data["label"]
    gid = input_data["gid"]
    features = input_data[v_columns]
    features_array = features.values

    #features_array = np.nan_to_num(features_array)
    print(np.isnan(features_array).any())
    print(np.isfinite(features_array).all())
    features_array[np.isnan(features_array)] = 0
    features_array[np.isinf(features_array)] = 0

    # transform txt file to libsvm file
    dump_svmlight_file(features_array, label, libsvm_data_path, zero_based=False)

def main():
    data_path = sys.argv[1]
    input_data_libsvm_path = sys.argv[2]
    txt_to_libsvm(data_path, input_data_libsvm_path)


if __name__ == '__main__':
    main()


