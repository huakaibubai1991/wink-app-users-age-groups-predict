from sklearn.datasets import load_svmlight_file
from joblib import Memory
import numpy as np

mem = Memory("./mycache")


@mem.cache
def get_data(libsvm_path):
    data = load_svmlight_file(libsvm_path)
    return data[0], data[1]


def get_qid_data(libsvm_path):
    data = load_svmlight_file(libsvm_path, query_id=True)
    return data[0], data[1].astype(np.int64)


def load_over_sampling(filename):
    import pickle
    filename_x = filename + "_x"
    filename_y = filename + "_y"
    x = pickle.load(open(filename_x, 'rb'))
    y = pickle.load(open(filename_y, 'rb'))
    return x, y


def save_over_sampling(x, y, filename):
    import pickle
    filename_x = filename + "_x"
    filename_y = filename + "_y"
    pickle.dump(x, open(filename_x, 'wb'))
    pickle.dump(y, open(filename_y, 'wb'))
    print("save successful~")

