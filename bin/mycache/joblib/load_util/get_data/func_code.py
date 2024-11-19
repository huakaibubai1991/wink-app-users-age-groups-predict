# first line: 8
@mem.cache
def get_data(libsvm_path):
    data = load_svmlight_file(libsvm_path)
    return data[0], data[1]
