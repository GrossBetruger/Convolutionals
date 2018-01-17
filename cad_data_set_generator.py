import scipy.io as spio
import os
import numpy as np


def matlab_file_to_cad(matlab_file_path):
    return spio.loadmat(matlab_file_path, squeeze_me=True).get('instance')


def prepare_training_set(train_dir):
    labels = os.listdir(train_dir)
    for i, label in enumerate(labels):
        l = np.zeros(len(labels), dtype=int)
        l[i] = 1
        for raw_data_path in os.listdir(os.path.join(train_dir, label)):
            cad = matlab_file_to_cad(os.path.join(train_dir, label, raw_data_path))
            yield [cad, l]


def prepare_test_set(test_dir):
    for raw_data_path in os.listdir(test_dir):
        cad = matlab_file_to_cad(os.path.join(test_dir, raw_data_path))
        yield cad


if __name__ == "__main__":
    data_gen = prepare_training_set("train_cad")
    for _ in data_gen:
        print _
    test_data_gen = prepare_test_set("test_cad")
    for _ in test_data_gen:
        print _[13]

    # print matlab_file_to_cad(os.path.join("table", "30", "train", 'table_000000992_1.mat'))[8]
