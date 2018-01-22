import scipy.io as spio
import os
import numpy as np
from random import randint


def matlab_file_to_cad(matlab_file_path):
    return spio.loadmat(matlab_file_path, squeeze_me=True).get('instance')


def prepare_data_set(dataset_dir, batch_size, channels, limit=None, balanced=True, fuzzing_mode=False):
    labels = os.listdir(dataset_dir)
    data_set_sizes = [len(os.listdir(os.path.join(dataset_dir, label))) for label in labels]
    data_set_sizes_all_equal = len(set(data_set_sizes)) == 1
    if limit is None and balanced and not data_set_sizes_all_equal:
        limit = min(data_set_sizes)
    for i, label in enumerate(labels):
        print "creating data for lable:", label, "--", "ord:", i
        l = np.zeros(len(labels), dtype=int)
        l[i] = 1
        if fuzzing_mode:
            # NEVER SET THIS FLAG TRUE unless you know what you're doing
            l[i] = randint(0, len(labels)-1)
        l = [l] * batch_size
        for raw_data_path in os.listdir(os.path.join(dataset_dir, label))[:limit]:
            cad = matlab_file_to_cad(os.path.join(dataset_dir, label, raw_data_path))
            cad = [cad.reshape(cad.shape[0], cad.shape[1], cad.shape[2], channels)] * batch_size
            yield [cad, l]


if __name__ == "__main__":
    data_gen = prepare_data_set("train_cad")
    for _ in data_gen:
        print _