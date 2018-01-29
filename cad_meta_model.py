import tensorflow as tf
from classifier_3d import create_dataset, \
    build_3dconv_cvnn, build_concat3dconv_cvnn, show_stats
import os
import numpy as np
from collections import Counter
from pickle import dump, load
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sys


def average_prediction_vectors(lst_of_raw_predictions):
    first_pred = np.array(lst_of_raw_predictions[0])
    for raw_pred in lst_of_raw_predictions[1:]:
        first_pred += np.array(raw_pred)
    return np.argmax(first_pred / float(len(lst_of_raw_predictions)))


def restore_model(saver, model_save_path, session):
    if os.path.exists(os.path.join(model_save_path, 'checkpoint')):
        saver.restore(session, tf.train.latest_checkpoint(model_save_path))
    else:
        raise Exception("MODEL NOT FOUND")


def train_high_level_model(dataset, classifier):
    X1 = list(dataset["regular_pred"])
    X2 = list(dataset["augmented_pred"])
    X3 = list(dataset["concat_pred"])
    assert len(X1) == len(X2)
    X = [X1[i] + X2[i] + X3[i] for i in range(len(X1))]
    classifier.fit(X, dataset["labels"])
    return classifier


def predict_index(model, raw_pred_reg, raw_pred_aug, raw_pred_concat):
    return model.predict([list(raw_pred_reg) + list(raw_pred_aug) + list(raw_pred_concat)])[0]


def run_model(data, label, network, model_path, counter):

    with tf.Session() as regular_session:
        tf.global_variables_initializer().run()
        inputs_reg, _, _, prediction_reg = network(mode)

        saver1 = tf.train.Saver(tf.trainable_variables())

        model_save_path = model_path
        restore_model(saver1, model_save_path, regular_session)

        raw_pred = regular_session.run([prediction_reg], feed_dict={inputs_reg: data})[0][0]
        print "\nraw prediction, model: ", model_path
        print raw_pred

        print "\nprediction", "model: ", model_path
        final_pred_idx = np.argmax(raw_pred)
        print final_pred_idx
        counter.update([label[0][final_pred_idx] == 1])
        show_stats(counter)

    tf.reset_default_graph()
    return raw_pred



if __name__ == "__main__":
    try:
        save_meta_model_path = sys.argv[1]
    except IndexError:
        print "USAGE: python cad_meta_model.py save_meta_model_path"

    machine_learning_data_set_path = "decision_tree_training_set_real_with_aug.json"
    if os.path.exists(machine_learning_data_set_path):
        with open(machine_learning_data_set_path) as f:
            decision_tree_dataset = load(f)

    forest = train_high_level_model(decision_tree_dataset, RandomForestClassifier())
    tree = train_high_level_model(decision_tree_dataset, DecisionTreeClassifier())
    logistic_regression = train_high_level_model(decision_tree_dataset, LogisticRegression())
    svm = train_high_level_model(decision_tree_dataset, SVC())

    print "done training"
    mode = "test"

    print "Test Dataset"
    data_set = create_dataset("test_cad_10.tar.gz")

    regular_counter = Counter()
    data_aug_counter = Counter()
    concat_counter = Counter()
    naive_counter = Counter()
    model_counter = Counter()

    decision_tree_dataset = {"regular_pred": [],
                             "augmented_pred": [],
                               "concat_pred": [],
                               "labels": []}

    decision_tree_dataset_counter = int()

    data_set_length = len(data_set)
    print "running models and training meta model"
    for i, batch in enumerate(data_set):
        print "SAMPLES LEFT:", data_set_length - i
        data, label = batch[0], batch[1]
        raw_pred_reg = run_model(data, label, build_3dconv_cvnn, "model_conv3dregular10_v1", regular_counter)
        raw_pred_aug = run_model(data, label, build_3dconv_cvnn, "model_conv3dregular10_augmented_v1", data_aug_counter)
        raw_pred_concat = run_model(data, label, build_concat3dconv_cvnn, "model_conv3dconcat10_v1", concat_counter)


        decision_tree_dataset["regular_pred"].append(list(raw_pred_reg))
        decision_tree_dataset["augmented_pred"].append(list(raw_pred_aug))
        decision_tree_dataset["concat_pred"].append(list(raw_pred_concat))
        decision_tree_dataset["labels"].append(np.argmax(label[0]))

        decision_tree_dataset_counter += 1
        if decision_tree_dataset_counter % 3 == 0:
            print "\nsaving decision tree dataset".upper(), len(decision_tree_dataset["labels"])
            with open(save_meta_model_path, "wb") as f:
                dump(decision_tree_dataset, f)

        naive_counter.update([label[0][average_prediction_vectors([raw_pred_reg, raw_pred_concat])] == 1])
        print "\nNAIVE AVERAGING"
        show_stats(naive_counter)
        model_predicted_index = predict_index(logistic_regression, raw_pred_reg, raw_pred_aug, raw_pred_concat)
        print "tree predicted index", model_predicted_index
        print "true label index", np.argmax(label[0])
        model_counter.update([label[0][model_predicted_index] == 1])

        print "\nMACHINE LEARNING PREDICTION"
        show_stats(model_counter)
        print "\n" * 2
