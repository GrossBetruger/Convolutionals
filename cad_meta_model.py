import tensorflow as tf
from classifier_3d import create_dataset, build_3dconv_cvnn, show_stats
import os
import numpy as np
from collections import Counter
from pickle import dump, load
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def average_prediction_vectors(lst_of_raw_predictions):
    first_pred = np.array(lst_of_raw_predictions[0])
    for raw_pred in lst_of_raw_predictions[1:]:
        first_pred += np.array(raw_pred)
    return np.argmax(first_pred / float(len(lst_of_raw_predictions)))


def restore_model(model_save_path, session):
    if os.path.exists(model_save_path + 'checkpoint'):
        saver.restore(session, tf.train.latest_checkpoint(model_save_path))


def train_high_level_model(dataset, classifier):
    X1 = list(dataset["regular_pred"])
    X2 = list(dataset["concat_pred"])
    assert len(X1) == len(X2)
    X = [X1[i] + X2[i] for i in range(len(X1))]
    classifier.fit(X, dataset["labels"])
    return classifier


def predict_index(model):
    return model.predict([list(raw_pred_reg) + list(raw_pred_concat)])[0]


if __name__ == "__main__":
    with open("desicion_tree_training_set.json") as f:
        decision_tree_dataset = load(f)

    forest = train_high_level_model(decision_tree_dataset, RandomForestClassifier())
    tree = train_high_level_model(decision_tree_dataset, DecisionTreeClassifier())
    logistic_regression = train_high_level_model(decision_tree_dataset, LogisticRegression())
    svm = train_high_level_model(decision_tree_dataset, SVC())

    print "done training"
    mode = "test"
    inputs, target_labels, final_pred, prediction = build_3dconv_cvnn(mode)
    print "Test Dataset"
    data_set = create_dataset("test_cad_10.tar.gz")

    saver = tf.train.Saver()

    counter = Counter()
    tree_counter = Counter()

    decision_tree_dataset = {"regular_pred": [],
                               "concat_pred": [],
                               "labels": []}

    decision_tree_dataset_counter = int()

    for batch in data_set:
        data, label = batch[0], batch[1]
        with tf.Session() as regular_session:
            tf.global_variables_initializer().run()
            model_save_path = "model_conv3dregular10_v1"
            restore_model(model_save_path, regular_session)

            raw_pred_reg = regular_session.run([prediction], feed_dict={inputs: data})[0][0]
            print "raw regular"
            print raw_pred_reg

        with tf.Session() as concat_session:
            tf.global_variables_initializer().run()
            model_save_path = "model_conv3dconcat10_v1"
            restore_model(model_save_path, concat_session)
            raw_pred_concat = concat_session.run([prediction], feed_dict={inputs: data})[0][0]
            print "raw concat"
            print raw_pred_concat

        decision_tree_dataset["regular_pred"].append(list(raw_pred_reg))
        decision_tree_dataset["concat_pred"].append(list(raw_pred_concat))
        decision_tree_dataset["labels"].append(np.argmax(label[0]))
        decision_tree_dataset_counter += 1
        if decision_tree_dataset_counter % 3 == 0:
            pass
            # print "saving desition tree dataset", len(decision_tree_dataset["labels"])
            # with open("desicion_tree_training_set.json", "wb") as f:
            #     dump(decision_tree_dataset, f)
        # print "Average Prediction:"
        counter.update([label[0][average_prediction_vectors([raw_pred_reg, raw_pred_concat])] == 1])
        print "NAIVE AVERAGING"
        show_stats(counter)
        model_predicted_index = predict_index(logistic_regression)
        print "tree predicted index", model_predicted_index
        print "true label index", np.argmax(label[0])
        tree_counter.update([label[0][model_predicted_index] == 1])

        print "MACHINE LEARNING PREDICTION"
        show_stats(tree_counter)
