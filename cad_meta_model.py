import tensorflow as tf
from classifier_3d import create_dataset, build_3dconv_cvnn
from functools import reduce
import numpy as np


def average_prediction_vectors(lst_of_raw_predictions):
    first_pred = np.array(lst_of_raw_predictions[0])
    for raw_pred in lst_of_raw_predictions[1:]:
        first_pred += np.array(raw_pred)
    return first_pred / float(len(lst_of_raw_predictions))




if __name__ == "__main__":
    mode = "test"
    inputs, target_labels, final_pred, prediction = build_3dconv_cvnn(mode)
    print "Test Dataset"
    data_set = create_dataset("test_cad_10.tar.gz")

    for batch in data_set:
        data, label = batch[0], batch[1]

        with tf.Session() as regular_session:
            tf.global_variables_initializer().run()

            raw_pred_reg = regular_session.run([prediction], feed_dict={inputs: data})[0][0]
            print "raw regular"
            print raw_pred_reg

        with tf.Session() as concat_session:
            tf.global_variables_initializer().run()

            raw_pred_concat = concat_session.run([prediction], feed_dict={inputs: data})[0][0]
            print "raw concat"
            print raw_pred_concat

        print "Average Prediction:"
        print average_prediction_vectors([raw_pred_reg, raw_pred_concat])
