import tensorflow as tf
from classifier_3d import create_dataset, build_3dconv_cvnn




if __name__ == "__main__":
    mode = "test"
    inputs, target_labels, final_pred, prediction = build_3dconv_cvnn(mode)
    print "Test Dataset"
    data_set = create_dataset("test_cad_10.tar.gz")

    for batch in data_set:
        data, label = batch[0], batch[1]

        with tf.Session() as regular_session:
            tf.global_variables_initializer().run()

            raw_pred_reg = regular_session.run([prediction], feed_dict={inputs: data})
            print "raw regular"
            print raw_pred_reg
            regular_session.close()

        with tf.Session() as concat_session:
            tf.global_variables_initializer().run()

            raw_pred_concat = concat_session.run([prediction], feed_dict={inputs: data})
            print "raw concat"
            print raw_pred_concat