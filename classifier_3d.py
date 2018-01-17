import datetime
from random import shuffle
from cad_data_set_generator import prepare_training_set
import tensorflow as tf
import numpy as np
import os


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')

    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

TARGET_ERROR_RATE = 0.001
batch_size = 1
number_of_targets = 2
inputs=tf.placeholder('float32',[batch_size, 30, 30, 30, 1],name='Input')
target_labels = tf.placeholder(dtype='float', shape=[None, number_of_targets], name="Targets")


output_size = 64
weight1 = tf.Variable(tf.random_normal(shape=[5, 5, 5, 1, output_size], stddev=0.02),name="Weight1")
biases1 = tf.Variable(tf.random_normal([output_size]),name='conv_biases')
conv1 = tf.nn.conv3d(inputs, weight1, strides=[1, 1, 1, 1, 1], padding="SAME") + biases1
relu1 = tf.nn.relu(conv1)
# skipping maxpool
maxpool1 = tf.nn.max_pool3d(relu1, ksize=[2, 2, 2, output_size, output_size], strides=[1, 1, 1, 1, 1], padding="SAME")

fully_connected1 = tf.contrib.layers.fully_connected(inputs=relu1, num_outputs=number_of_targets)
softmax1 = tf.nn.softmax(fully_connected1)


# cost=tf.nn.softmax_cross_entropy_with_logits(logits=fully_connected1, labels=target_labels)
# cost=tf.reduce_mean(cost)
# optimizer=tf.train.AdamOptimizer().minimize(cost)

cost=tf.squared_difference(target_labels, softmax1)
cost=tf.reduce_mean(cost)
optimizer=tf.train.AdamOptimizer().minimize(cost)


print "generating data set, this may take a while..."
training_set = list(prepare_training_set("train_cad"))
print "shuffling data set"
shuffle(training_set)

saver = tf.train.Saver()
model_save_path="./model_3d_conv/"
model_name='CADClassifier'



with tf.Session() as sess:
    tf.global_variables_initializer().run()

    filename = "./summary_log_CAD/run" + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%s")

    if os.path.exists(model_save_path + 'checkpoint'):
        # saver = tf.train.import_meta_graph('./saved '+modelName+'/model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_save_path))

    writer = tf.summary.FileWriter(filename, sess.graph)

    step = 1
    for data, label in training_set:
        # print label
        # print [[data]]*10
        data = np.array(data)
        data = [data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)] * batch_size

        print "\ntraining... step: ", step
        print "labels:", [label] * batch_size
        err, _ =  sess.run([cost, optimizer],feed_dict={inputs: data, target_labels: [label] * batch_size})
        print "error rate:", err
        step += 1
        if step % 3 == 0:
            print "saving model..."
            saver.save(sess, model_save_path + model_name)
            print "model saved"

        # print(i,error)
