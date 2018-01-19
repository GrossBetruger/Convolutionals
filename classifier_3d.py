import datetime
from random import shuffle
from cad_data_set_generator import prepare_training_set
from functools import reduce
import operator
import tensorflow as tf
import os
import sys

SAVING_INTERVAL = 10

MEAN = 0.0

STDDEV = 0.05

FILTER_WIDTH = 5

FILTER_HEIGHT = 5

FILTER_DEPTH = 5

CHANNELS = 3

CAD_WIDTH = 30

CAD_HEIGHT = 30

CAD_DEPTH = 30

OUTPUT_SIZE = 64

LEARNING_RATE = 0.4


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')

    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def flatten(input_layer):
    # with tf.name_scope("Flatten") as scope:
    input_size = input_layer.get_shape().as_list()
    new_size = reduce(operator.mul, input_size, 1)
    return tf.reshape(input_layer, [-1, new_size])


def attach_dense_layer(input_layer, size, summary=False):
    # with tf.name_scope("Dense") as scope:
    input_size = input_layer.get_shape().as_list()[-1]
    weights = tf.Variable(tf.random_normal([input_size, size], stddev=STDDEV, mean=MEAN), name='dense_weigh')
    if summary:
        tf.summary.histogram(weights.name, weights)
    biases = tf.Variable(tf.random_normal([size], stddev=STDDEV, mean=MEAN), name='dense_biases')
    dense = tf.matmul(input_layer, weights) + biases
    return dense


def attach_sigmoid_layer(input_layer):
    # with tf.name_scope("Activation") as scope:
    return tf.nn.sigmoid(input_layer)


def smooth(proba):
    return 0 if proba < .5 else 1


def to_pred(probas):
    return [smooth(pred) for pred in probas]


def is_certain(probas, confidence):
    return any(x >= confidence for x in probas)


TARGET_ERROR_RATE = 0.001
batch_size = 1
number_of_targets = 2
inputs=tf.placeholder('float32', [batch_size, CAD_DEPTH, CAD_HEIGHT, CAD_WIDTH, CHANNELS], name='Input')
# maybe simplify targets placeholder
target_labels = tf.placeholder(dtype='float', shape=[None, number_of_targets], name="Targets")
# maybe depth of filter should be 30
weight1 = tf.Variable(tf.random_normal(shape=[FILTER_DEPTH, FILTER_HEIGHT, FILTER_WIDTH, CHANNELS, OUTPUT_SIZE], stddev=STDDEV, mean=MEAN), name="Weight1")
biases1 = tf.Variable(tf.random_normal([OUTPUT_SIZE], stddev=STDDEV, mean=MEAN), name='conv_biases')
conv1 = tf.nn.conv3d(inputs, weight1, strides=[1, 1, 1, 1, 1], padding="SAME") + biases1
relu1 = tf.nn.relu(conv1)
# skipping maxpool
# maxpool1 = tf.nn.max_pool3d(relu1, ksize=[2, 2, 2, OUTPUT_SIZE, OUTPUT_SIZE], strides=[1, 2, 2, 2, 1], padding="SAME")

# fully_connected1 = tf.contrib.layers.fully_connected(inputs=relu1, num_outputs=number_of_targets)
flat_layer1 = flatten(relu1)
dense_layer1 = attach_dense_layer(flat_layer1, 32)

# sigmoid2 = attach_sigmoid_layer(flat_layer1)
relu2 = tf.nn.relu(dense_layer1)
dense_layer2 = attach_dense_layer(relu2, number_of_targets)
softmax1 = tf.nn.softmax(dense_layer2)


cost=tf.nn.softmax_cross_entropy_with_logits(logits=softmax1, labels=target_labels)
cost=tf.reduce_mean(cost)
optimizer=tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

# cost=tf.squared_difference(target_labels, softmax1)
# cost=tf.reduce_mean(cost)
# optimizer=tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)


try:
    mode = sys.argv[1]
except IndexError:
    print "\nUSAGE: python classifier_3d.py <train/test>\n"
    quit()

print "generating data set, this may take a while..."
training_set = list(prepare_training_set("train_cad", batch_size, CHANNELS))
print "shuffling data set"
shuffle(training_set)

saver = tf.train.Saver()
model_save_path="./model_3d_conv_v11/"
model_name='CADClassifier'



with tf.Session() as sess:
    tf.global_variables_initializer().run()

    filename = "./summary_log_CAD/run" + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%s")

    if os.path.exists(model_save_path + 'checkpoint'):
        saver.restore(sess, tf.train.latest_checkpoint(model_save_path))

    writer = tf.summary.FileWriter(filename, sess.graph)

    step = 1
    if mode == "train":

        for data, label in training_set:
                # print "\ntraining... step: ", step
                # print "labels:", label
                err, _ =  sess.run([cost, optimizer],feed_dict={inputs: data, target_labels: label})
                print "error rate:", str(err)
                step += 1
                if step % SAVING_INTERVAL == 0:
                    print "saving model..."
                    saver.save(sess, model_save_path + model_name)
                    print "model saved"

    elif mode == "test":
        true_count = int()
        false_count = int()
        for data, label in training_set:
            print "\ntesting... step: ", step
            target = "labels:", label[0]

            raw_pred = sess.run([softmax1], feed_dict={inputs: data})[0][0]
            print "raw prediction", raw_pred
            pred = to_pred(raw_pred)

            if pred == target:
                true_count += 1
            else:
                false_count += 1

            print "true count", true_count
            print "false count", false_count
            total = false_count + true_count
            print "precision", float(true_count) / total

            step += 1

    else:
        raise Exception("invalid mode")
