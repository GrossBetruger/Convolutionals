import datetime
from random import shuffle
from cad_data_set_generator import prepare_training_set
from functools import reduce
import operator
import tensorflow as tf
import os
import sys
import pickle
from collections import Counter

2500 = 2500

EPOCHS = 10

WINDOWS_SIZE = 2

SAVING_INTERVAL = 100

MEAN = 0.0

STDDEV = 0.05

FILTER_WIDTH = 5

FILTER_HEIGHT = 5

FILTER_DEPTH = 5

CHANNELS = 1

CAD_WIDTH = 30

CAD_HEIGHT = 30

CAD_DEPTH = 30

OUTPUT_SIZE = 8

LEARNING_RATE = 0.2

TARGET_ERROR_RATE = 0.001

BATCH_SIZE = 1

NUMBER_OF_TARGETS = 2

LIMIT = 2500



def flatten(input_layer):
    input_size = input_layer.get_shape().as_list()
    new_size = reduce(operator.mul, input_size, 1)
    return tf.reshape(input_layer, [-1, new_size])


def attach_dense_layer(input_layer, size, summary=False):
    input_size = input_layer.get_shape().as_list()[-1]
    weights = tf.Variable(tf.random_normal([input_size, size], stddev=STDDEV, mean=MEAN), name='dense_weigh')
    if summary:
        tf.summary.histogram(weights.name, weights)
    biases = tf.Variable(tf.random_normal([size], stddev=STDDEV, mean=MEAN), name='dense_biases')
    dense = tf.matmul(input_layer, weights) + biases
    return dense


def attach_sigmoid_layer(input_layer):
    return tf.nn.sigmoid(input_layer)


def create_optimization(target_labels, dense_layer, naive=False):
    if naive:
        cost = tf.squared_difference(target_labels, dense_layer)
    else:
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer, labels=target_labels)
    cost = tf.reduce_mean(cost)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
    return cost, optimizer


def smooth(proba):
    return 0 if proba < .5 else 1


def to_pred(probas):
    return [smooth(pred) for pred in probas]


def is_certain(probas, confidence):
    return any(x >= confidence for x in probas)


def serialize(data, fname):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def unserialize(fname):
    with open(fname) as f:
        return pickle.load(f)


def smart_data_fetcher(dump_path):
    print "generating data set, please wait..."
    if os.path.exists(dump_path):
        print "fetching from cache..."
        return unserialize(dump_path)
    else:
        print "creating training set..."
        training_set = list(prepare_training_set("train_cad", BATCH_SIZE, CHANNELS))
        print "shuffling data set"
        shuffle(training_set)
        print "caching data set"
        serialize(training_set, dump_path)
        return training_set


def predict(data, label, inputs, prediction):
    raw_pred = sess.run([prediction], feed_dict={inputs: data})[0][0]
    print "raw prediction", raw_pred
    pred = to_pred(raw_pred)
    target = list(label[0])
    print "LABELS:", target
    return pred == target


def print_model(sess):
    print "Model Variables"
    for var in sess.graph.get_collection('variables'):
        print var
    print
    print "Model Trainable Variables"
    for trainable in sess.graph.get_collection('trainable_variables'):
        print trainable
    print
    print "Model Train Optimizers"
    for train_op in sess.graph.get_collection('train_op'):
        print train_op


def parse_flags():
    try:
        mode = sys.argv[1]
        optimization_model = sys.argv[2]
        return mode, optimization_model
    except IndexError:
        print "\nUSAGE: python classifier_3d.py <train/test> <naive/cross>\n"
        quit()


def show_stats(counter):
    stats = dict(counter)
    total = int()
    for k, v in stats.iteritems():
        print k, v
        total += v
    print "precision:", stats.get(True, 0) / float(total)


def create_dataset():
    training_set = list(prepare_training_set("train_cad", BATCH_SIZE, CHANNELS, LIMIT))
    # training_set = smart_data_fetcher("dump_training_CADs")
    print "training set size:", len(training_set)
    shuffle(training_set)
    return training_set


def build_3dconv_nn(optimization_model):
    inputs=tf.placeholder('float32', [BATCH_SIZE, CAD_DEPTH, CAD_HEIGHT, CAD_WIDTH, CHANNELS], name='Input')
    # maybe simplify targets placeholder
    target_labels = tf.placeholder(dtype='float', shape=[None, NUMBER_OF_TARGETS], name="Targets")
    # maybe depth of filter should be 30
    weight1 = tf.Variable(tf.random_normal(shape=[FILTER_DEPTH, FILTER_HEIGHT, FILTER_WIDTH, CHANNELS, OUTPUT_SIZE], stddev=STDDEV, mean=MEAN), name="Weight1")
    biases1 = tf.Variable(tf.random_normal([OUTPUT_SIZE], stddev=STDDEV, mean=MEAN), name='conv_biases')
    conv1 = tf.nn.conv3d(inputs, weight1, strides=[1, 1, 1, 1, 1], padding="SAME") + biases1
    relu1 = tf.nn.relu(conv1)
    # skipping maxpool
    maxpool1 = tf.nn.max_pool3d(relu1, ksize=[1, WINDOWS_SIZE, WINDOWS_SIZE, WINDOWS_SIZE, 1],
                                strides=[1, WINDOWS_SIZE, WINDOWS_SIZE, WINDOWS_SIZE, 1], padding="SAME")

    weight2 = tf.Variable(tf.random_normal(shape=[FILTER_DEPTH, FILTER_HEIGHT, FILTER_WIDTH, maxpool1.get_shape().as_list()[-1], OUTPUT_SIZE],
                                           stddev=STDDEV, mean=MEAN), name="Weight2")
    biases2 = tf.Variable(tf.random_normal([OUTPUT_SIZE], stddev=STDDEV, mean=MEAN), name='conv_biases')
    conv2 = tf.nn.conv3d(maxpool1, weight2, strides=[1, 1, 1, 1, 1], padding="SAME") + biases2
    relu2 = tf.nn.relu(conv2)
    # skipping maxpool

    #
    weight3 = tf.Variable(tf.random_normal(shape=[FILTER_DEPTH, FILTER_HEIGHT, FILTER_WIDTH, relu2.get_shape().as_list()[-1], OUTPUT_SIZE],
                                           stddev=STDDEV, mean=MEAN), name="Weight3")
    biases3 = tf.Variable(tf.random_normal([OUTPUT_SIZE], stddev=STDDEV, mean=MEAN), name='conv_biases')
    conv3 = tf.nn.conv3d(relu2, weight3, strides=[1, 1, 1, 1, 1], padding="SAME") + biases3
    relu3 = tf.nn.relu(conv3)
    maxpool3 = tf.nn.max_pool3d(relu3, ksize=[1, WINDOWS_SIZE, WINDOWS_SIZE, WINDOWS_SIZE, 1],
                                strides=[1, WINDOWS_SIZE, WINDOWS_SIZE, WINDOWS_SIZE, 1], padding="SAME")


    # fully_connected1 = tf.contrib.layers.fully_connected(inputs=relu1, num_outputs=number_of_targets)
    flat_layer1 = flatten(maxpool3)
    dense_layer1 = attach_dense_layer(flat_layer1, 50)

    # sigmoid2 = attach_sigmoid_layer(flat_layer1)
    relu2 = tf.nn.relu(dense_layer1)
    dense_layer2 = attach_dense_layer(relu2, NUMBER_OF_TARGETS)
    prediction = tf.nn.softmax(dense_layer2)
    # prediction = attach_sigmoid_layer(dense_layer2)



    naive_optimization = True if optimization_model == "naive" else False
    cost, optimizer = create_optimization(target_labels=target_labels,
                                          dense_layer=dense_layer2, naive=naive_optimization)


    return inputs, target_labels, cost, optimizer, prediction


def run_session(training_set, cost, optimizer, prediction, inputs, target_labels, mode, epochs):
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    model_save_path = "./model_conv3d_v1/"
    model_name = 'CAD_Classifier'

    if os.path.exists(model_save_path + 'checkpoint'):
        saver.restore(sess, tf.train.latest_checkpoint(model_save_path))

    print_model(sess)
    step = 1
    counter = Counter()
    if mode == "train":
        for epoch in range(epochs):
            for data, label in training_set:
                    err, _ =  sess.run([cost, optimizer],feed_dict={inputs: data, target_labels: label})
                    print "error rate:", str(err)
                    step += 1
                    if step % SAVING_INTERVAL == 0:
                        print "saving model..."
                        saver.save(sess, model_save_path + model_name)
                        print "model saved"
                        counter.update([predict(data, label, inputs, prediction)])
                        show_stats(counter)

    elif mode == "test":
        for data, label in training_set:
            predict(data, label, inputs, prediction)
    else:
        raise Exception("invalid mode")


if __name__ == "__main__":
    mode, optimization_model = parse_flags()
    inputs, target_labels, cost, optimizer, prediction = build_3dconv_nn(optimization_model)
    training_set = create_dataset()
    with tf.Session() as sess:
        run_session(training_set, cost, optimizer, prediction, inputs, target_labels, mode, EPOCHS)