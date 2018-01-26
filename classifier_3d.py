from random import shuffle
from cad_data_set_generator import prepare_data_set, prepare_data_set_smart_wrapper
from functools import reduce
import operator
import tensorflow as tf
import os
import sys
import pickle
from collections import Counter
import numpy as np

EPOCHS = 10

WINDOWS_SIZE = 2

SAVING_INTERVAL = 5

MEAN = 0.0

STDDEV = 0.05

FILTER_WIDTH = 3

FILTER_HEIGHT = 3

FILTER_DEPTH = 3

CHANNELS = 1

CAD_WIDTH = 30

CAD_HEIGHT = 30

CAD_DEPTH = 30

OUTPUT_SIZE = 8 # need to be 64

BATCH_SIZE = 12

LEARNING_RATE = BATCH_SIZE * 0.0001

TARGET_ERROR_RATE = 0.001

NUMBER_OF_TARGETS = 10

LIMIT = 2500

FC_NEURONS = 50 # need to be 2048

COST_FUNCTION = "cross"  #cross/sqr


def flatten(input_layer):
    input_size = input_layer.get_shape().as_list()
    new_size = input_size[-1] * input_size[-2] * input_size[-3]* input_size[-4]
    #new_size = reduce(operator.mul, input_size, 1)
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


def create_optimization(target_labels, dense_layer):
    if COST_FUNCTION == "sqr":
        cost = tf.squared_difference(target_labels, dense_layer)
    else:
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer, labels=target_labels)
    cost = tf.reduce_mean(cost)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
    return cost, optimizer


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
        training_set = list(prepare_data_set("train_cad", BATCH_SIZE, CHANNELS))
        print "shuffling data set"
        shuffle(training_set)
        print "caching data set"
        serialize(training_set, dump_path)
        return training_set


def predict(data, label, inputs, final_pred, prediction, mode):

    class_pred, raw_pred = sess.run([final_pred,prediction], feed_dict={inputs: data})
    print "raw prediction", raw_pred
    print "Predict class:",class_pred
    if mode == "test":
        return [label[0][class_pred[0]] == 1]
    assert len(label) == len(class_pred)
    return [label_vec[class_pred[i]] == 1 for i, label_vec in enumerate(label)]


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
        return mode
    except IndexError:
        print "\nUSAGE: python classifier_3d.py <train/test> \n"
        quit()


def show_stats(counter):
    stats = dict(counter)
    total = int()
    for k, v in stats.iteritems():
        print k ,":", v
        total += v
    print "Total:", total
    print "precision:", stats.get(True, 0) / float(total)


def create_dataset(dataset_path):
    data_set = list(prepare_data_set_smart_wrapper(dataset_path, BATCH_SIZE, CHANNELS, LIMIT))
    # data_set = smart_data_fetcher("dump_training_CADs")
    print "data set size:", len(data_set)
    shuffle(data_set)
    return data_set


def build_3dconv_nn(mode):

    inputs=tf.placeholder('float32', [BATCH_SIZE, CAD_DEPTH, CAD_HEIGHT, CAD_WIDTH, CHANNELS], name='Input')
    target_labels = tf.placeholder(dtype='float', shape=[BATCH_SIZE, NUMBER_OF_TARGETS], name="Targets")
    weight1 = tf.Variable(tf.random_normal(shape=[FILTER_DEPTH, FILTER_HEIGHT, FILTER_WIDTH, CHANNELS, OUTPUT_SIZE], stddev=STDDEV, mean=MEAN), name="Weight1")
    biases1 = tf.Variable(tf.random_normal([OUTPUT_SIZE], stddev=STDDEV, mean=MEAN), name='conv_biases')
    conv1 = tf.nn.conv3d(inputs, weight1, strides=[1, 1, 1, 1, 1], padding="SAME") + biases1
    relu1 = tf.nn.relu(conv1)
    maxpool1 = tf.nn.max_pool3d(relu1, ksize=[1, WINDOWS_SIZE, WINDOWS_SIZE, WINDOWS_SIZE, 1],
                                strides=[1, WINDOWS_SIZE, WINDOWS_SIZE, WINDOWS_SIZE, 1], padding="SAME")

    weight2 = tf.Variable(tf.random_normal(shape=[FILTER_DEPTH, FILTER_HEIGHT, FILTER_WIDTH, maxpool1.get_shape().as_list()[-1], OUTPUT_SIZE],
                                           stddev=STDDEV, mean=MEAN), name="Weight2")
    biases2 = tf.Variable(tf.random_normal([OUTPUT_SIZE], stddev=STDDEV, mean=MEAN), name='conv_biases')
    conv2 = tf.nn.conv3d(maxpool1, weight2, strides=[1, 1, 1, 1, 1], padding="SAME") + biases2
    relu2 = tf.nn.relu(conv2)
    weight3 = tf.Variable(tf.random_normal(shape=[FILTER_DEPTH, FILTER_HEIGHT, FILTER_WIDTH, relu2.get_shape().as_list()[-1], OUTPUT_SIZE],
                                           stddev=STDDEV, mean=MEAN), name="Weight3")
    biases3 = tf.Variable(tf.random_normal([OUTPUT_SIZE], stddev=STDDEV, mean=MEAN), name='conv_biases')
    conv3 = tf.nn.conv3d(relu2, weight3, strides=[1, 1, 1, 1, 1], padding="SAME") + biases3
    relu3 = tf.nn.relu(conv3)
    maxpool3 = tf.nn.max_pool3d(relu3, ksize=[1, WINDOWS_SIZE, WINDOWS_SIZE, WINDOWS_SIZE, 1],
                                strides=[1, WINDOWS_SIZE, WINDOWS_SIZE, WINDOWS_SIZE, 1], padding="SAME")
    dropout = tf.nn.dropout(maxpool3, 0.5)
    flat_layer1 = flatten(dropout)
    dense_layer1 = attach_dense_layer(flat_layer1, FC_NEURONS)
    relu4 = tf.nn.relu(dense_layer1)
    print "relu", relu4
    if mode == "test":
        relu4 =tf.reduce_max(relu4, axis=0, keep_dims=True)
        dense_layer2 = attach_dense_layer(relu4, NUMBER_OF_TARGETS)
        prediction = tf.nn.softmax(dense_layer2)
        final_pred = tf.argmax(prediction, axis=1)
        return inputs, target_labels, final_pred, prediction
    else:
        dense_layer2 = attach_dense_layer(relu4, NUMBER_OF_TARGETS)
        prediction = tf.nn.softmax(dense_layer2)

        final_pred =tf.argmax(tf.reshape(prediction, [BATCH_SIZE, NUMBER_OF_TARGETS]), axis=1)
        cost, optimizer = create_optimization(target_labels=target_labels,
                                          dense_layer=dense_layer2)


    return inputs, target_labels, cost, optimizer,final_pred, prediction


def run_session(data_set, cost, optimizer,final_pred, prediction, inputs, target_labels, mode, epochs):
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
            for batch in data_set:
                data, label = batch[0], batch[1]
                err, _ = sess.run([cost, optimizer], feed_dict={inputs: data, target_labels: label})
                print "error rate:", str(err)
                step += 1
                if step % SAVING_INTERVAL == 0:
                    print "saving model..."
                    saver.save(sess, model_save_path + model_name)
                    print "model saved"
                    counter.update(predict(data, label, inputs, final_pred, prediction, mode))
                    show_stats(counter)

    elif mode == "test":
        for batch in data_set:
            data, label = batch[0], batch[1]
            counter.update(predict(data, label, inputs, final_pred, prediction, mode))
            show_stats(counter)
    else:
        raise Exception("invalid mode")


if __name__ == "__main__":
    mode = parse_flags()
    if mode == "train":
        inputs, target_labels, cost, optimizer, final_pred, prediction = build_3dconv_nn(mode)
        print "Train Dataset"
        data_set = create_dataset("train_cad_10.tar.gz")
        with tf.Session() as sess:
            run_session(data_set, cost, optimizer, final_pred, prediction, inputs, target_labels, mode, EPOCHS)
    else:
        inputs, target_labels, final_pred, prediction = build_3dconv_nn(mode)
        print "Test Dataset"
        data_set = create_dataset("test_cad_10.tar.gz")
        with tf.Session() as sess:
            run_session(data_set, [], [], final_pred, prediction, inputs, target_labels, mode, EPOCHS)
