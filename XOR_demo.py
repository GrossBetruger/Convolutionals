import tensorflow as tf
import numpy as np
import os

TARGET_ERROR_RATE = 0.001
SAVING_INTERVAL = 50

inputs=tf.placeholder('float',[None,2],name='Input')
targets=tf.placeholder('float',name='Target')

weight1=tf.Variable(tf.random_normal(shape=[2,3],stddev=0.02),name="Weight1")
biases1=tf.Variable(tf.random_normal(shape=[3],stddev=0.02),name="Biases1")

hLayer=tf.matmul(inputs,weight1)
hLayer=hLayer+biases1

hLayer=tf.sigmoid(hLayer, name='hActivation')

weight2=tf.Variable(tf.random_normal(shape=[3,1],stddev=0.02),name="Weight2")
biases2=tf.Variable(tf.random_normal(shape=[1],stddev=0.02),name="Biases2")

output=tf.matmul(hLayer,weight2)
output=output+biases2
output=tf.sigmoid(output, name='outActivation')

cost=tf.squared_difference(targets, output)
cost=tf.reduce_mean(cost)
optimizer=tf.train.AdamOptimizer().minimize(cost)

inp=[[0,0],[0,1],[1,0],[1,1]]
out=[[0],[1],[1],[0]]

inp=np.array(inp)
out=np.array(out)


saver = tf.train.Saver()
model_save_path="./model_xor_simple_nn_v1/"
model_name='XOR_Classifier'


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    if os.path.exists(model_save_path + 'checkpoint'):
        saver.restore(sess, tf.train.latest_checkpoint(model_save_path))
    error = 1
    i = 0
    while error > TARGET_ERROR_RATE:
        error, _ =sess.run([cost,optimizer],feed_dict={inputs: inp,targets:out})
        print(i,error)
        i += 1
        if i % SAVING_INTERVAL == 0:
            print "saving model..."
            saver.save(sess, model_save_path + model_name)
            print "model saved"

    while True:
        a = input("type 1st input :")
        b = input("type 2nd input :")
        inp = [[a, b]]
        inp = np.array(inp)
        prediction = sess.run([output], feed_dict={inputs: inp})
        print(prediction)