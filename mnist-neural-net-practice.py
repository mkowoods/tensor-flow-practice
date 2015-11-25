import numpy as np
import tensorflow as tf
import json


import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


sess = tf.InteractiveSession()

#TODO: needded to set-up misses
misses = {'data': [],
          'label': [],
          'guess': []}

x = tf.placeholder("float", [None, 784])
y_actuals = tf.placeholder("float", [None, 10])


W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#SoftMax Regression 
y_est = tf.nn.softmax(tf.matmul(x, W) + b)

#measuerment of error
cross_entropy = -tf.reduce_sum(y_actuals * tf.log(y_est))

#train model
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess.run(tf.initialize_all_variables())

for _ in range(1000):
    
    data, labels = mnist.train.next_batch(50)
    train_step.run(feed_dict = {x : data, y_actuals : labels})
    

def test_accuracy():
    label_est = sess.run(y_est, feed_dict = {x : mnist.test.images})
    acc_rate = np.equal(np.argmax(label_est, 1), np.argmax(mnist.test.labels, 1)).astype("float").mean()
    return acc_rate


    






    