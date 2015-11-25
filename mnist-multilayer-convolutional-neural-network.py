import tensorflow as tf
import numpy as np
import json
import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



def weight_variable(shape):
    init = tf.truncated_normal(shape = shape, stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.constant(value = 0.1, shape = shape)
    return tf.Variable(init)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_poll_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')

#image vector

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

#reshape the image into a 1x28x28x1 tensor
x_image = tf.reshape(x, [-1, 28, 28, 1])



#first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32]) #create a Weight Tensor that maps a 5x5 matrix to 32 features
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #input [None, 28, 28, 1]; output [None, 28, 28, 32]
h_pool1 = max_poll_2x2(h_conv1) #input [None, 28, 28, 32];  output [None, 14, 14, 32]


#second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #output [None, 14, 14, 64]
h_pool2 = max_poll_2x2(h_conv2) #output [None, 7, 7, 64]


#Densely Connected Layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64]) #returns matrix that is [None, 7*7*64]

W_fc1 = weight_variable([7 * 7 * 64, 1024]) #fully connected layer maps 7*7*64 -> 1024
b_fc1 = bias_variable([1024]) #bias for fully connected layer


h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob = keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)



sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

import time

start = time.time()


# recommended 20K iterations for testing
# should include save function to let the model save results
# should also use logging functiokn

for i in range(3000):
    
    data, labels = mnist.train.next_batch(50)

    if i % 50 == 0:
        
        y_est = sess.run(y_conv, feed_dict = {keep_prob : 1.0, x : data})
        
        acc = (np.argmax(y_est, 1) == np.argmax(labels, 1)).mean()
        
        print 'Iteration', i, 'Acc:', acc, 'time', time.time() - start
        start = time.time()
        
    sess.run(train_step, feed_dict = {keep_prob : 0.5, x : data, y : labels})
    
    if i % 1000 == 0:
        y_est = sess.run(y_conv, feed_dict={keep_prob : 1.0, x : mnist.test.images})
        acc_est = (np.argmax(y_est, 1) == np.argmax(mnist.test.labels, 1)).mean()
        print 'Iter', i,  'Test Accuracy', acc_est
        
    