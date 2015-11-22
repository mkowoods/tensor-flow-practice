import numpy as np
import tensorflow as tf
import json


import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_tr, y_tr = mnist.train.next_batch(5000)
x_te, y_te = mnist.test.next_batch(200)

xtr_node = tf.placeholder("float", [None, 784])
xte_node = tf.placeholder("float", [784])



distance = tf.reduce_sum(tf.abs(tf.sub(xtr_node, xte_node)), reduction_indices= 1)

pred = tf.argmin(distance, dimension = 0)

accuracy = 0.0


misses = {'data': [],
          'label': [],
          'guess': []}

with tf.Session() as sess:
    tf.initialize_all_variables()
    num_samples = x_te.shape[0]
    
    for i in range(num_samples):
        nn_idx = sess.run(pred, feed_dict = {xtr_node : x_tr, xte_node : x_te[i, :]})

        predict_ = np.argmax(y_tr[nn_idx, :])
        actual_ = np.argmax(y_te[i, :])
        
        if predict_ == actual_:
            accuracy += 1.0/num_samples
        else:
            misses['data'].append(x_te[i, :].tolist())
            misses['label'].append(actual_)
            misses['guess'].append(predict_)
            
        
        print 'Test:', i, 'Prediction:', predict_, 'Actuals: ', actual_
    
    print accuracy


def write_misses(misses):
    json.dump(misses, open('misses.json', 'wb'))
    
