import input_data
#import pylab as plt
import tensorflow as tf




mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None,784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#estimate of y
y_hat = tf.nn.softmax(tf.matmul(x, W) + b)


#known y
y = tf.placeholder("float", [None,10])

cross_entropy = -tf.reduce_sum(y * tf.log(y_hat))

train_step = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



init = tf.initialize_all_variables()

final_weights = None

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict = {x: batch_xs, y: batch_ys})

    print sess.run(accuracy, feed_dict = {x : mnist.test.images, y : mnist.test.labels})
    



    