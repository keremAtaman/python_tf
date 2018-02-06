# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:12:52 2018

@author: K.Ataman
"""

import tensorflow as tf
import numpy as np

training_epochs = 100
learning_rate = 0.01
batch_size = 100
num_entries = 1000
num_features_input = 3
num_features_output = 2

def create_dataset(num_entries, num_features_input, num_features_output):
    x = np.random.rand(num_entries, num_features_input)
    y = np.zeros([num_entries,num_features_output])
    y_divisor = 1 / num_features_output
    for i in range(num_entries):
        y_entry = 0
        for j in range(num_features_input):
            if j%2==0:
                y_entry += x[i, j]
            else:
                y_entry = y_entry*x[i, j]
        for j in range(1, num_features_output + 1):
            if y_entry < j * y_divisor:
                y[i, j - 1] = 0
            else:
                y[i, j - 1] = 1
    return [x,y]

def get_batch(batch, batch_size, x, y):
    x = x[batch*batch_size : (batch + 1) * batch_size]
    y = y[batch*batch_size : (batch + 1) * batch_size]
    return [x,y]
    
x = tf.placeholder("float", [None, num_features_input])
y = tf.placeholder("float", [None, num_features_output])

W = tf.Variable(tf.zeros([num_features_input, num_features_output]))
b = tf.Variable(tf.zeros([num_features_output]))

result = tf.matmul(x, W) + b
activation = tf.nn.softmax(result)
cross_entropy = y * tf.log(activation)
#reduce_mean just calculates the sum, don't worry, no reduction is invovled
cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy, reduction_indices = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

avg_set = []
epoch_set = []

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0
        #create new dataset for each epoch to avoid overfitting
        x_, y_ = create_dataset(num_entries, num_features_input, num_features_output)
        for batch in range(num_entries // batch_size):
            batch_x, batch_y = get_batch(batch, batch_size, x_ , y_)
            #Notice how "optimizer", which is the variable we want, is the first entry
            sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})
            avg_cost+= sess.run(cost, feed_dict = {x: batch_x, y: batch_y}) / batch_size
        print ("Epoch: ", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost))
        avg_set.append(avg_cost)