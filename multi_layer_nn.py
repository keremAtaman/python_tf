# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:09:08 2018

@author: K.Ataman
"""
#Two layered simple NN
#feel free to compare the performance of this to basic_nn ' s cost

import tensorflow as tf
import numpy as np

num_hidden_1 = 256
num_hidden_2 = num_hidden_1
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
#first layer
w_1 = tf.Variable(tf.random_normal([num_features_input, num_hidden_1]))
b_1 = tf.Variable(tf.random_normal([num_hidden_1]))
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w_1), b_1))
#second layer
w_2 = tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2]))
b_2 = tf.Variable(tf.random_normal([num_hidden_2]))
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w_2), b_2))
#output layer
w_o = tf.Variable(tf.random_normal([num_hidden_2, num_features_output]))
b_o    = tf.Variable(tf.random_normal([num_features_output]))
layer_o = tf.add(tf.matmul(layer_2, w_o), b_o)
#cost and optimization
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = layer_o, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

avg_set = []

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