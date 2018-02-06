# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:41:24 2018

@author: K.Ataman
"""

import tensorflow as tf

lstm = tf.nn.rnn_cell.BasicLSTMCell(size)
state = tf.zeros([batch_size, lstm.state_size])
loss = 0.0
for i in dataset:
    output, state = lstm(i, state)
    logits = tf.matmul(output, w) + b
    probabilities = tf.nn.softmax(logits)
    loss += loss_function(probabilities, y)
    
