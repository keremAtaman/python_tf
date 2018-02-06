# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:27:29 2018

@author: K.Ataman
"""

"""
Convolutional Neural Network uses three main ideas:
    Local receptive fields:
        The smaller area of the picture being multiplied by the kernel
    Convolution:
        Using smaller "Kernel" multipliers (ie if a pic is 25 by 25, the kernel 
        can be 2x2 etc). In our case the convolution is getting a small area of
        the pic and convolving it (aka we shift the area we do operations with, 
        going from top left all the way to bottom right
        with a kernel to get output
        
        Instead of each pixel getting a weihgt, the convolution will get a 
        weight
    Pooling:
        Takes convolved output and simplifying it
    Local connectivity:
        How "connected" or related one datapoint is to another
        
    Example:
        We have 28*28 pic
        We have a 5x5 convo layer and 3 feature maps
        We have 3 feature maps (converts complex output to simplified output,
                                such as "this is a line")
        We have 3*(28-5+1)*(28-5+1) = 3*24*24 sized hidden layer
        The last layer is fully connected, aka it connects all neurons of the
            max pooling layer to all 10 output neurons, which are used to 
            recognize the outtput
"""

import tensorflow as tf

learning_rate = 0.001
num_training_steps = 1000
batch_size = 128
#input size does not contain color channels
input_size = [28, 28]
num_color_channels = 1
num_feature_maps = 32
reduction_factor = 2
#in our case, this is the number of digits [0-9]
num_features_output = 10
kernel_size = [5, 5]
#Drop out units in hidden, input and output to prevent overfitting.
#Essentially eliminates some neurons
dropout = 0.75

'''
Functions
'''
#tf.nn.relu is activation for cont but not always differentiable functions
#tf.nn.bias_add adds the bias (b in our case) where b is 1D. This is a 
    #special case of tf.add for 1D only
#tf.nn.conv2d computes 2D convo given 4D input and filter
    #Here, w is filter
    #Stride controls how depth columns around the spatial dimensions 
        #(width and height) are allocated. When the stride is 1 then we 
        #move the filters one pixel at a time. So we are moving by 1 in
        #each dimension

def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add\
                      (tf.nn.conv2d(img, w,\
                                    strides=[1, 1, 1, 1],\
                                    padding='SAME'),b))

def max_pool(img, k):
    return tf.nn.max_pool(img, \
                          ksize=[1, k, k, 1],\
                          strides=[1, k, k, 1],\
                          padding='SAME')
    
'''
tf stuff
'''

x_ = tf.placeholder(tf.float32, [None, input_size[0] * input_size[1]])
#reshape data, where -1 is "fill it for me"
#output is 4d where the dimensions are None * height * width * \
#                               num color channels (1 in our case)
x = tf.reshape(x_, shape = [-1, input_size[0], input_size[1], 
                            num_color_channels])
y = tf.placeholder(tf.float32, [None, num_features_output])



#fist convolutional layer
w_1 = tf.Variable(tf.random_normal([kernel_size[0], kernel_size[1], 1, 
                                    num_feature_maps]))
b_1 = tf.Variable(tf.random_normal([num_feature_maps]))
#This essentially computes a 2D convo 
conv_1 = conv2d(x, w_1, b_1)
#take k x k region of convo layer and summarize it using pooling
#after pooling, the dimension of image gets reduced by k. So if k is 2
#and convo result is 28*28*3, layers are now 14*14*3
conv_1 = max_pool(conv_1, k = reduction_factor)
#do the dropout
keep_prob = tf. placeholder(tf.float32)
conv_1 = tf.nn.dropout(conv_1,keep_prob)

#second convo layer
w_2 = tf.Variable(tf.random_normal([kernel_size[0], kernel_size[1],
                                    num_feature_maps, 
                                    num_feature_maps * reduction_factor]))
b_2 = tf.Variable(tf.random_normal([num_feature_maps * reduction_factor]))
conv_2 = conv2d(conv_1, w_1, b_1)
conv_2 = max_pool(conv_2, k = reduction_factor)
conv_2 = tf.nn.dropout(conv_2, keep_prob)

#densely connected layer
w_d = tf.Variable(tf.random_normal([input_size[0]/(2 * reduction_factor), 
                                    input_size[1]/(2 * reduction_factor),
                                      num_feature_maps * reduction_factor * 2,
                                      num_feature_maps * reduction_factor *
                                      num_feature_maps * reduction_factor * 2 ]))

b_d = tf.Variable(tf.random_normal([num_feature_maps * reduction_factor *
                                      num_feature_maps * reduction_factor * 2]))
#convert output to a 2D matrix
d = tf.reshape(conv_2, [-1, w_d.get_shape().as_list()[0]])
d = tf.nn.relu(tf.add(tf.matmul(d, w_d),b_d))
d = tf.nn.dropout(d, keep_prob)

#readout layer

w_o = tf.Variable(tf.random_normal([num_feature_maps * reduction_factor *
                                      num_feature_maps * reduction_factor * 2, 
                                      num_features_output]))
b_o = tf.Variable(tf.random_normal([num_features_output]))
#one last neural net thiggamagic
pred = tf.add(tf.matmul(d, w_o), b_o)

#cost etc
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#was our prediction correct? (compare the 1st axes of the inputs)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#accuracy is the net mean of all the correct assumptions
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step*batch_size < num_training_steps:
        batch_x, batch_y = get_next_batch()
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,\
                    keep_prob: dropout})
        #note how we don't dropout anything for accuracy and loss test
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,\
                                            keep_prob: 1.})
        loss = sess.run(cost, feed_dict={x: batch_x,\
                                         y: batch_y,\
                                         keep_prob: 1.})
        step += 1