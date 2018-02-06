# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:30:37 2018

@author: K.Ataman
"""
"""
Find linear relation of the kind y = Ax + B + G(n) where G(n) is Additive White
Gaussian Noise
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_points = 500
x_point = []
y_point = []
a = 0.22
b = 0.78

"""generate random data"""
for i in range(num_points):
    x = np.random.normal(0.0,0.5)
    # y= ax+b with some additional noise
    y = a*x + b +np.random.normal(0.0,0.1)
    x_point.append([x])
    y_point.append([y])

#A represents our estimation of A    , with uniform initialization
A = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
B = tf.Variable(tf.zeros([1]))
y = A * x_point + B
#mean square error is defined as cost
cost_function = tf.reduce_mean(tf.square(y - y_point))
#optimize cost using gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cost_function)

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    for step in range(0,21):
        #train (includes data feeding)
        session.run(train)
        #plot results
        if (step % 5) == 0:
            plt.plot(x_point,y_point,'o',
                label='step = {}'
                .format(step))
            plt.plot(x_point,
                session.run(A) *
                x_point +
                session.run(B))
            plt.legend()
            plt.show()