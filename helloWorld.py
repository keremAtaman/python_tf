# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:13:42 2018

@author: K.Ataman
"""

import tensorflow as tf
x = tf.constant(1,name='x')
y = tf.Variable(x+9,name='y')
#does not work, as we just said "when you run the session, add 9 to the 
#value of y
print(y)

model = tf.global_variables_initializer()
with tf.Session() as session:
    #run the model, initialize x and y for use
    session.run(model)
    #actually do the calculation now
    print(session.run(y))