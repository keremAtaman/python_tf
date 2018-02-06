# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:25:14 2018

@author: K.Ataman
"""

import tensorflow as tf


a = tf.constant(10,name="a")
b = tf.constant(90,name="b")
y = tf.Variable(a+b*2, name="y")


model = tf.global_variables_initializer()
with tf.Session() as session:
    #merges all summaries from the default graph
    merged = tf.summary.merge_all
    #Writes the summary of the variables after execution
    writer = tf.summary.FileWriter\
        ("/tmp/tensorflowlogs",session.graph)
    session.run(model)
    print(session.run(y))
    
"""
You can view the graph by the following steps:
    Go to anaconda prompt
    Type "activate tensorflow"
    Type tensorboard --logdir=/tmp/tensorflowlogs
    Open a browser and go to localhost:6006
    
"""