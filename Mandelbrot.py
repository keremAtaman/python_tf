# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:56:35 2018

@author: K.Ataman
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
Does mendalbro's seris:
    1. Z has initial value equal to 0, Z(0) = 0
    2. Choose the complex number c as the current point. In the Cartesian plane, the
    abscissa axis (horizontal line) represents the real part, while the axis of ordinates
    (vertical line) represents the imaginary part of c.
    3. Iteration: Z(n + 1) = Z(n)2 + c
    Stop when Z(n)2 is larger than the maximum radius;
"""

#creates a mesh grid between given variables

"""
Visulaization:
    Y has 600 columns
    [[-1.3   -1.3   -1.3   ..., -1.3   -1.3   -1.3  ]
     [-1.295 -1.295 -1.295 ..., -1.295 -1.295 -1.295]
     [-1.29  -1.29  -1.29  ..., -1.29  -1.29  -1.29 ]
     ..., 
     [ 1.285  1.285  1.285 ...,  1.285  1.285  1.285]
     [ 1.29   1.29   1.29  ...,  1.29   1.29   1.29 ]
     [ 1.295  1.295  1.295 ...,  1.295  1.295  1.295]]
    
    X has 600 columns
    [[-2.    -1.995 -1.99  ...,  0.985  0.99   0.995]
     [-2.    -1.995 -1.99  ...,  0.985  0.99   0.995]
     [-2.    -1.995 -1.99  ...,  0.985  0.99   0.995]
     ..., 
     [-2.    -1.995 -1.99  ...,  0.985  0.99   0.995]
     [-2.    -1.995 -1.99  ...,  0.985  0.99   0.995]
     [-2.    -1.995 -1.99  ...,  0.985  0.99   0.995]]
"""
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
#z is the complex number made from X and Y variables
"""
    Z=
    array([[-2.000-1.3j  , -1.995-1.3j  , -1.990-1.3j  , ...,  0.985-1.3j  ,
         0.990-1.3j  ,  0.995-1.3j  ],
       [-2.000-1.295j, -1.995-1.295j, -1.990-1.295j, ...,  0.985-1.295j,
         0.990-1.295j,  0.995-1.295j],
       [-2.000-1.29j , -1.995-1.29j , -1.990-1.29j , ...,  0.985-1.29j ,
         0.990-1.29j ,  0.995-1.29j ],
       ..., 
       [-2.000+1.285j, -1.995+1.285j, -1.990+1.285j, ...,  0.985+1.285j,
         0.990+1.285j,  0.995+1.285j],
       [-2.000+1.29j , -1.995+1.29j , -1.990+1.29j , ...,  0.985+1.29j ,
         0.990+1.29j ,  0.995+1.29j ],
       [-2.000+1.295j, -1.995+1.295j, -1.990+1.295j, ...,  0.985+1.295j,
         0.990+1.295j,  0.995+1.295j]])
"""
Z = X+1j*Y
#c is a constant that holds the value of Z
c = tf.constant(Z.astype(np.complex64))
#yoink the format of c, with the exception that zs is a variable, 
#so it can change over time
zs = tf.Variable(c)
#next step's value holder. Initialized as zero
ns = tf.Variable(tf.zeros_like(c, tf.float32))


zs_ = zs*zs + c
#stop condition
not_diverged = tf.abs(zs_) < 4

#group multiple operations using tf.group
#cast is (shape, dType)
step = tf.group(zs.assign(zs_),\
                ns.assign_add(tf.cast(not_diverged, tf.float32)))


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


#do the operation 200 times
for i in range(200): 
    step.run()
    
plt.imshow(ns.eval())
plt.show()