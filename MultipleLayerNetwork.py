# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 22:39:45 2019

@author: 11012
"""

import numpy as np
import tensorflow as tf

def generate(sample_size, mean,cov, diff, regression):
    num_classes=2
    sample_per_class=(sample_size/2)
    
    X0=np.random.multivariate_normal(mean,cov,sample_per_class)
    Y0=np.zeros(sample_per_class)
    
    for ci, d in enumerate(diff):
        X1=np.random.multivariate_normal(mean+d,cov,sample_per_class)
        Y1=(ci+1)*np.ones(sample_per_class)
        
        X0=np.concatenate((X0,X1))
        Y0=np.concatenate(((Y0,Y1))
        
        
    if regression==False:
        class_int = [Y==class_number for class_number in range(num_classes)]
        Y=np.asarray(np.hstack(class_int),dtype=np.float32)
    X,Y =shuffle(X0,Y0)
    
    return X,Y



input_features=tf.placeholder(tf.float32,[None, input_dim])
output_lables=tf.placeholder(tf.float32,[None,lab_dim])

W=tf.Variable(tf.random_normal([input_dim,lab_dim],name="weight"))
b=tf.Variable(tf.zeros(lab_dim),name="bias")

output=tf.nn.sigmoid(tf.multiply(input_features,W)+b)
cross_entropy=-(input_lables*tf.log(output)+(1-input_lables).tf.log(1-output))
ser=tf.square(input_lables-ouput)

loss=tf.reduce_mean(cross_entropy)
err=tf.reduce_mean(ser)


optimizer=tf.train.AdadeltaOptimizer(0.04)
#the  book said this one is the best choice for it dynamicly adjust the gradient.                       
tain=optimizer.minimize(loss)


    
