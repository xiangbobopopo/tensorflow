# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
train, test, validation

"""
import tensorflow  as  tf
import  numpy  as  np
import matplotlib.pyplot as plt

import pylab 

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

trainimg=mnist.train.images
trainimg_labels=mnist.train.labels
testimg=mnist.test.images
testimg_labels=mnist.test.labels

#initialization
x=tf.placeholder("float",[None, 784])
y=tf.placeholder("float",[None,10])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
#activation funciont
active=tf.nn.softmax(tf.matmul(x,W)+b)
#costfunction
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(active),reduction_indices=1))
#optimizer
learning_rate=0.03
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#prediction
pre = tf.equal(tf.arg_max(active,1),tf.argmax(y,1))
#accuracy
accr=tf.cast(pre,float)
#init
init=tf.global_variables_initializer()

training_epoch = 50
batch_size = 100
display_step = 5

sess=tf.Session()
sess.run(init)

for epoch in range(training_epoch):
    avg_cost=0.
    num_batch=int(mnist.train.num_examples/batch_size)
    



print("mnist loaded successfully")
print(testimg.shape)
print(testimg_labels.shape)
print(trainimg.shape)
print(trainimg_labels.shape)
print(trainimg_labels[0])

