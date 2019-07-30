# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 02:23:57 2019
@author: 11012
"""
import tensorflow as tf
import  numpy  as  np
import matplotlib.pyplot as plt
import pylab

#data import
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

training_images=mnist.train.images
training_labels=mnist.train.labels
testing_images=mnist.test.images
testing_labels=mnist.test.labels
validation_images=mnist.validation.images
validation_lables=mnist.validation.images

#placeholder, hereby the None  means
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
#define the  modle
w=tf.Variable(tf.random_normal([784,10]))
b=tf.Variable(tf.zeros([10]))
pre=tf.nn.softmax(tf.matmul(x,w)+b)

#backward propagation
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pre),reduction_indices=1))
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#define batch, batch size, and display step
training_batch=10
batch_size=100
display_steps=1


with tf.Session() as  sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_batch):
        avg_cost=0.
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xm, batch_ym=mnist.train.next_batch(batch_size)
            _, c=sess.run([optimizer, cost],feed_dict={x:batch_xm, y:batch_ym})
            avg_cost += c/total_batch            
        if(epoch+1)%display_steps==0:
            print("epoch:",epoch,"cost:",avg_cost)
    print("finished training")
            




