# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 22:20:24 2019

@author: 11012
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#define the parameters
weights={
        'wc1':tf.Variable(tf.random_normal([3,3,1,64],stddev=0.1)),
        'wc2':tf.Variable(tf.random_normal([3,3,64,128],stddev=0.1)),
        'wd1':tf.Variable(tf.random_normal([7*7*128,1024],stddev=0,1)),
        'wd2':tf.Variable(tf.random_normal([1024,n_output],stddev=0.1))
        }
biases={
        'bc1':tf.Variable(tf.random_normal([64],stddev=0.1)),
        'bc2':tf.Variable(tf.random_normal([128],stddev=0,1)),
        'bd1':tf.Variable(tf.random_normal([1024],stddev=0.1)),
        'bd2':tf.Variable(tf.random_normal([n_output],stddev=0.1))
        }

def conv_basic(_input,_w,_b,_keepratio):
    _input_r=tf.reshape(_input,[-1,28,28,1])
    _conv1=tf.nn.conv2d(_input_r,_w['wc1'],strides=[1,1,1,1],padding='SAME')
    _conv1=tf.nn.relu(tf.nn.bias_add(_conv1,_b['bc1']))
    _pool1=tf.nn.max_pool(_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
   
    _pool_dr1=tf.nn.dropout(_pool2,_keepratio)
    
    