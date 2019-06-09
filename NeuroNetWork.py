# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:13:21 2019

@author: 11012
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#network topoloty
n_input=784
n_hidden1=256
n_hidden2=128
n_output=10
#hereby i define input and output
x=tf.placeholder("float",[None,n_input])
y=tf.placeholder("float",[None,n_output])

#hereby i define the parameters in the hidden layers
stddev=0.1
weights={
        "w1":tf.Variable(tf.random_normal([n_input,n_hidden1],stddev=stddev)),
        "w2":tf.Variable(tf.random_normal([n_hidden1,n_hidden2],stddev=stddev)),
        "out":tf.Variable(tf.random_normal([n_hidden2,n_output],stddev=stddev))
        }
biases={"b1":tf.Variable(tf.random_normal([n_hidden1])),
        "b2":tf.Variable(tf.random_normal([n_hidden2])),
        "out":tf.Variable(tf.random_normal([n_output]))
        }

def multilayer_perceptron(_X, _weights,_biases):
    #the sigmoid here, activation crucialo;    
    layer1=tf.nn.sigmoid(tf.add(tf.matmul(_X,_weights['w1']),_biases['b1']))
    layer2=tf.nn.sigmoid(tf.add(tf.matmul(layer1,_weights['w2']),_biases['b2']))
    return (tf.matmul(layer2,_weights['out']) + _biases['out'])
#prediction
pred=multilayer_perceptron(x,weights,biases)
#loss and optimizer
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
corr=tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
accr=tf.reduce_mean(tf.cast(corr,"float"))
#cost and optimizer active
tf.global_variables_initializer()
print("functions are ready")

training_epochs=200
batch_size=100
display_step=4
sess=tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(training_epochs):
    avg_cost=0
    total_batch=int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds={x:batch_xs,y:batch_ys}
        sess.run(optimizer,feed_dict=feeds)
        avg_cost+=sess.run(cost,feed_dict=feeds)
    avg_cost=avg_cost/total_batch
    if(epoch+1)%display_step==0:
        print(epoch,training_epochs,avg_cost)
        feeds={x:batch_xs,y:batch_ys}
        train_acc=sess.run(accr,feed_dict=feeds)
        print(train_acc)
        feeds={x:mnist.test.images,y:mnist.test.labels}
        test_acc=sess.run(accr,feed_dict=feeds)
        print(test_acc)
print("optimization finished")
    







