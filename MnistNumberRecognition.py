# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
train, test, validation
"""
import tensorflow as tf
import  numpy  as  np
import matplotlib.pyplot as plt
import pylab

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

trainimg=mnist.train.images
trainimg_labels=mnist.train.labels
testimg=mnist.test.images
testimg_labels=mnist.test.labels
print("mnist loaded successfully")
#initialization
x=tf.placeholder("float",[None, 784])
y=tf.placeholder("float",[None,10])
W=tf.Variable(tf.random_normal([784,10]))
b=tf.Variable(tf.zeros([10]))
#activation funciont
active=tf.nn.softmax(tf.matmul(x,W)+b)
#costfunction
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(active),reduction_indices=1))
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
saver=tf.train.Saver()
path="D:/13pythoncode/tensorflow/sessions/"
#prediction，用于比较计算值与真实值是否相等,这里的tf.argmax(m, 1),
#会取出one_hot编码中下标为1元素的下标
pre = tf.equal(tf.argmax(active,1),tf.argmax(y,1))
#accuracy
accr=tf.reduce_mean(tf.cast(pre,float)) 

#training batch setup 
training_epoch = 20
batch_size = 100
display_step = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epoch):    
        avg_cost=0.
        num_batch=int(mnist.train.num_examples/batch_size)
        #calculation
        for i in range(num_batch):
            batch_xs, batch_ys=mnist.train.next_batch(batch_size)
            opt,cst=sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            avg_cost+=cst/num_batch       
        #display
        if epoch % display_step==0:
            train_acc=sess.run(accr,feed_dict={x:batch_xs,y:batch_ys})
            test_acc=sess.run(accr,feed_dict={x:mnist.test.images,y:mnist.test.labels})
            print(epoch,training_epoch,avg_cost,train_acc,test_acc)
        print("done")
    saver.save(sess,path+"mnist.ckpt")    
print("run the second round...")


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,path+"mnist.ckpt")
    correcct_prediction=tf.equal(tf.arg_max(active,1),tf.arg_max(y,1))
    accuracy=tf.reduce_mean(tf.cast(correcct_prediction,tf.float32))
    print("accuraccy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
    output=tf.arg_max(active,1)
    batch_xs,batch_ys=mnist.validation.next_batch(4)
    output_val, active_value=sess.run([output,active],feed_dict={x:batch_xs,y:batch_ys})
    print(output_val,active_value,batch_ys)
    im=batch_xs[0]
    im=im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
    
    im=batch_xs[1]
    im=im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
    
    im=batch_xs[2]
    im=im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
    
    im=batch_xs[3]
    im=im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()




