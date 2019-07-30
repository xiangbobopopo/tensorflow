# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow  as  tf
import  numpy  as  np
import matplotlib.pyplot as plt

train_X=np.linspace(-1,1,100) 
train_Y=2*train_X + np.random.randn(*train_X.shape) * 0.3
plt.plot(train_X,train_Y,'ro',label='Original data')

#hereby we define the input value and the forward propagation;
#it's like, we get the data, we get the function(by apply the parameters to the function)
X=tf.placeholder("float")
Y=tf.placeholder("float")
W=tf.Variable(tf.random_normal([1]),name="weight")
b=tf.Variable(tf.zeros([1]),name="bias")
z=tf.multiply(X,W)+b
tf.summary.histogram("z",z)

#hereby we define the backward proopagation
cost=tf.reduce_mean(tf.square(Y-z))
tf.summary.histogram("loss",cost)
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
saver=tf.train.Saver()
save_dir="D:/13pythoncode/tensorflow/summaries"

#we define our running batch  here
train_batch=20
display_step=2
#define a dict to store the intermediary result;
plotdata={"batchsize":[],"loss":[]}
#built a sesson and run the established code;
#first we need to initialize the variables; and then run the iterariton
#data, placeholder,
with tf.Session() as  sess:
    sess.run(tf.global_variables_initializer())
    merge_operation=tf.summary.merge_all()
    summary_writer=tf.summary.FileWriter("D:/13pythoncode/tensorflow/summaries",sess.graph)
    for epoch in range(train_batch):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
            summary_str=sess.run(merge_operation,feed_dict={X:x,Y:y})
            summary_writer.add_summary(summary_str,epoch)            
        if epoch%display_step==0:
            loss=sess.run(cost,feed_dict={X:train_X, Y:train_Y})
            if (loss!="NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
                print("epoch:",epoch,"loss:",loss,"w:",sess.run(W),"b:",sess.run(b))
    
    print("finished")
    
    saver.save(sess,"D:/13pythoncode/tensorflow/test.cpkt")
    print("epoch:",epoch,"loss:",loss,"w:",sess.run(W),"b:",sess.run(b))
    print("prediction:",sess.run(z,feed_dict={X:2}))


#restore the stored data 
with tf.Session() as  sess2:
    sess2.run(tf.global_variables_initializer())
    saver.restore(sess2, "D:/13pythoncode/tensorflow/test.cpkt")
    print(sess2.run([z],feed_dict={X:26}))
    

    
