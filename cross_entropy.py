# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 00:28:06 2019

@author: 11012

cross entropy test
"""
import tensorflow as tf


lables=[[0.4,0.1,0.5],[0.2,0.7,0.1]]
logits=[[2,0.5,6],[0.1,0,3]]
logit_scaled01=tf.nn.softmax(logits)
logit_scaled02=tf.nn.softmax(logit_scaled01)

result1=tf.nn.softmax_cross_entropy_with_logits(labels=lables,logits=logits)
result2=tf.nn.softmax_cross_entropy_with_logits(labels=lables,logits=logit_scaled01)
cost=-tf.reduce_sum(lables*tf.log(logit_scaled01),1)
with tf.Session() as sess:
    print("01:",sess.run(logit_scaled01),"02:",sess.run(logit_scaled02))
    print("result1:",sess.run(result1),"result2:",sess.run(result2),"cost:",sess.run(cost))



