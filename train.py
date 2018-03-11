from layer import mobilenet_layer
import tensorflow as tf
import numpy as np
import random
import cv2 as cv 
from scipy.misc import imread, imresize
import os
import gc 
import pandas as pd

min_batch=50
base_dir="/home/tiger/tiny-imagenet-200"
train_dir=base_dir+"/train"
test_dir=base_dir+"/test"
val_dir=base_dir+"/val"

global_label=[]
label=[]
image_dir=[]	
zeros=np.ndarray(shape=[1,200])
i=0
for type in os.listdir(train_dir):
  global_label.append(type)
  sub_path=os.path.join(train_dir,type,"images")
  zeros[0,i]=0
  for image_name in os.listdir(sub_path):
     image_path=os.path.join(sub_path,image_name)
     label.append(i)
     image_dir.append(image_path)
  i+=1
c=list(zip(label,image_dir))
random.shuffle(c)
label[:],image_dir[:]=zip(*c)
del c
gc.collect()
"""
val_anno=pd.read_csv(val_dir+"/val_annotations.txt",sep="/t",header=None,names=["file","type","x","y","h","w"])
for idx, line in val_anno.iterrows():
   print idx
"""
x=tf.placeholder(tf.float32,[None,64,64,3])
y=tf.placeholder(tf.float32,[None,200])
'''
conv1 = tf.layers.conv2d(
            inputs=x, 
            filters=32, 
            kernel_size=[5,5],
            padding='SAME',
            activation=tf.nn.relu, 
            name="conv1")

#shape after conv1: [-1, 64, 64, 32]
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

conv2 = tf.layers.conv2d(
            inputs=pool1, 
            filters=64,
            kernel_size=[5,5],
            padding='SAME',
            activation=tf.nn.relu, 
            name="conv2")

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# Dense Layer
pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense, rate=0.4)
dropout_reshape = tf.reshape(dropout, [-1, 8 * 8 * 64])
logits = tf.layers.dense(inputs=dropout_reshape, units=200, name='output')
y_pred = tf.nn.softmax(logits, name="Y_proba")
'''
y_pred=tf.nn.softmax(mobilenet_layer(x))
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_pred))
train_step=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
saver=tf.train.Saver()

batch=np.zeros(shape=[min_batch,64,64,3])
batch_label=np.zeros(shape=[min_batch,200])
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  j=0 
  k=0
  for i in range(0, len(label)):      
      img= imread(image_dir[i])
      index=label[i]
      label_vector=np.zeros(shape=[1,200])
      label_vector[0,index]=1
      if(img.shape==(64,64,3)): 
        #mean = [123.68, 116.779, 103.939]
        batch[j,:,:,:]=img
        batch_label[j,:]=label_vector
        j+=1 
        k+=1
        if(j == min_batch):
          j=0
          print np.sum(batch)
          print np.sum(batch_label)
          sess.run(train_step,feed_dict={x:batch,y:batch_label})
          loss=cross_entropy.eval(feed_dict={x:batch,y:batch_label})
          print ("loss=",loss)
          if (k%500 ==0):           
            print(k/500)
  saver.save(sess,"mobile")
   
