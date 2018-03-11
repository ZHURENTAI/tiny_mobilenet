import tensorflow as tf
import numpy as np
import cv2 as cv
from scipy.misc import imread, imresize
from collections import namedtuple
sess = tf.Session()
is_training=True
conv2d=namedtuple("conv2d",["kernel","padding","stride","bias","activation","batch_normalization"])
separable_conv2d=namedtuple("separable_conv2d",["deep_kernel","point_kernel","padding","stride","activation","batch_normalization"])
pool=namedtuple("pool",["kernel","stride","Type","padding"])
dropout=namedtuple("dropout",["keep_prob"])
full_connect=namedtuple("full_connect",["shape","ini_type"])
layers_def=[
  conv2d(kernel=[3,3,3,32],padding="SAME",stride=[1,2,2,1],bias=[32],activation=tf.nn.relu6,batch_normalization=True),
  separable_conv2d(deep_kernel=[3,3,32,1],point_kernel=[1,1,32,64],padding="SAME",stride=[1,1,1,1],activation=tf.nn.relu,batch_normalization=True),
  separable_conv2d(deep_kernel=[3,3,64,1],point_kernel=[1,1,64,128],padding="SAME",stride=[1,2,2,1],activation=tf.nn.relu,batch_normalization=True),
  separable_conv2d(deep_kernel=[3,3,128,1],point_kernel=[1,1,128,128],padding="SAME",stride=[1,1,1,1],activation=tf.nn.relu,batch_normalization=True),
  separable_conv2d(deep_kernel=[3,3,128,1],point_kernel=[1,1,128,256],padding="SAME",stride=[1,2,2,1],activation=tf.nn.relu,batch_normalization=True),
  separable_conv2d(deep_kernel=[3,3,256,1],point_kernel=[1,1,256,256],padding="SAME",stride=[1,1,1,1],activation=tf.nn.relu,batch_normalization=True),
  separable_conv2d(deep_kernel=[3,3,256,1],point_kernel=[1,1,256,512],padding="SAME",stride=[1,2,2,1],activation=tf.nn.relu,batch_normalization=True),
  separable_conv2d(deep_kernel=[3,3,512,1],point_kernel=[1,1,512,512],padding="SAME",stride=[1,1,1,1],activation=tf.nn.relu,batch_normalization=True),
  separable_conv2d(deep_kernel=[3,3,512,1],point_kernel=[1,1,512,512],padding="SAME",stride=[1,1,1,1],activation=tf.nn.relu,batch_normalization=True),
  separable_conv2d(deep_kernel=[3,3,512,1],point_kernel=[1,1,512,512],padding="SAME",stride=[1,1,1,1],activation=tf.nn.relu6,batch_normalization=True),
  separable_conv2d(deep_kernel=[3,3,512,1],point_kernel=[1,1,512,512],padding="SAME",stride=[1,1,1,1],activation=tf.nn.relu,batch_normalization=True),
  separable_conv2d(deep_kernel=[3,3,512,1],point_kernel=[1,1,512,512],padding="SAME",stride=[1,1,1,1],activation=tf.nn.relu,batch_normalization=True),
  separable_conv2d(deep_kernel=[3,3,512,1],point_kernel=[1,1,512,1024],padding="SAME",stride=[1,2,2,1],activation=tf.nn.relu,batch_normalization=True),
  separable_conv2d(deep_kernel=[3,3,1024,1],point_kernel=[1,1,1024,1024],padding="SAME",stride=[1,1,1,1],activation=tf.nn.relu,batch_normalization=True),
  pool(kernel=[1,2,2,1],padding="VALID",stride=[1,1,1,1],Type=tf.nn.avg_pool),
  dropout(keep_prob=0.5),
  full_connect(shape=200,ini_type=tf.random_normal)
]

def mobilenet_layer(x):
  with tf.name_scope("preprocess"):
    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    net =x-mean
  with tf.name_scope("conv_layers"):
    for i,layer in enumerate(layers_def):
      if isinstance(layer,separable_conv2d):
          weights = tf.Variable(tf.random_normal(layer.deep_kernel, stddev=0.1), name="depthwise_conv"+str(i))
          bias = tf.Variable(tf.constant(0.0,shape=[net.shape[-1]]),name="bias"+str(i))
          net=tf.nn.depthwise_conv2d(net,weights,[1,1,1,1],layer.padding)  
          net=tf.nn.bias_add(net,bias)
          if(layer.batch_normalization):
             net=tf.layers.batch_normalization(net,training=is_training) 
          net=layer.activation(net)  
          weights = tf.Variable(tf.random_normal(layer.point_kernel, stddev=0.1), name="pointwise_conv"+str(i))
          bias = tf.Variable(tf.constant(0.0,shape=[layer.point_kernel[-1]]),name="bias"+str(i))
          net=tf.nn.conv2d(net,weights,layer.stride,layer.padding)
          net=tf.nn.bias_add(net,bias)
          if(layer.batch_normalization):
             net=tf.layers.batch_normalization(net,training=is_training) 
          net=layer.activation(net)
          print net.get_shape()  
      if isinstance(layer,conv2d):
          bias = tf.Variable(tf.constant(0.0,shape=[layer.kernel[-1]]),name="bias"+str(i))
          weights = tf.Variable(tf.random_normal(layer.kernel,stddev=0.1), name="conv2d"+str(i))
          net=tf.nn.conv2d(net,weights,layer.stride,layer.padding)
          net=tf.nn.bias_add(net,bias)
          if(layer.batch_normalization):
             net=tf.layers.batch_normalization(net,training=is_training) 
          net=layer.activation(net)
          print net.get_shape()  
      if isinstance(layer,pool):
          net=layer.Type(net, layer.kernel, layer.stride, layer.padding,name="pool"+str(i))
          net=tf.reshape(net,[-1,net.shape[-1]])
          print net.get_shape()
      if isinstance(layer,dropout):
          if(is_training):
            net=tf.nn.dropout(net,layer.keep_prob)
          print net.get_shape()
      if isinstance(layer,full_connect):
          weights=tf.Variable(layer.ini_type([int(net.shape[-1]),layer.shape],stddev=0.1),name="fcw"+str(i))
          bias = tf.Variable(tf.constant(0.0,shape=[layer.shape]),name="fcb"+str(i))
          net=tf.nn.bias_add(tf.matmul(net,weights),bias)
          print net.get_shape()
  return net 



