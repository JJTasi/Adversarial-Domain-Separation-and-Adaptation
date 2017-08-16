from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as ly
import os
import sys
import timeit
from util import *

import scipy.io as sio
import numpy as np

def xavier_init(shape, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    #low = -constant*np.sqrt(6.0/(np.sum(shape))) 
    #high = constant*np.sqrt(6.0/(np.sum(shape)))
    
    fan_in=shape[0] if len(shape)==2 else np.prod(shape[0:2])
    fan_out=shape[-1]
    
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    
    return tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32)
def weight_variable(shape):
    # use Xavier initialization
    initial = xavier_init(shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def deconv2d(x,W,shape):
    return tf.nn.conv2d_transpose(x,W, shape, strides=[1,4,4,1], padding='SAME')

##############################################
######## DNN #################################
##############################################
class DNNLayer(object):
    def __init__(self, layer_struct,batch_norm, name=''):
        #layer struct=[input_dim, output_dim]
        
        self.Weight=weight_variable(layer_struct)
        self.Bias=bias_variable([layer_struct[-1]])
        self.name=name
        self.batch_norm=batch_norm
        
    def __call__(self, X_input, activation=tf.nn.relu):
        #feedward
        self.hidden=tf.matmul(X_input, self.Weight)+self.Bias
        if activation=='linear':
            if self.batch_norm==True:
                self.hidden=ly.batch_norm(self.hidden)
        else:
            if self.batch_norm==True:
                self.hidden=ly.batch_norm(self.hidden)
            self.hidden=activation(self.hidden)
        
        return self.hidden
    
class DNN_struct(object):
    def __init__(self):
        self.layer_struct=[]
        self.activation=[]
        self.batch_norm=[]
##############################################
class DNN_Block(object):
    def __init__(self, X_input, batch_size, struct, name=''):
        l=len(struct.activation)
        self.Layer_group=[]
        self.batch_size=batch_size
        if l==0:
            self.output=X_input
            print("%s is empty, link input to output directly" % (name))
            return
        
        print("%s is constructed with hidden layer number %i" % (name, l-1))
        
        layer_name = '%s_L%i' % (name, 1)
        
        tmplayer=DNNLayer(struct.layer_struct[0:2], struct.batch_norm[0], layer_name)
        tmphidden=tmplayer(X_input,struct.activation[0])
        
        self.Layer_group.append(tmphidden)
        
        
        for i in range(l-1):
            j=i+1
            layer_name='%s_L%i' % (name, j+1)
            
            tmplayer=DNNLayer(struct.layer_struct[j:j+2],struct.batch_norm[j], layer_name)
            tmphidden=tmplayer(self.Layer_group[-1],struct.activation[j])
            self.Layer_group.append(tmphidden)
            
        
        self.output=self.Layer_group[-1]
##############################################
######## CNN #################################
##############################################


class CNNLayer(object):
    def __init__(self, batch_size, layer_struct, batch_norm, name='',pooling='down'):
        #initail
        # layer struct = [filter row ; filter col; input_dim; output_dim]
        self.Weight=weight_variable(layer_struct)
        self.Bias=bias_variable([layer_struct[-1]])
        self.name=name
        self.pool=pooling
        self.batch_size=batch_size
        self.batch_norm=batch_norm
        
    def __call__(self, X_input, activation=tf.nn.relu, up_sample_shape=None):
        #feedforward
        self.hidden=conv2d(X_input,self.Weight)+self.Bias
        if self.pool=='down':
            if activation=='linear':
                if self.batch_norm==True:
                    self.hidden=ly.batch_norm(self.hidden)
            else:
                if self.batch_norm==True:
                    self.hidden=ly.batch_norm(self.hidden)
                self.hidden=activation(self.hidden)
            self.hidden=max_pool_2x2(self.hidden)
        elif self.pool=='None':
            if activation=='linear':
                if self.batch_norm==True:
                    self.hidden=ly.batch_norm(self.hidden)
            else:
                if self.batch_norm==True:
                    self.hidden=ly.batch_norm(self.hidden)
                self.hidden=activation(self.hidden)
        elif self.pool=='up':
            stride=4
            if activation=='linear':
                self.hidden=deconv2d(X_input, self.Weight, shape=[tf.shape(X_input)[0],28,28,16])
            else:
                self.hidden=activation(deconv2d(X_input, self.Weight, shape=[tf.shape(X_input)[0],28,28,16]))
        
        return self.hidden

##############################################
class CNN_struct(object):
    def __init__(self):
        self.layer_struct=[]
        self.activation=[]
        self.pooling=[]
        self.batch_norm=[]
        
##############################################
class CNN_Block(object):
    def __init__(self, X_input,batch_size, struct ,name=''):
        l=len(struct.activation)
        
        self.Layer_group=[]
        self.batch_size=batch_size
        if l==0:
            self.output=X_input
            print("%s is empty, link input to output directly " % (name))
            return
        
        print("%s is constructed with hidden layer number %i" % (name, l))
        
        layer_name='%s_L%i' % (name, 1)
        tmplayer=CNNLayer(self.batch_size, layer_struct=struct.layer_struct[0], batch_norm=struct.batch_norm[0], name=layer_name,pooling=struct.pooling[0])
        tmphidden=tmplayer(X_input, struct.activation[0])
        
        self.Layer_group.append(tmphidden)
        
        for i in range(l-1):
            j=i+1
            layer_name='%s_L%i' % (name, j)
            
            tmplayer=CNNLayer(self.batch_size, layer_struct=struct.layer_struct[j], batch_norm=struct.batch_norm[j], name=layer_name,pooling=struct.pooling[j])
            tmphidden=tmplayer(self.Layer_group[-1], struct.activation[j])
            self.Layer_group.append(tmphidden)
        self.output=self.Layer_group[-1]
################################################
class CNN_decoder(object):
    def __init__(self,struct, name=''):
        self.struct=struct
        self.name=name
    def __call__(self, X_input):
        
        batch_size=tf.shape(X_input)[0]
        
        
        fc=DNNLayer([7*7*48*2,300],True,'decoder_fc')
        fc_hidden=fc(X_input,tf.nn.relu)
        
        images=tf.reshape(fc_hidden, [batch_size,10,10,3])
        
        hl_0=CNNLayer(batch_size,self.struct.layer_struct[0],self.struct.batch_norm[0],('L1_%s'%(self.name)),pooling=None)
        hidden_0=hl_0(images, self.struct.activation[0])
        hidden_0=tf.image.resize_nearest_neighbor(hidden_0,(14,14))
        
        hl_1=CNNLayer(batch_size,self.struct.layer_struct[1],self.struct.batch_norm[1],('L1_%s'%(self.name)),pooling=None)
        hidden_1=hl_1(hidden_0, self.struct.activation[1])
        hidden_1=tf.image.resize_nearest_neighbor(hidden_1,(28,28))
        
        hl_2=CNNLayer(batch_size,self.struct.layer_struct[2],self.struct.batch_norm[2],('L1_%s'%(self.name)),pooling=None)
        hidden_2=hl_2(hidden_1, self.struct.activation[2])
        
        return hidden_2