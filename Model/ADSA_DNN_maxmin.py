from __future__ import print_function

import tensorflow as tf
import os
import sys
import timeit

import scipy.io as sio
import numpy as np
import util
import cnnet as nn

#####################################
class ADSA_struct(object):
    def __init__(self):
        self.FE_struct=nn.DNN_struct()
        self.TC_struct=nn.DNN_struct()
        self.PSFE_struct=nn.CNN_struct()
        self.PTFE_struct=nn.CNN_struct()
        self.DC_struct=nn.DNN_struct()
        self.Sep_struct=nn.DNN_struct()
        
#####################################        
class ADSAModel(object):
    def __init__(self, struct, x_dim, y_dim):
        self.struct=struct
        self.x_dim=x_dim
        self.y_dim=y_dim
        self._build_model()
        
    def _build_model(self):
        
        self.X = tf.placeholder(tf.float32, [None, self.x_dim])
        self.y = tf.placeholder(tf.float32, [None, self.y_dim])
        self.domain_labels = tf.placeholder(tf.float32, [None, 2])
        self.domain_sep_labels=tf.placeholder(tf.float32, [None, 3])
        self.batch_size=tf.placeholder(tf.int32,[2])
        self.flip_scale=tf.placeholder(tf.float32, [])
        self.train_flag= tf.placeholder(tf.bool, [])
        
        X_input= self.X
        
        #DNN model for feature extraction
        with tf.variable_scope('shared_feature_extractor'):
            self.feature_extract=nn.DNN_Block(X_input, self.batch_size, self.struct.FE_struct, name='Shared_Feature_extractor')
            self.feature=tf.reshape(self.feature_extract.output, [-1, self.struct.TC_struct.layer_struct[0]])
        with tf.variable_scope('source_private_feature_extractor'):
            source_input=tf.slice(X_input, [0,0],[self.batch_size[0], -1])
            self.source_private_feature_extract=nn.DNN_Block(source_input, self.batch_size, self.struct.PSFE_struct, name='Source_private_feature_extractor')
            self.source_private_feature=tf.reshape(self.source_private_feature_extract.output, [-1, self.struct.TC_struct.layer_struct[0]])
        with tf.variable_scope('target_private_feature_extractor'):
            target_input=tf.slice(X_input, [self.batch_size[0],0], [self.batch_size[1],-1])
            self.target_private_feature_extract=nn.DNN_Block(target_input, self.batch_size, self.struct.PTFE_struct, name='Target_private_feature_extractor')
            self.target_private_feature=tf.reshape(self.target_private_feature_extract.output, [-1, self.struct.TC_struct.layer_struct[0]])
        # DNN model for class prediction
        with tf.variable_scope('label_predictor'):
            
            self.source_label=tf.slice(self.y, [0,0], [self.batch_size[0], -1])
            self.target_label=tf.slice(self.y, [self.batch_size[0],0], [self.batch_size[1], -1])
            
            self.task_classifier=nn.DNN_Block(self.feature, self.batch_size, self.struct.TC_struct, name='Task_classifier')
            
            source_logits=tf.slice(self.task_classifier.output,[0,0],[self.batch_size[0],-1])
            target_logits=tf.slice(self.task_classifier.output,[self.batch_size[0],0],[self.batch_size[1],-1])
            
            self.source_pred=tf.nn.softmax(source_logits)
            self.target_pred=tf.nn.softmax(target_logits)
            self.pred_loss=tf.nn.softmax_cross_entropy_with_logits(source_logits, self.source_label)
        # Domain adaptation discriminator
        with tf.variable_scope('domain_predictor'):
            feat=self.feature
            self.domain_classifier=nn.DNN_Block(feat, self.batch_size, self.struct.DC_struct, name='Domain_adapt_classifier')
            d_logits=self.domain_classifier.output
            self.domain_adapt_pred=tf.nn.softmax(d_logits)
            self.domain_adapt_loss=tf.nn.softmax_cross_entropy_with_logits(d_logits, self.domain_labels)
        # Domain separation discriminator
        with tf.variable_scope('domain_sep_classifier'):
            self.sep_feat=tf.concat(0,[self.feature, self.source_private_feature, self.target_private_feature])
            self.domain_sep_classifier=nn.DNN_Block(self.sep_feat, self.batch_size, self.struct.Sep_struct, name='Domain_sep_classifier')
            s_logits=self.domain_sep_classifier.output
            self.domain_sep_pred=tf.nn.softmax(s_logits)
            self.domain_sep_loss=tf.nn.softmax_cross_entropy_with_logits(s_logits, self.domain_sep_labels)
            
    def get_feature(self):
        source_feature=tf.slice(self.feature, [0,0],[self.batch_size[0],-1])
        target_feature=tf.slice(self.feature, [self.batch_size[0],0],[self.batch_size[1],-1])
        
        return [source_feature, target_feature, self.source_private_feature, self.target_private_feature]
        