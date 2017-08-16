from __future__ import print_function

import os
import sys
import timeit
import pickle

import scipy.io as sio
import numpy as np
import tensorflow as tf

#sys.path.insert(0, "/home/jay/DVTL/Model/")
import nnet as nn
import criteria	as er
import util

def features_plot(features_model, source_data, target_data, sample_n, description, plot_enable=True):
                
    train_fts_source, train_labels_source = source_data[0]
    valid_fts_source, valid_labels_source = source_data[1]
    test_fts_source, test_labels_source = source_data[2]
    
    train_fts_target, train_labels_target = target_data[0]
    valid_fts_target, valid_labels_target = target_data[1]
    test_fts_target, test_labels_target = target_data[2]        
    
    y_dim = np.shape(train_labels_source)[1]

    S_labels = train_labels_source[0:sample_n, :]
    T_labels = train_labels_target[0:sample_n, :]

    zj_S = features_model[0][0:sample_n, :]
    zj_T = features_model[1][0:sample_n, :]
    zi_S = features_model[2][0:sample_n, :]
    zi_T = features_model[3][0:sample_n, :]
    
    zj_S, zj_T , zi_S, zi_T= util.domain_feature_tsne(zj_S, zj_T, zi_S, zi_T)
    
    filename = '/home/jay/DSN/tensorflow/DSN_GAN_loss/Experimental_Result/Amazon/%s.png'
    label_zj_S = []
    label_zj_T = []
    label_zi_S = []
    label_zi_T = []

    for i in range(y_dim):
        '''
        label_zj_S.append( zj_S[np.where(S_labels[:,i] == 1)[0], :] )
        label_zj_T.append( zj_T[np.where(T_labels[:,i] == 1)[0], :] )
        label_zi_S.append( zi_S[np.where(S_labels[:,i] == 1)[0], :] )
        label_zi_T.append( zi_T[np.where(T_labels[:,i] == 1)[0], :] )
        '''
    
    #Source zy feature
    title = 'Source_zy_feature_%s' % (description)
    fts = ()
    '''
    for i in range(y_dim):
        fts = fts+(label_zj_S[i][:,0], label_zj_S[i][:,1])
    label = ['negative', 'positive']
    '''
    fts=(zj_S[:,0],zj_S[:,1])+(zi_S[:,0],zi_S[:,1])

    label = ['joint','individual']
    color = [1, 2]
    marker = [1, 1]
    line = False   
    legend = True
    util.data2plot(title=title, fts=fts, label=label, color=color, marker=marker, line=line, legend = legend, plot_enable=plot_enable, filename=filename)            
        
    #Target zy feature
    title = 'Target_zy_feature_%s' % (description)
    fts = ()
    '''
    for i in range(y_dim):
        fts = fts+(label_zy_T[i][:,0], label_zy_T[i][:,1])
    label = ['negative', 'positive']
    '''
    fts=(zj_T[:,0],zj_T[:,1])+(zi_T[:,0],zi_T[:,1])
    label = ['joint','individual']
    color = [1, 2]
    marker = [3, 3]
    line = False 
    legend = True
    util.data2plot(title=title, fts=fts, label=label, color=color, marker=marker, line=line, legend = legend, plot_enable=plot_enable, filename=filename) 
    
    #Both source, target zy feature
    title = 'Zy_feature_%s' % (description)
    fts = ()
    tmp = ()
    '''
    for i in range(y_dim):
        fts = fts+(label_zy_S[i][:,0], label_zy_S[i][:,1])
        fts = fts+(label_zy_T[i][:,0], label_zy_T[i][:,1])
    label = ['Source:negative', 'Target:negative', 'Source:positive', 'Target:positive']
    '''
    fts=(zj_S[:,0],zj_S[:,1])+(zi_S[:,0],zi_S[:,1])+(zj_T[:,0],zj_T[:,1])+(zi_T[:,0],zi_T[:,1])
    label=['source joint', 'source individual', 'target joint', 'target individual']
    color = [1, 2, 1, 2]
    marker = [1, 1, 3, 3]
    line = False 
    legend = True
    util.data2plot(title=title, fts=fts, label=label, color=color, marker=marker, line=line, legend = legend, plot_enable=plot_enable, filename=filename)  
