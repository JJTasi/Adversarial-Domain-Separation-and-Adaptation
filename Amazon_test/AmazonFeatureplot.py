from __future__ import print_function

import os
import sys
import timeit
import pickle

import scipy.io as sio
import numpy as np
import tensorflow as tf

sys.path.insert(0, "/home/jay/DSN/DSN_with_diff_tri/Model/")
import nnet as nn
import criteria	as er
import util

def features_plot(features,labels_source, labels_target ,sample_n, description):
    # features = [source_share_feature, target_share_feature, source_private_feature, target_private_feature]
    y_dim=np.shape(labels_source)[1]

    S_labels=labels_source[0:sample_n]
    T_labels=labels_target[0:sample_n]
    
    source_share_feature=features[0][0:sample_n,:]
    target_share_feature=features[1][0:sample_n,:]
    source_private_feature=features[2][0:sample_n,:]
    target_private_feature=features[3][0:sample_n,:]
    
    z_ssf, z_tsf, z_spf, z_tpf = util.feature_tsne(source_share_feature, target_share_feature, source_private_feature, target_private_feature)
    
    #save tsen feature
    np.savetxt('z_ssf.txt', z_ssf)
    np.savetxt('z_tsf.txt', z_tsf)
    np.savetxt('z_spf.txt', z_spf)
    np.savetxt('z_tpf.txt', z_tpf)
    
    label_z_ssf=[]
    label_z_tsf=[]
    label_z_spf=[]
    label_z_tpf=[]
    
    for i in range(y_dim):
        label_z_ssf.append( z_ssf[np.where(S_labels[:,i]==1)[0],:] )
        label_z_tsf.append( z_tsf[np.where(T_labels[:,i]==1)[0],:] )
        label_z_spf.append( z_spf[np.where(S_labels[:,i]==1)[0],:] )
        label_z_tpf.append( z_tpf[np.where(T_labels[:,i]==1)[0],:] )
    
    # Source zy feature 
    title = 'Source_z_feature_%s' % (description)
    fts=()
    for i in range(y_dim):
        fts=fts+(label_z_ssf[i][:,0],label_z_ssf[i][:,1])
        fts=fts+(label_z_spf[i][:,0],label_z_spf[i][:,1])
        
    label = ['negative_share', 'positive_share', 'negative_private', 'positive_private']
   
    color=[1, 2, 3 , 7]
    marker=[1,1, 1, 1]
    line = False
    legend=True
    util.data2plot(title=title, fts=fts, label=label, color=color, marker=marker, line=line, legend = legend, plot_enable=True)
    
    # Target zy feature
    title = 'Target_z_feature_%s' % (description)
    fts=()
    for i in range(y_dim):
        fts=fts+(label_z_tsf[i][:,0],label_z_tsf[i][:,1])
        fts=fts+(label_z_tpf[i][:,0],label_z_tpf[i][:,1])
        
    label = ['negative_share', 'positive_share', 'negative_private', 'positive_private']
    color=[1, 2, 3, 7]
    marker=[3,3, 3,3]
    line = False
    legend=True
    util.data2plot(title=title, fts=fts, label=label, color=color, marker=marker, line=line, legend = legend, plot_enable=True)
    
    # Both source, target zy feature
    title = 'Zy_feature_%s' % (description)
    fts = ()
    tmp = ()
    for i in range(y_dim):
        fts=fts+(label_z_ssf[i][:,0],label_z_ssf[i][:,1])
        fts=fts+(label_z_spf[i][:,0],label_z_spf[i][:,1])
        fts=fts+(label_z_tsf[i][:,0],label_z_tsf[i][:,1])
        fts=fts+(label_z_tpf[i][:,0],label_z_tpf[i][:,1])
        
    label = ['negative_share_S', 'positive_share_S', 'negative_private_S', 'positive_private_S', 'negative_share_T', 'positive_share_T', 'negative_private_T', 'positive_private_T']
    color = [1, 2, 3, 7, 1, 2, 3, 7]
    marker = [1, 1, 1, 1, 3, 3, 3, 3]
    line = False 
    legend = True
    util.data2plot(title=title, fts=fts, label=label, color=color, marker=marker, line=line, legend = legend, plot_enable=True)
    
    
    