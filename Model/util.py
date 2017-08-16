
from __future__ import print_function

import tensorflow as tf
import os
import sys
import timeit
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import scipy.stats as spy
import scipy.io as sio
import numpy as np
from sklearn.manifold import TSNE


def feature_tsne(f_S, f_T):
    N_S = np.shape(f_S)[0]
    N_T = np.shape(f_T)[0]
    N = N_S + N_T
    f_dim = np.shape(f_S)[1]
    f = np.append(f_S, f_T, axis = 0)
    
    tsne_model=TSNE(n_components=2, random_state=0)
    fr=tsne_model.fit_transform(f)
    
    fr_S = fr[0:N_S, :]
    fr_T = fr[N_S:N_S+N_T, :]
    
    return [fr_S, fr_T]
def domain_feature_tsne(f_S_S, f_T_S, f_S_P, f_T_P):
    N_S_S=np.shape(f_S_S)[0]
    N_T_S=np.shape(f_T_S)[0]
    N_S_P=np.shape(f_S_P)[0]
    N_T_P=np.shape(f_T_P)[0]
    N=N_S_S+N_T_S+N_S_P+N_T_P
    f_dim=np.shape(f_S_S)[1]
    
    f=np.append(f_S_S,f_T_S,axis=0)
    f=np.append(f,f_S_P, axis=0)
    f=np.append(f,f_T_P, axis=0)
    
    tsne_model=TSNE(n_components=2, random_state=0)
    fr=tsne_model.fit_transform(f)
    
    fr_S_S=fr[0:N_S_S,:]
    fr_T_S=fr[N_S_S:N_S_S+N_T_S,:]
    fr_S_P=fr[N_S_S+N_T_S:N_S_S+N_T_S+N_S_P,:]
    fr_T_P=fr[N_S_S+N_T_S+N_S_P:N_S_S+N_T_S+N_S_P+N_T_P,:]
    
    return [fr_S_S, fr_T_S, fr_S_P, fr_T_P]

def data2plot(title, fts, label, color, marker, line=False, legend=False, plot_enable=True, filename=''):
    """
    given the data, create plot.
    """
    #chart create
    color_chart = ['red', 'blue', 'green', 'c', 'm', 'y', 'k', '#00ff77', '#ff0077', '#770055']
    marker_chart = ['None', 'o', 'x', '*']
    if line:
        linestyle='-'
    else:
        linestyle='None'
        
    #Python plot 
    markersize = 20
    fig, ax = plt.subplots(figsize=(30, 30))
    for i in range(len(fts)/2):
        ax.plot(fts[i*2], fts[i*2+1], color=color_chart[color[i]-1], marker=marker_chart[marker[i]],
            linestyle=linestyle, label=label[i], markersize=markersize)
        ax.axis('off')
    #plt.title(title)
    #if legend:
    #    plt.legend(fontsize='xx-large')
    #filename = '/home/jay/DSN/tensorflow/DSN_GAN_loss/Experimental_Result/%s.png' % (title)
    filename=filename%(title)
    plt.savefig(filename)    
    
    if plot_enable == False:
        plt.close(fig)
        
def plot_image(images, recon_imgs, n_sample):
    figsize=(10,5)
    fig=plt.figure(figsize=figsize)
    for i in range(n_sample):
        ax=fig.add_subplot(2,10,2*i+1)
        ax.imshow(images[i,:,:,:])
        ax.set_title('Ori')
        ax.axis('off')
        ax=fig.add_subplot(2,10,2*(i+1))
        ax.imshow(recon_imgs[i,:,:,:])
        ax.set_title('Rec')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]
        