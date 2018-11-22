# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 18:03:30 2018

@author: Dell
"""

import numpy as np
import matplotlib.pyplot as plt

# File path for the cifar-100 dataset
file_path = "cifar-100-python/"

# Number of classfication classes
number_classes = 20

# Reads cifar dataset files and returns a dictionary
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# loads classes from dataset file and returns a list of labels for the 20 superclasses
def load_classes(file_name):
    # reads the file
    dict = unpickle(file_name)
    # superclass labels 
    raw = dict[b'coarse_label_names']
    # convert from binary string to normal string
    names = [x.decode('utf-8') for x in raw]  
    return names

# Prints classes labels
def print_classes(labels):
    for i in range(len(labels)):
        print(i ," " , labels[i])

# Converts pixel value to a value between 0 ... 1
def convert_images(raw):
    raw_float = np.array(raw,dtype = float) /255.0
    return raw_float

def zero_center(imgs, mean_img):
    imgs = imgs - mean_img
    return imgs

# loads trainig images and their labels
def load_data(file_name, reshape = False, one_hot = False):
    # reads file
    data = unpickle(file_name)
    # raw images. (N x 3072) matrix
    raw_imgs = data[b'data']     
    # class label of each image.  (N x 1) matrix                  
    labels  = np.array(data[b'coarse_labels'])
    
    if (reshape):
        raw_imgs = raw_imgs.reshape(raw_imgs.shape[0] , 3, 32,32)  #.transpose(0,2,3,1)   # Reshape to [width: 3, height: 32, depth: 32]
     
    return raw_imgs, labels

def loadCifar100(train_num , reshape = True , center = True):
    Xtr , Ytr = load_data(file_path + "train")
    Xtest, Ytest = load_data(file_path + "test")
    
    s = np.random.permutation(Xtr.shape[0])
    Xtr, Ytr   = Xtr[s] , Ytr[s]
    
    Xtrain , Ytrain = Xtr[0:train_num] , Ytr[0:train_num]
    Xvalid , Yvalid = Xtr[train_num:] , Ytr[train_num:]
    if (center):
        mean_img = Xtrain.mean()
    
        Xtrain = Xtrain - mean_img
        Xvalid = Xvalid - mean_img
        Xtest = Xtest - mean_img
    
    if (reshape):
        Xtrain = Xtrain.reshape(Xtrain.shape[0] , 3, 32,32)  #.transpose(0,2,3,1)   # Reshape to [width: 3, height: 32, depth: 32]
        Xvalid = Xvalid.reshape(Xvalid.shape[0] , 3,32,32)
        Xtest  = Xtest.reshape(Xtest.shape[0], 3,32,32)
     
    return Xtrain , Ytrain , Xvalid , Yvalid, Xtest, Ytest
    
    
    

def ccrn(Ypredicted, Ytest):
    Ytest_l = Ytest.tolist()
    print(Ytest)
    print(Ypredicted)
    class_occurence  = [ Ytest_l.count(i) for i in range(number_classes)]  
    correct_y = np.zeros((number_classes), dtype = np.int)
    for i in range(number_classes): 
        for j in range(len(Ytest)):
            if ( Ytest[j] == Ypredicted[j] and Ytest[j] == i):
                correct_y[i] = correct_y[i] + 1
    ccrn = correct_y / class_occurence
    return ccrn
 