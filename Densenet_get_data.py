# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:57:56 2020

@author: Administrator
"""
# -*- coding: utf-8 -*-
"""
Created on Mon April 15 2019
@author: Ruoyu Chen
The Resnet34 networks
"""
import numpy as np
import os
from scipy.io import wavfile
from scipy import signal
from sklearn import preprocessing
from python_speech_features import mfcc,delta
import Get_data as gd
import matplotlib.pyplot as plt

def wavtomfcc(wavdict,fs):
    features_mfcc = np.zeros((len(wavdict),59*26))
    for i in range (len(wavdict)):
        mfcc_ = mfcc(wavdict[i,:],samplerate=fs,winlen=0.025,numcep=26,nfft=512)
        features_mfcc[i] = np.reshape(mfcc_,[-1,59*26])
    return features_mfcc

def get_data_(filepath):
    test_x = []
    test_x_flat = []
    label = []
    labellist = []
    labellist_cover = []
    for (dirpath, dirnames, filenames) in os.walk(filepath):
        lis = list(range(len(filenames)))
        num_count=0
        for k in range(int(len(lis)/2)):
            print('processing the file of --> '+ str(k))
            num_count = num_count + 1
            filename_text = filenames[k*2]
            filename_wav = filenames[k*2+1]
            filepath_text = os.sep.join([dirpath, filename_text])
            filepath_wav = os.sep.join([dirpath, filename_wav])    
#            print('The results of ' + filename_wav)
            tmp_test_x, tmp_test_x_flat, tmp_label, tmp_labellist, fs, tmp_labellist_cover, audio = gd.get(filepath_wav,filepath_text)

            ##################删除多余的背景框，使数据集达到平衡##################
            tmp_test_x = np.array(tmp_test_x)
            tmp_test_x_flat = np.array(tmp_test_x_flat)
            tmp_label = np.array(tmp_label)
            tmp_labellist = np.array(tmp_labellist)
            tmp_labellist_cover = np.array(tmp_labellist_cover)

            t_index_1 = np.where(tmp_labellist_cover[:,2] == 1)
            t_index_0 = np.where(tmp_labellist_cover[:,2] == 0)
            
            tmp_test_x_1 = tmp_test_x[t_index_1]
            tmp_test_x_flat_1 = tmp_test_x_flat[t_index_1]
            tmp_label_1 = tmp_label[t_index_1]
            tmp_labellist_cover_1 = tmp_labellist_cover[t_index_1]
            
            tmp_test_x_0 = tmp_test_x[t_index_0]
            tmp_test_x_flat_0 = tmp_test_x_flat[t_index_0]
            tmp_label_0 = tmp_label[t_index_0]
            tmp_labellist_cover_0 = tmp_labellist_cover[t_index_0]
            
            tmp_test_x_0 = tmp_test_x_0[:len(tmp_test_x_1)]
            tmp_test_x_flat_0 = tmp_test_x_flat_0[:len(tmp_test_x_1)]
            tmp_label_0 = tmp_label_0[:len(tmp_test_x_1)]
            tmp_labellist_cover_0 = tmp_labellist_cover_0[:len(tmp_test_x_1)]           
            
            tmp_test_x = np.vstack((tmp_test_x_0,tmp_test_x_1))
            tmp_test_x_flat = np.vstack((tmp_test_x_flat_0,tmp_test_x_flat_1))
            tmp_label = np.vstack((tmp_label_0,tmp_label_1))
            tmp_labellist_cover = np.vstack((tmp_labellist_cover_0,tmp_labellist_cover_1))
            
            if k == 0:
                test_x = tmp_test_x
                test_x_flat = tmp_test_x_flat
                label = tmp_label
                labellist = tmp_labellist
                labellist_cover = tmp_labellist_cover
            else:
                test_x = np.vstack((test_x,tmp_test_x))
                test_x_flat = np.vstack((test_x_flat,tmp_test_x_flat))
                label = np.vstack((label,tmp_label))
                labellist = np.vstack((labellist,tmp_labellist))
                labellist_cover = np.vstack((labellist_cover,tmp_labellist_cover))               
            ###################################################################
#            ###############################画中间图############################
#            length = len(audio)
#            for j in range(len(tmp_test_x_flat)):
#                if tmp_labellist_cover[j,2] == 1:
#                    label_line = np.zeros(length,int)
#                    label_line1 = np.zeros(length,int)                    
#                    label_line[tmp_test_x_flat[j][0]:tmp_test_x_flat[j][1]] = np.ones(tmp_test_x_flat[j][1]-tmp_test_x_flat[j][0])
#                    label_line1[tmp_test_x_flat[j][0]:tmp_test_x_flat[j][1]] = np.ones(tmp_test_x_flat[j][1]-tmp_test_x_flat[j][0])*(-1)   
#            
#
#                    plt.plot(list(np.arange(0,len(audio)/fs,1/fs)),audio,'r-',list(np.arange(0,len(label_line)/fs,1/fs)),label_line,'k-',list(np.arange(0,len(label_line1)/fs,1/fs)),label_line1,'k-')
#                    plt.title('label is 1' )
#                    plt.show()
#            ##################################################################
#            for j in range(len(tmp_test_x)):
#                test_x.append(tmp_test_x[j])
#            for j in range(len(tmp_test_x_flat)):
#                test_x_flat.append(tmp_test_x_flat[j])
#            for j in range(len(tmp_label)):
#                label.append(tmp_label[j])
#            for j in range(len(tmp_labellist)):
#                labellist.append(tmp_labellist[j])
#            for j in range(len(tmp_labellist_cover)):
#                labellist_cover.append(tmp_labellist_cover[j])  
    train_xx = wavtomfcc(test_x,fs) #这里正负样本不均匀
    return train_xx, test_x_flat, label, labellist ,labellist_cover

def get_train_test(data, data_list, label, labellist_cover):   
    num = len(data)
    
    ##打乱顺序#####
    shuffle_ix = np.random.permutation(np.arange(num))
    data = data[shuffle_ix]
    data_list = data_list[shuffle_ix] 
    label = label[shuffle_ix] 
    labellist_cover = labellist_cover[shuffle_ix] 
    
    train_data = data[0:int(num/10*8),:]
    test_data = data[int(num/10*8):num,:]
    
    train_data_list = data_list[0:int(num/10*8),:]
    test_data_list = data_list[int(num/10*8):num,:]
    
    train_label = label[0:int(num/10*8),:]
    test_label = label[int(num/10*8):num,:]
    
    train_labellist_cover = labellist_cover[0:int(num/10*8),:]
    test_labellist_cover = labellist_cover[int(num/10*8):num,:]
    return train_data,test_data,train_data_list,test_data_list,train_label,test_label,train_labellist_cover,test_labellist_cover
    