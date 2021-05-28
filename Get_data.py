# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:09:09 2020

@author: Administrator
"""


import tensorflow as tf
import numpy as np
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
import Shannon_th
from sklearn.cluster import MeanShift
from scipy import interpolate
from python_speech_features import mfcc

def txt_to_matrix(filename,fs):
    file=open(filename)
    lines=file.readlines()
    rows=len(lines)#文件行数
    datamat=np.zeros((rows,1))#初始化矩阵
    row=0
    for line in lines:
        line=line.strip().split('\t')#strip()默认移除字符串首尾空格或换行符
        #datamat[row,:]=line[:]####自己数据集，pascal数据集用这个
        datamat[row,:]=float(line[0])####PhysioNet 2016 数据集用这个
        row+=1
    datamat=datamat.reshape([int(len(datamat)/2),2])
    return np.int32(datamat*fs)####自己数据集，PhysioNet数据集为np.int32(datamat*fs)

def get_wavlist(filepath_wav,filepath_text):	
    fs, audio = wavfile.read(filepath_wav)     
    audio = audio/max(abs(audio))
    labellist = txt_to_matrix(filepath_text,fs)
    return audio, fs, labellist

def get_wav(audio,wav_flat):
    wav = []
    length = len(wav_flat)
    for i in range(length):
        if (wav_flat[i][1]-wav_flat[i][0])<4800:
            y = interpolate.interp1d(range(wav_flat[i][1]-wav_flat[i][0]), audio[wav_flat[i][0]:wav_flat[i][1]], kind='cubic')
            xint = np.linspace(0,wav_flat[i][1]-wav_flat[i][0]-1,4800)
            wav_same_length = y(xint)            
            for j in range(len(wav_same_length)):
                wav.append(wav_same_length[j])
        elif (wav_flat[i][1]-wav_flat[i][0])>4800:#############这里有个大问题，直接裁剪，会把边边的给去掉
            n = np.int32((wav_flat[i][1]-wav_flat[i][0]-4800)/2)
            wav_same_length = audio[wav_flat[i][0]+n:wav_flat[i][1]-n]
            for j in range(len(wav_same_length)):
                wav.append(wav_same_length[j])
        else:
            for j in range(wav_flat[i][0],wav_flat[i][1],1):
                wav.append(audio[j])
    return wav[0:int(len(wav)/4800)*4800]#????????????

def get_normal(wav_x):
    for i in range (len(wav_x)):
        wav_x[i]=wav_x[i]/max(abs(wav_x[i]))
    return wav_x

def get_label(test_x_flat,labellist):
    label = np.zeros((len(test_x_flat),2))
    labellist_cover = np.zeros((len(test_x_flat),3))
    for i in range(len(test_x_flat)):
        label[i,0]=1
        for j in range(len(labellist)):
            if (test_x_flat[i][0] <= labellist[j][0]) and (test_x_flat[i][1] >= labellist[j][1]) and (j ==0 or (test_x_flat[i][0] > labellist[j-1][1])) and (j ==len(labellist)-1 or (test_x_flat[i][1] < labellist[j+1][0])):
                label[i,0]=0
                label[i,1]=1
#                labellist_cover[i,0]=labellist[j,0]
#                labellist_cover[i,1]=labellist[j,1]
                GTx = (labellist[j][0] + labellist[j][1]) / 2
                GTw = labellist[j][1] - labellist[j][0]
                anchorx = (test_x_flat[i][0] + test_x_flat[i][1]) / 2
                anchorw = test_x_flat[i][1] - test_x_flat[i][0]
                labellist_cover[i,0] = (GTx - anchorx) / anchorw
                labellist_cover[i,1] = np.log(GTw / anchorw)
                labellist_cover[i,2]=1
                break
    return label,labellist_cover

def get(filepath_wav,filepath_text):
    audio,fs,labellist = get_wavlist(filepath_wav,filepath_text)
    test_x_flat = Shannon_th.segment(audio,fs)
    label, labellist_cover = get_label(test_x_flat,labellist)
    test_ = get_wav(audio,test_x_flat)#得到数据，包括裁剪与差值
    test_x = np.reshape(test_,[-1,4800])
    test_x = get_normal(test_x)
    # test_x是具体每一段的数据，test_x_flat是每一段的左右边界值，label是类别标签，
    # labellist是具体的心音短标签，labellist_cover是与label对应的对应位置的真实位置标签
    return test_x, test_x_flat, label, labellist , fs , labellist_cover ,audio