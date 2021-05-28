# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:16:37 2020
@author: Administrator
"""
import math
import numpy as np
import os
import random
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal

def gen_wavlist(wavpath):	
	for (dirpath, dirnames, filenames) in os.walk(wavpath):
		lis = list(range(len(filenames)))
		random.shuffle(lis)
		for i in range(len(lis)):
			j = lis[i]
			filename = filenames[j]
			if filename.endswith('.wav'):     
				filepath = os.sep.join([dirpath, filename])
				fs, audio = wavfile.read(filepath)
				audio = audio/max(abs(audio))
	return audio,fs

def Shannon_Entropy(X,h,mu,sigma):
    N = len(X)
    T = 20
    A = 1/(N*h*math.sqrt(2*math.pi))
    wd = 20*sigma/T
    #print(111);
    #print(np.shape(np.arange(mu-10*sigma)));
    #print(np.shape(np.arange(mu+10*sigma)));
    #print(np.shape(np.arange(wd)));
    x = np.arange(mu-10*sigma,mu+10*sigma,wd)
    #print(np.shape(x));
    infoentropy = 0
    for k in range(T):
        pk = 0
        for n in range(N):
            pk = pk + math.exp(-0.5*((x[k]+wd/2-X[n])/h)**2)
        pk = A * pk
        infoentropy = infoentropy - pk*math.log(pk)*wd
    return infoentropy

def low_filter(x):
    (b,a) = signal.butter(N=6,Wn=0.1,btype='lowpass',output='ba')
    yn_filtered = signal.filtfilt(b,a,x) 
    return yn_filtered

def segment(audio,fs):
#    print("求取香浓熵前的一些处理--------------------")
    xlen = len(audio)
    #每段单独处理
    seg_len = np.int32(0.02*fs)
    N = seg_len+1
    seg_overlap = np.int32(seg_len/2)
    seg_num = np.int32((xlen-seg_overlap-1)/(seg_len-seg_overlap))
    Shannon = []
    th = [] 
#    print("开始求取香浓熵----------")
    for k in range(seg_num):
        X = audio[k*(seg_len-seg_overlap):k*(seg_len-seg_overlap)+N]
        X = np.reshape(X,[-1,1])
        mu = np.sum(X)/N#均值
        sigma = math.sqrt(np.dot((X-np.dot(mu,np.ones((N,1),int))).T,(X-np.dot(mu,np.ones((N,1),int))))/(N-1))#方差
        #print(sigma)
        #threshold = mu - sigma*0.1#门限
        h = 1.06*sigma*(N**(-0.2))#高斯核带宽
#        print("开始关键步骤----------" + str(k))
        infoentropy = Shannon_Entropy(X,h,mu,sigma)
#        print("关键步骤结束----------")
        #if infoentropy>threshold:
        if infoentropy>-1.5:#原本是-0.5，目前测试下来-1.0最好
            th.append(1)
        else:
            th.append(0)
        Shannon.append(infoentropy)###香农曲线
    #滤波
    Shannon = low_filter(Shannon)
#    print("香浓熵求取结束--------------------")
    #求极大值 
#    print("开始提取预选框分割--------------------")
    anchor = [] 
    points = []
    for i in range(1,seg_num-1):
        if (((Shannon[i]>Shannon[i-1])and(Shannon[i]>Shannon[i+1]))and(th[i]==1)):#设定的阈值 and(th[i]==1)
            point = i*(seg_len-seg_overlap)
            points.append(point/8000)
            #左右分界是2：1
            #考虑了边界的情况
            """
            #0.25/2000  
            if (point+1300)>xlen and (point-700)<0:
                anchor.append([10,xlen-10])
            elif (point+1300)<=xlen and (point-700)<0:
                anchor.append([10,point+1300])
            elif (point+1300)>xlen and (point-700)>=0:
                anchor.append([point-700,xlen-10])
            else:
                anchor.append([point-700,point+1300])
            if (point+700)>xlen and (point-1300)<0:
                anchor.append([10,xlen-10])
            elif (point+700)<=xlen and (point-1300)<0:
                anchor.append([10,point+700])
            elif (point+700)>xlen and (point-1300)>=0:
                anchor.append([point-1300,xlen-10])
            else:
                anchor.append([point-1300,point+700])
            if (point+1000)>xlen and (point-1000)<0:
                anchor.append([10,xlen-10])
            elif (point+1000)<=xlen and (point-1000)<0:
                anchor.append([10,point+1000])
            elif (point+1000)>xlen and (point-1000)>=0:
                anchor.append([point-1000,xlen-10])
            else:
                anchor.append([point-1000,point+1000]) 
            #0.3/2400
            if (point+1600)>xlen and (point-800)<0:
                anchor.append([10,xlen-10])
            elif (point+1600)<=xlen and (point-800)<0:
                anchor.append([10,point+1600])
            elif (point+1600)>xlen and (point-800)>=0:
                anchor.append([point-800,xlen-10])
            else:
                anchor.append([point-800,point+1600])
            if (point+800)>xlen and (point-1600)<0:
                anchor.append([10,xlen-10])
            elif (point+800)<=xlen and (point-1600)<0:
                anchor.append([10,point+800])
            elif (point+800)>xlen and (point-1600)>=0:
                anchor.append([point-1600,xlen-10])
            else:
                anchor.append([point-1600,point+800])
            if (point+1000)>xlen and (point-1000)<0:
                anchor.append([10,xlen-10])
            elif (point+1200)<=xlen and (point-1200)<0:
                anchor.append([10,point+1200])
            elif (point+1200)>xlen and (point-1200)>=0:
                anchor.append([point-1200,xlen-10])
            else:
                anchor.append([point-1200,point+1200]) 
            """
            #0.4/3200
            if (point+2100)>xlen and (point-1100)<0:
                anchor.append([10,xlen-10])
            elif (point+2100)<=xlen and (point-1100)<0:
                anchor.append([10,point+2100])
            elif (point+2100)>xlen and (point-1100)>=0:
                anchor.append([point-1100,xlen-10])
            else:
                anchor.append([point-1100,point+2100])
            if (point+1100)>xlen and (point-2100)<0:
                anchor.append([10,xlen-10])
            elif (point+1100)<=xlen and (point-2100)<0:
                anchor.append([10,point+1100])
            elif (point+1100)>xlen and (point-2100)>=0:
                anchor.append([point-2100,xlen-10])
            else:
                anchor.append([point-2100,point+1100])
            if (point+1600)>xlen and (point-1600)<0:
                anchor.append([10,xlen-10])
            elif (point+1600)<=xlen and (point-1600)<0:
                anchor.append([10,point+1600])
            elif (point+1600)>xlen and (point-1600)>=0:
                anchor.append([point-1600,xlen-10])
            else:
                anchor.append([point-1600,point+1600]) 
                          
            #0.5/4000
            if (point+2600)>xlen and (point-1400)<0:
                anchor.append([10,xlen-10])
            elif (point+2600)<=xlen and (point-1400)<0:
                anchor.append([10,point+2600])
            elif (point+2600)>xlen and (point-1400)>=0:
                anchor.append([point-1400,xlen-10])
            else:
                anchor.append([point-1400,point+2600])
            if (point+1400)>xlen and (point-2600)<0:
                anchor.append([10,xlen-10])
            elif (point+1400)<=xlen and (point-2600)<0:
                anchor.append([10,point+1400])
            elif (point+1400)>xlen and (point-2600)>=0:
                anchor.append([point-2600,xlen-10])
            else:
                anchor.append([point-2600,point+1400])
            if (point+2000)>xlen and (point-2000)<0:
                anchor.append([10,xlen-10])
            elif (point+2000)<=xlen and (point-2000)<0:
                anchor.append([10,point+2000])
            elif (point+2000)>xlen and (point-2000)>=0:
                anchor.append([point-2000,xlen-10])
            else:
                anchor.append([point-2000,point+2000])  
            #0.6/4800
            if (point+3200)>xlen and (point-1600)<0:
                anchor.append([10,xlen-10])
            elif (point+3200)<=xlen and (point-1600)<0:
                anchor.append([10,point+3200])
            elif (point+3200)>xlen and (point-1600)>=0:
                anchor.append([point-1600,xlen-10])
            else:
                anchor.append([point-1600,point+3200])
            if (point+1600)>xlen and (point-3200)<0:
                anchor.append([10,xlen-10])
            elif (point+1600)<=xlen and (point-3200)<0:
                anchor.append([10,point+1600])
            elif (point+1600)>xlen and (point-3200)>=0:
                anchor.append([point-3200,xlen-10])
            else:
                anchor.append([point-3200,point+1600])
            if (point+2000)>xlen and (point-2000)<0:
                anchor.append([10,xlen-10])
            elif (point+2400)<=xlen and (point-2400)<0:
                anchor.append([10,point+2400])
            elif (point+2400)>xlen and (point-2400)>=0:
                anchor.append([point-2400,xlen-10])
            else:
                anchor.append([point-2400,point+2400]) 
            """
            #0.7/5600
            if (point+3700)>xlen and (point-1900)<0:
                anchor.append([10,xlen-10])
            elif (point+3700)<=xlen and (point-1900)<0:
                anchor.append([10,point+3700])
            elif (point+3700)>xlen and (point-1900)>=0:
                anchor.append([point-1900,xlen-10])
            else:
                anchor.append([point-1900,point+3700])
            if (point+1900)>xlen and (point-3700)<0:
                anchor.append([10,xlen-10])
            elif (point+1900)<=xlen and (point-3700)<0:
                anchor.append([10,point+1900])
            elif (point+1900)>xlen and (point-3700)>=0:
                anchor.append([point-3700,xlen-10])
            else:
                anchor.append([point-3700,point+1900])
            if (point+2000)>xlen and (point-2000)<0:
                anchor.append([10,xlen-10])
            elif (point+2800)<=xlen and (point-2800)<0:
                anchor.append([10,point+2800])
            elif (point+2800)>xlen and (point-2800)>=0:
                anchor.append([point-2800,xlen-10])
            else:
                anchor.append([point-2800,point+2800]) 
            """
            #未考虑边界的情况
            """
            #0.25/2000
            if (point+1300)<xlen and (point-700)>0:
                anchor.append([point-700,point+1300])
            if (point-1300)>0 and (point+700)<xlen:
                anchor.append([point-1300,point+700])
            if ((point+1000)<xlen and (point-1000)>0):
                anchor.append([point-1000,point+1000])              
            #0.3/2400
            if (point+1600)<xlen and (point-800)>0:
                anchor.append([point-800,point+1600])
            if (point-1600)>0 and (point+800)<xlen:
                anchor.append([point-1600,point+800])
            if ((point+1200)<xlen and (point-1200)>0):
                anchor.append([point-1200,point+1200])            
            #0.4/3200
            if (point+2100)<xlen and (point-1100)>0:
                anchor.append([point-1100,point+2100])
            if (point-2100)>0 and (point+1100)<xlen:
                anchor.append([point-2100,point+1100])
            if ((point+1600)<xlen and (point-1600)>0):
                anchor.append([point-1600,point+1600])
            """
            """
            #0.5/4000
            if (point+2600)<xlen and (point-1400)>0:
                anchor.append([point-1400,point+2600])
            if (point-2600)>0 and (point+1400)<xlen:
                anchor.append([point-2600,point+1400])
            if ((point+2000)<xlen and (point-2000)>0):
                anchor.append([point-2000,point+2000])
            #0.6/4800
            if (point+3200)<xlen and (point-1600)>0:
                anchor.append([point-1600,point+3200])
            if (point-3200)>0 and (point+1600)<xlen:
                anchor.append([point-3200,point+1600])
            if ((point+2400)<xlen and (point-2400)>0):
                anchor.append([point-2400,point+2400])
            #0.7/5600
            if (point+3700)<xlen and (point-1900)>0:
                anchor.append([point-1900,point+3700])
            if (point-3700)>0 and (point+1900)<xlen:
                anchor.append([point-3700,point+1900])
            if ((point+2800)<xlen and (point-2800)>0):
                anchor.append([point-2800,point+2800])
            """
            """
            #0.75/6000
            if ((point+4800)<xlen and (point-4800)>0 ):
                anchor.append([point-1200,point+4800])
                anchor.append([point-3000,point+3000])
                anchor.append([point-4800,point+1200])
            """
#    print("预选框提取结束--------------------")
#    amptitu = np.ones((len(points)))
#    画图 
#
#    plt.plot(list(np.arange(0,len(audio)/8000,1/8000)),audio,'b-') 
#    plt.show()
#    plt.plot(range(len(Shannon))*((seg_len-seg_overlap)/8000),Shannon,'k-',points,amptitu,'b*')   
#    plt.show()
    return anchor 