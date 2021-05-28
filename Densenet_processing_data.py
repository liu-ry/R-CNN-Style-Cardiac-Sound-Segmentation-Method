# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:32:10 2020

@author: Administrator
"""

import numpy as np
import Densenet_get_data as ggd

###处理数据
train_xx, test_x_flat, label_, labellist ,labellist_cover  = ggd.get_data_('./data_raw')##########有问题
train_xx,test_data,train_data_list,test_data_list,train_yy,test_label,train_labellist_cover,test_labellist_cover = ggd.get_train_test(train_xx,test_x_flat,label_,labellist_cover )
np.save('train_xx.npy', train_xx)
np.save('test_data.npy', test_data)
np.save('train_data_list.npy', train_data_list)
np.save('test_data_list.npy', test_data_list)
np.save('test_label.npy', test_label)
np.save('train_labellist_cover.npy', train_labellist_cover)
np.save('test_labellist_cover.npy', test_labellist_cover)
np.save('train_yy.npy', train_yy)
print("数据处理完毕")
