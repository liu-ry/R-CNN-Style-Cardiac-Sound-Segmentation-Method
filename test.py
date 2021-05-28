# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:43:03 2020
@author: Administrator
"""
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from python_speech_features import mfcc
import Get_data as gd

def wavtomfcc(wavdict,fs):
    features_mfcc = np.zeros((len(wavdict),59*26))
    for i in range (len(wavdict)):
        mfcc_ = mfcc(wavdict[i],samplerate=fs,winlen=0.025,numcep=26,nfft=512)
        features_mfcc[i] = np.reshape(mfcc_,[-1,59*26])
    return features_mfcc

def two_boxes_iou(box1, box2):
    b1_x0, b1_x1 = box1
    b2_x0, b2_x1 = box2

    int_l_x0 = max(b1_x0, b2_x0)
    int_l_x1 = min(b1_x0, b2_x0)
    
    int_r_x0 = max(b1_x1, b2_x1)
    int_r_x1 = min(b1_x1, b2_x1)

    int_area = max((int_r_x1 - int_l_x0 + 1),0)

    b1_area = max((int_r_x1 - int_l_x1 + 1),0)
    b2_area = max((int_r_x0 - int_l_x0 + 1),0)

    # 分母加个1e-05，避免除数为 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou

def bbox_transform_env(anchors, bbox_pred, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    # The Mean and std are calculated from COCO dataset.
    # Bounding box normalization was firstly introduced in the Fast R-CNN paper.
    # See https://github.com/fizyr/keras-retinanet/issues/1273#issuecomment-585828825  for more details
    if mean is None:
        mean = np.array([0.0, 0.0])
    if std is None:
        std = np.array([0.4, 0.1])

    bbox_pred = bbox_pred * std + mean

    anchors = np.array(anchors)
    
#    anchor_widths  = anchors[:, 1] - anchors[:, 0]
#    anchors[:, 0] = bbox_pred[:, 0] * anchor_widths + anchors[:, 0]
#    anchors[:, 1] = bbox_pred[:, 1] * anchor_widths + anchors[:, 1]
    
    #################################改过之后###################################
    anchor_0 = (anchors[:,0] + anchors[:, 1]) / 2
    anchor_1 = anchors[:,1] - anchors[:, 0]
    
    anchors[:,0] = bbox_pred[:, 0] * anchor_1 + anchor_0
    anchors[:,1] = np.exp(bbox_pred[:, 1]) * anchor_1
    
    result_ = np.zeros((len(anchors),2))
    result_[:,1] = anchors[:,0] + anchors[:,1] / 2
    result_[:,0] = anchors[:,0] - anchors[:,1] / 2
    ###########################################################################
    return result_


def softmax(x, axis=1):
    # 计算每行的最大值
    row_max = x.max(axis=axis)
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max=row_max.reshape(-1, 1)
    x = x - row_max
    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s

#使用NMS方法，对结果去重
def non_max_suppression(bbox_pred,cls_pred, anchors, confidence_threshold=0.7, iou_threshold=0.2):
    # 过滤掉概率小于0.5的预测值
    anchors = np.array(anchors)
    idxs = np.argmax(cls_pred, axis=-1)#返回每一行里面最大的，也就是类别
    indexs = range(cls_pred.shape[0])#生成一个个数相等的分类索引
    cls_pred = cls_pred[indexs, idxs]#把每一个框的最大值所在的概率值找出来
    t_index = np.where(cls_pred > confidence_threshold)#概率值大于0.5的索引
    anchors = anchors[t_index]#概率值大于0.5的anchor
    cls_pred = np.expand_dims(cls_pred, -1)#在最后插入一维？？？？？？？？？？？？
    cls_pred = cls_pred[t_index]#概率值大于0.5的类别概率

    labels_ = idxs[t_index]#概率大于0.5的各标签的索引
    labels_ = np.unique(labels_)#去除数组中的重复数字，并进行排序之后输出
    idxs = np.expand_dims(idxs, -1)#在最后插入一维？？？？？？？？？？？？？？？？
    labels = idxs[t_index]#概率大于0.5的各标签
    bbox_pred = bbox_pred[t_index]#概率大于0.5的回归框？？？？？已经回归了？？？？

    bbox_pred = bbox_transform_env(anchors,bbox_pred)#把anchor回归

    predictions = np.concatenate([labels, cls_pred, bbox_pred], axis=-1)#把类别、分类概率、回归框绑定到一起
    result = []
    # print(f'正例样本数：{len(predictions)}')
    for label in labels_:#在每种类别循环
        if label != 0:
            idxs = predictions[:, 0] == label#将标签类别等于label的给找出来
            label_pred_boxes = predictions[idxs]#将其多个组合找出来
            while len(label_pred_boxes) > 0:#存在则循环，按每个类别进行寻找
                idxs = np.argsort(-label_pred_boxes[:, 1])  # 降序排序
                label_max_box = label_pred_boxes[idxs[0]]#最大的给找出来
                label_pred_boxes = label_pred_boxes[idxs[1:]]#其他的待处理
                result.append(label_max_box)#最大的放到最终的结果列表里
                box1 = label_max_box[2:4]#最大概率值框的位置信息
    
                for i, box2 in enumerate(label_pred_boxes[:, 2:4]):#其余框的位置信息
                    iou = two_boxes_iou(box1, box2)#求iou
                    if iou > iou_threshold:#如果iou过大
                        label_pred_boxes[i, 0] = 0#则设置别背景####################
                        pass
                label_pred_boxes = label_pred_boxes[label_pred_boxes[:, 0] > 0]#将剩余的非背景的boxs找出来继续循环
                if len(label_pred_boxes) == 1:#如果只剩一个
                    label_pred_boxes = np.reshape(label_pred_boxes, (1, 4))
    return np.array(result),bbox_pred  # (n_boxes, 1+1+4)

with tf.Session() as sess:
    # 加载模型 import_meta_graph填的名字meta文件的名字 
    saver = tf.train.import_meta_graph("D:/刘仁雨的文件/two-stage/two-stage/model/dense.ckpt.meta")
    saver.restore(sess, tf.train.latest_checkpoint("D:/刘仁雨的文件/two-stage/two-stage/model"))
    print('load model successfully.')

    for (dirpath, dirnames, filenames) in os.walk('D:/刘仁雨的文件/two-stage/two-stage/PASCAL/test_data'):
        lis = list(range(len(filenames)))
        
        num_cross_val = int(int(len(lis)/2)/10)
        num_count=0
        
        #初始化之类的
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('input/x-input:0')
        label = graph.get_tensor_by_name('input/y-input:0')
        boxes = graph.get_tensor_by_name('input/boxes-input:0')
        learning_rate = graph.get_tensor_by_name('learning_rate_/learning_rate__:0')
        training_flag = graph.get_tensor_by_name('training_flag_/training_flag__:0')
        #prediction = graph.get_tensor_by_name('prediction/linear/BiasAdd:0')
        x_bbox = graph.get_tensor_by_name('prediction/dense_3/BiasAdd:0')
        x_cls = graph.get_tensor_by_name('prediction/dense_4/BiasAdd:0')   
        
        ############添加的评价指标###############
        right_segments=[]
        total_segments=[]
        total_right_segments=0
        total_total_segments=0
        total_pre_segments=0
        very_bad_names = []
        very_bad_right = []
        very_bad_total = []
        
        tp_=np.array([0])
        fp_=np.array([0])
        fn_=np.array([0])
        acc_=np.array([0])
        pre_=np.array([0])
        sen_=np.array([0])
        f1_=np.array([0])
        ########################################        
        
        #time_start = time.time() #开始计时 
        for k in range(int(len(lis)/2)):
            print('processing the file of --> '+ str(k))
            num_count = num_count + 1
            filename_text = filenames[k*2]
            filename_wav = filenames[k*2+1]
            filepath_text = os.sep.join([dirpath, filename_text])
            filepath_wav = os.sep.join([dirpath, filename_wav])  
            tmp_test_x, tmp_test_x_flat, tmp_label, tmp_labellist, fs, tmp_labellist_cover, audio = gd.get(filepath_wav,filepath_text)
            test_xx = wavtomfcc(tmp_test_x,fs)
            
            print('The results of ' + filename_wav)

            test_feed_dict = {
                x: test_xx,
                label: tmp_label,
                boxes: tmp_labellist_cover,
                learning_rate: 0,
                training_flag : False
            }
            res_bbox, res_cls = sess.run([x_bbox, x_cls], feed_dict=test_feed_dict)
#            res_cls = sess.run(x_cls, feed_dict=test_feed_dict)
            res_cls = softmax(res_cls)
            result,bbox_pred = non_max_suppression(res_bbox, res_cls, tmp_test_x_flat, confidence_threshold=0.0, iou_threshold=0.0)
            
            
            if result.size == 0:
                print("识别为空")
                continue            
            labellist = result[:,2:4]
            labellist = np.sort(labellist,0)
            if labellist[0,0] <= 0:
                labellist[0,0] = 1
            if labellist[len(labellist)-1,1] >= len(audio):
                labellist[len(labellist)-1,1] = len(audio)-1

            #画图
            length = len(audio)
            label_line = np.zeros(length,int)
            label_line1 = np.zeros(length,int)
            for j in range(int(len(labellist[:,0]))): 
                label_line[int(labellist[j,0]):int(labellist[j,1])] = np.ones(int(labellist[j,1])-int(labellist[j,0]))
                label_line1[int(labellist[j,0]):int(labellist[j,1])] = np.ones(int(labellist[j,1])-int(labellist[j,0]))*(-1)      
            
            label_line2 = np.zeros(length,int)
            label_line3 = np.zeros(length,int)
            for j in range(int(len(tmp_labellist[:,0]))): 
                label_line2[int(tmp_labellist[j,0]):int(tmp_labellist[j,1])] = np.ones(int(tmp_labellist[j,1])-int(tmp_labellist[j,0]))
                label_line3[int(tmp_labellist[j,0]):int(tmp_labellist[j,1])] = np.ones(int(tmp_labellist[j,1])-int(tmp_labellist[j,0]))*(-1)      
            ##########################统计分割正确个数##########################
            count_right_segment = 0 
            for i in range(len(labellist[:,0])):
                for j in range(len(tmp_labellist[:,0])):
                    if labellist[i,0]<tmp_labellist[j,0] and labellist[i,1]>tmp_labellist[j,1]:
                        count_right_segment =count_right_segment + 1
            right_segments.append(count_right_segment)
            total_segments.append(len(tmp_labellist[:,0]))
            total_right_segments = total_right_segments + count_right_segment
            total_total_segments = total_total_segments + len(tmp_labellist[:,0])
            total_pre_segments = total_pre_segments + len(labellist[:,0])
            #将识别结果不好的保存下来
            if count_right_segment<int(len(tmp_labellist[:,0])*1.0/10*7)+1:
                very_bad_names.append(filename_wav)
                very_bad_right.append(count_right_segment)
                very_bad_total.append(len(tmp_labellist[:,0]))     
            ###################################################################                         
            #画图            
            plt.plot(list(np.arange(0,len(audio)/fs,1/fs)),audio,'r-',list(np.arange(0,len(label_line)/fs,1/fs)),label_line,'b-',list(np.arange(0,len(label_line1)/fs,1/fs)),label_line1,'b-')
            plt.title(filepath_wav )
            plt.show()
            plt.plot(list(np.arange(0,len(audio)/fs,1/fs)),audio,'r-',list(np.arange(0,len(label_line2)/fs,1/fs)),label_line2,'k-',list(np.arange(0,len(label_line3)/fs,1/fs)),label_line3,'k-')
            plt.title(filepath_wav )
            plt.show()   
#########################################统计##################################
            print('The correct number of heart sound segments: ' + str(count_right_segment))
            print('The total number of heart sound segments: ' + str(len(tmp_labellist[:,0])))
            print('The segmentation accuracy is: ' + str(count_right_segment/len(tmp_labellist[:,0]))) 
            
            tp = count_right_segment
            fp = len(labellist[:,0]) - count_right_segment
            fn = len(tmp_labellist[:,0]) - count_right_segment
            tp_=np.append(tp_,tp)
            fp_=np.append(fp_,fp)
            fn_=np.append(fn_,fn)                
            acc_ = np.append(acc_,tp/(tp+fp+fn))
            pre_ = np.append(pre_,tp/(tp+fp))
            sen_ = np.append(sen_,tp/(tp+fn))
            f1_ = np.append(f1_,2*tp/(2*tp+fp+fn))
        TP = total_right_segments
        FP = total_pre_segments - total_right_segments
        FN = total_total_segments - total_right_segments
        print(TP)
        print(FP)
        print(FN)
        ACC = TP/(TP+FP+FN)
        PRE = TP/(TP+FP)
        SEN = TP/(TP+FN)   
        F1 = 2*TP/(2*TP+FP+FN)
        print("The ACC is：" + str(ACC))
        print("The PRE is：" + str(PRE))
        print("The SEN is：" + str(SEN))
        print("The F1 is：" + str(F1))
###############################################################################