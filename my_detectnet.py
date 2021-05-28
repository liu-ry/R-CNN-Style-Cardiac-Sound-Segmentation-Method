#-- coding: utf-8 --
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten
import math
# 在p3-p7层上选用的anchors拥有的像素区域大小从32x32到512x512,每层之间的长度是两倍的关系。
# 每个金字塔层有三种长宽比例[1:2 ,1:1 ,2:1]，有三种尺寸大小[2^0, 2^（1/3)， 2^（2/3)]。
# 总共便是每层9个anchors。大小从32像素到813像素。 32 = 32 * 2^0, 813 = 512 * 2^（2/3)

# 分类子网络和回归子网络的参数是分开的，但结构却相似。都是用小型FCN网络，
# 将金字塔层作为输入，接着连接4个3x3的卷积层，fliter为金字塔层的通道数（论文中是256)，
# 每个卷积层后都有RELU激活函数，这之后连接的是fliter为KA
# （K是目标种类数，A是每层的anchors数，论文中是9)的3x3的卷积层，激活函数是sigmoid。

def detectnet(input,num_classes):####这里的list是retinanet里面每一层，在我这里其实就一层
    x_bbox = create_reg_model(input)
    x_cls = create_cls_model(input,num_classes)
    #res_array = []

###############################################################################
#    def concat(x):
#        y = tf.concat(x, axis=1)
#        return y
#    res_bbox = tf.keras.layers.Lambda(concat)(res_bbox)
#    res_cls = tf.keras.layers.Lambda(concat)(res_cls) 
###############################################################################
    return x_bbox, x_cls

#########################################################################################################################
def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return tf.nn.relu(network)
    
def create_reg_model(input,filters=256, n_anchors=1):
#    x_bbox = conv_layer(input,filters,kernel=[3,3],stride=1)
#    x_bbox = conv_layer(x_bbox,filters,kernel=[3,3],stride=1)
#    x_bbox = conv_layer(x_bbox,filters,kernel=[3,3],stride=1)
#    x_bbox = conv_layer(x_bbox,filters,kernel=[3,3],stride=1)
#    x_bbox = tf.layers.conv2d(x_bbox, n_anchors * 2, kernel_size=[3,3], strides=1, padding='same')
#    #########2020/11/10###############
#    x_bbox = flatten(x_bbox)
    x_bbox = tf.layers.dense(inputs=input, units=2)
#    ##################################
#    x_bbox = tf.reshape(x_bbox,(-1, 2))
    #这里可能需要加分类激活函数
    return x_bbox

def create_cls_model(input,num_classes, filters=256, n_anchors=1):
#    x_cls = conv_layer(input,filters,kernel=[3,3],stride=1)
#    x_cls = conv_layer(x_cls,filters,kernel=[3,3],stride=1)
#    x_cls = conv_layer(x_cls,filters,kernel=[3,3],stride=1)
#    x_cls = conv_layer(x_cls,filters,kernel=[3,3],stride=1)    
#    x_cls = tf.layers.conv2d(x_cls, n_anchors * num_classes, kernel_size=[3,3], strides=1, padding='same')
    #########2020/11/10###############
#    x_cls = flatten(x_cls)
    x_cls = tf.layers.dense(inputs=input, units=2)
#    x_cls = tf.nn.softmax(x_cls)
    ##################################
    return x_cls



