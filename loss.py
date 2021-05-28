# -*- coding: utf-8 -*-
# @Time    : 2020/3/5 23:25
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : loss.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np
def focal(y_true, y_pred, alpha=0.25, gamma=2.0):
#    labels = y_true[:, :, :-1]
#    anchor_state = y_true[:, :, -1] # -1 for ignore, 0 for background, 1 for object
#    pre_cls = y_pred
    labels = y_true
    anchor_state = y_true[:,1]
    pre_cls =y_pred

#    #filter out ignore anchors
#    indices = tf.where(tf.math.not_equal(anchor_state, -1))
#    labels = tf.gather_nd(labels, indices)
#    pre_cls = tf.gather_nd(pre_cls,indices)

    #compute the focal loss
    alpha_factor = tf.ones_like(labels) * alpha#创建一个和labels一样的张量，元素都为1；再乘以alpha
    alpha_factor = tf.where(tf.math.equal(labels,1), alpha_factor, 1 - alpha_factor)#第一个元素为true，则为第二个元素，否则为第三个元素
    focal_weight = tf.where(tf.math.equal(labels,1), 1 - pre_cls, pre_cls)
    focal_weight = alpha_factor * focal_weight ** gamma#整个前面加的系数
    cls_loss = focal_weight * tf.keras.backend.binary_crossentropy(labels,pre_cls)#完整的loss 

    # compute the normalizer: the number of positive anchors
    normalizer = tf.where(tf.math.equal(anchor_state, 1))#返回anchor_state中位1的地方
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)#数据类型转换为   float32
    normalizer = tf.maximum(tf.keras.backend.cast_to_floatx(1.0), normalizer)#返回1.0与normalizer里面的最大值

    return tf.keras.backend.sum(cls_loss) / normalizer
    pass

def smooth_l1(y_true, y_pred,sigma=3.0):
#    sigma_squared = sigma ** 2
    sigma_squared = 1.0
    # separate target and state
    regression = y_pred
    regression_target = y_true[:,:2] #回归标签
    anchor_state = y_true[:,2] #标记是前景还是背景
    # filter out "ignore" anchors
    indices = tf.where(tf.equal(anchor_state, 1))
    regression = tf.gather_nd(regression, indices)
    regression_target = tf.gather_nd(regression_target, indices)

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = regression - regression_target
    regression_diff = tf.abs(regression_diff)
    regression_loss = tf.where(tf.less(regression_diff, 1.0 / sigma_squared),
                               0.5 * sigma_squared * tf.pow(regression_diff, 2),
                               regression_diff - 0.5 / sigma_squared)

    # compute the normalizer: the number of positive anchors
    normalizer = tf.maximum(1, tf.shape(indices)[0])
    normalizer = tf.cast(normalizer, dtype=tf.float32)
    return tf.keras.backend.sum(regression_loss) / normalizer
    pass