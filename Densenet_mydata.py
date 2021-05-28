# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:32:10 2020

@author: Administrator
"""
import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import Densenet_get_data as ggd
import matplotlib.pyplot as plt
import time
import loss
import my_detectnet as md

growth_k = 12
nb_block = 2 
init_learning_rate = 1e-2
epsilon = 1e-8 
dropout_rate = 0.4

# Momentum Optimizer will use 
nesterov_momentum = 0.9
weight_decay = 1e-4

# Label & batch_size
class_num = 2
batch_size = 1000

total_epochs = 30

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network

def Global_Average_Pooling(x, stride=1):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """
    return global_avg_pool(x, name='Global_avg_pooling')

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')

##################################  attention  ################################
slim = tf.contrib.slim

def combined_static_and_dynamic_shape(tensor):
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape

def convolutional_block_attention_module(feature_map, index, inner_units_ratio=0.5):
    with tf.variable_scope("cbam_%s" % (index)):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)
        # channel attention
        channel_avg_weights = tf.nn.avg_pool(value=feature_map,ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],strides=[1, 1, 1, 1],padding='VALID')
        channel_max_weights = tf.nn.max_pool(value=feature_map,ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],strides=[1, 1, 1, 1],padding='VALID')
        channel_avg_reshape = tf.reshape(channel_avg_weights,[feature_map_shape[0], 1, feature_map_shape[3]])
        channel_max_reshape = tf.reshape(channel_max_weights,[feature_map_shape[0], 1, feature_map_shape[3]])
        channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)
 
        fc_1 = tf.layers.dense(inputs=channel_w_reshape,units=feature_map_shape[3] * inner_units_ratio,name="fc_1",activation=tf.nn.relu)
        fc_2 = tf.layers.dense(inputs=fc_1,units=feature_map_shape[3],name="fc_2",activation=None)
        channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
        channel_attention = tf.nn.sigmoid(channel_attention, name="channel_attention_sum_sigmoid")
        channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, 1, feature_map_shape[3]])
        feature_map_with_channel_attention = tf.multiply(feature_map, channel_attention)
        # spatial attention
        channel_wise_avg_pooling = tf.reduce_mean(feature_map_with_channel_attention, axis=3)
        channel_wise_max_pooling = tf.reduce_max(feature_map_with_channel_attention, axis=3)
 
        channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],1])
        channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],1])
 
        channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
        spatial_attention = slim.conv2d(channel_wise_pooling,1,[7, 7],padding='SAME',activation_fn=tf.nn.sigmoid,scope="spatial_attention_conv")
        feature_map_with_attention = tf.multiply(feature_map_with_channel_attention, spatial_attention)
        return feature_map_with_attention   
###############################################################################

class DenseNet():
    def __init__(self, x, nb_blocks, filters, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.model = self.Dense_net(x)

    def bottleneck_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            # x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            in_channel = x.shape[-1]
            x = conv_layer(x, filter=int(in_channel)/2, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)
            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)
            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))
            layers_concat.append(x)
            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)
            x = Concatenation(layers_concat)
            return x

    def Dense_net(self, input_x):
        x = conv_layer(input = input_x, filter = 2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')
        x = Max_Pooling(x, pool_size=[3,3], stride=2)
        for i in range(self.nb_blocks) :
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))
            x = convolutional_block_attention_module(x,i)  #  attention
        """
        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')
        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')
        x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')
        """
        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        print("Densenet 最后一层的神经元个数：")
        print(x)
        
        ###### Classification and regression #######
        x = tf.layers.dense(inputs=x, units=100)
        x = Relu(x)
        x = tf.layers.dense(inputs=x, units=100)
        x = Relu(x)
        
        x_bbox, x_cls = md.detectnet(x,2)
        #########################        
#        x = tf.reshape(x, [-1, 10])
#        x = tf.nn.softmax(x)
        return x_bbox, x_cls
 
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,shape=[None, 59*26],name='x-input')

    label = tf.placeholder(tf.float32,(None,2),name='y-input')
    boxes = tf.placeholder(tf.float32,(None,3),name='boxes-input')
    with tf.name_scope('batch_images_'):
        batch_images = tf.reshape(x,[-1, 59, 26, 1],name='batch_images__')

with tf.name_scope('training_flag_'):
    training_flag = tf.placeholder(tf.bool,name='training_flag__')

with tf.name_scope('learning_rate_'):
    learning_rate = tf.placeholder(tf.float32,name='learning_rate__')

with tf.name_scope('prediction'):
    x_bbox, x_cls = DenseNet(x=batch_images, nb_blocks=nb_block, filters=growth_k, training=training_flag).model
    print(x_bbox)
    print(x_cls)
 

#x = tf.placeholder(tf.float32, shape=[None, 59*26])
#print(x)
#batch_images = tf.reshape(x, [-1, 59, 26, 1])
#label = tf.placeholder(tf.float32, shape=[None, 2])
#
#training_flag = tf.placeholder(tf.bool)
#
#learning_rate = tf.placeholder(tf.float32, name='learning_rate')
##tf.reset_default_graph()
#logits = DenseNet(x=batch_images, nb_blocks=nb_block, filters=growth_k, training=training_flag).model
#print(logits)
    
    
cost_cla = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=x_cls))
#cost_cla = tf.reduce_mean(loss.focal(y_true = label, y_pred = x_cls))
cost_reg = tf.reduce_mean(loss.smooth_l1(y_true = boxes, y_pred = x_bbox))

cost = cost_cla + 5*cost_reg


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
train = optimizer.minimize(cost)

correct_prediction = tf.equal(tf.argmax(x_cls, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph)
    global_step = 0
    epoch_learning_rate = init_learning_rate
    precise0 = []
    precise1 = []
    loss0 = []
    loss1 = []
    train_xx,train_yy,test_data,test_label = ggd.get_data('D:/刘仁雨的文件/two-stage/two-stage/tmp/train_data','D:/刘仁雨的文件/two-stage/two-stage/tmp/test_data')##########
    time_start = time.time() #开始计时 

    ###处理数据
    train_xx, test_x_flat, label_, labellist ,labellist_cover  = ggd.get_data_('D:/刘仁雨的文件/two-stage/two-stage/physionet/training_data')##########有问题
    train_xx,test_data,train_data_list,test_data_list,train_yy,test_label,train_labellist_cover,test_labellist_cover = ggd.get_train_test(train_xx,test_x_flat,label_,labellist_cover )
    
#    np.save('train_xx.npy', train_xx)
#    np.save('test_data.npy', test_data)
#    np.save('train_data_list.npy', train_data_list)
#    np.save('test_data_list.npy', test_data_list)
#    np.save('test_label.npy', test_label)
#    np.save('train_labellist_cover.npy', train_labellist_cover)
#    np.save('test_labellist_cover.npy', test_labellist_cover)
#    np.save('train_yy.npy', train_yy)
    print("数据处理完毕")

#    ###加载数据
#    train_xx_health = np.load('D:/刘仁雨的文件/two-stage/two-stage/my_data/train_data_all/processed_data2(mydata_all_health)/train_xx.npy')    
#    test_data_health = np.load('D:/刘仁雨的文件/two-stage/two-stage/my_data/train_data_all/processed_data2(mydata_all_health)/test_data.npy')    
#    train_data_list_health = np.load('D:/刘仁雨的文件/two-stage/two-stage/my_data/train_data_all/processed_data2(mydata_all_health)/train_data_list.npy')    
#    test_data_list_health = np.load('D:/刘仁雨的文件/two-stage/two-stage/my_data/train_data_all/processed_data2(mydata_all_health)/train_xx.npy')    
#    test_label_health = np.load('D:/刘仁雨的文件/two-stage/two-stage/my_data/train_data_all/processed_data2(mydata_all_health)/test_label.npy')    
#    train_labellist_cover_health = np.load('D:/刘仁雨的文件/two-stage/two-stage/my_data/train_data_all/processed_data2(mydata_all_health)/train_labellist_cover.npy')    
#    test_labellist_cover_health = np.load('D:/刘仁雨的文件/two-stage/two-stage/my_data/train_data_all/processed_data2(mydata_all_health)/test_labellist_cover.npy')    
#    train_yy_health = np.load('D:/刘仁雨的文件/two-stage/two-stage/my_data/train_data_all/processed_data2(mydata_all_health)/train_yy.npy')  
#    #####混合健康与as数据######
#    train_xx_as = np.load('D:/刘仁雨的文件/two-stage/two-stage/my_data/train_data_all/processed_data3(mydata_all_as)/train_xx.npy')    
#    test_data_as = np.load('D:/刘仁雨的文件/two-stage/two-stage/my_data/train_data_all/processed_data3(mydata_all_as)/test_data.npy')    
#    train_data_list_as = np.load('D:/刘仁雨的文件/two-stage/two-stage/my_data/train_data_all/processed_data3(mydata_all_as)/train_data_list.npy')    
#    test_data_list_as = np.load('D:/刘仁雨的文件/two-stage/two-stage/my_data/train_data_all/processed_data3(mydata_all_as)/train_xx.npy')    
#    test_label_as = np.load('D:/刘仁雨的文件/two-stage/two-stage/my_data/train_data_all/processed_data3(mydata_all_as)/test_label.npy')    
#    train_labellist_cover_as = np.load('D:/刘仁雨的文件/two-stage/two-stage/my_data/train_data_all/processed_data3(mydata_all_as)/train_labellist_cover.npy')    
#    test_labellist_cover_as = np.load('D:/刘仁雨的文件/two-stage/two-stage/my_data/train_data_all/processed_data3(mydata_all_as)/test_labellist_cover.npy')    
#    train_yy_as = np.load('D:/刘仁雨的文件/two-stage/two-stage/my_data/train_data_all/processed_data3(mydata_all_as)/train_yy.npy') 
#    train_xx,test_data,train_data_list,test_data_list,test_label,train_labellist_cover,test_labellist_cover,train_yy = ggd.mix_health_as(train_xx_health,test_data_health,train_data_list_health,test_data_list_health,test_label_health,train_labellist_cover_health,test_labellist_cover_health,train_yy_health,train_xx_as,test_data_as,train_data_list_as,test_data_list_as,test_label_as,train_labellist_cover_as,test_labellist_cover_as,train_yy_as)
#    
#    #############################使训练达到平衡#################################
#    t_index_1 = np.where(train_labellist_cover[:,2] == 1)
#    t_index_0 = np.where(train_labellist_cover[:,2] == 0)
#    
#    train_xx_1 = train_xx[t_index_1]
#    train_data_list_1 = train_data_list[t_index_1]
#    train_yy_1 = train_yy[t_index_1]
#    train_labellist_cover_1 = train_labellist_cover[t_index_1]
#    
#    train_xx_0 = train_xx[t_index_0]
#    train_data_list_0 = train_data_list[t_index_0]
#    train_yy_0 = train_yy[t_index_0]
#    train_labellist_cover_0 = train_labellist_cover[t_index_0]  
#    
#    train_xx_0 = train_xx_0[:len(train_xx_1)]
#    train_data_list_0 = train_data_list_0[:len(train_xx_1)]
#    train_yy_0 = train_yy_0[:len(train_xx_1)]
#    train_labellist_cover_0 = train_labellist_cover_0[:len(train_xx_1)]     
#    
#    train_xx = np.vstack((train_xx_0,train_xx_1))
#    train_data_list = np.vstack((train_data_list_0,train_data_list_1))
#    train_yy = np.vstack((train_yy_0,train_yy_1))
#    train_labellist_cover = np.vstack((train_labellist_cover_0,train_labellist_cover_1))  
#    ###########################################################################

#    #####加载数据#####
#    train_xx = np.load('D:/刘仁雨的文件/two-stage/two-stage/train_xx.npy')    
#    test_data = np.load('D:/刘仁雨的文件/two-stage/two-stage/test_data.npy')    
#    train_data_list = np.load('D:/刘仁雨的文件/two-stage/two-stage/train_data_list.npy')    
#    test_data_list = np.load('D:/刘仁雨的文件/two-stage/two-stage/train_xx.npy')    
#    test_label = np.load('D:/刘仁雨的文件/two-stage/two-stage/test_label.npy')    
#    train_labellist_cover = np.load('D:/刘仁雨的文件/two-stage/two-stage/train_labellist_cover.npy')    
#    test_labellist_cover = np.load('D:/刘仁雨的文件/two-stage/two-stage//test_labellist_cover.npy')    
#    train_yy = np.load('D:/刘仁雨的文件/two-stage/two-stage/train_yy.npy') 
#    print("load data successfully")
    
    train_t1=0
    train_t0=0
    test_t1=0
    test_t0=0
    for i in range(len(train_labellist_cover)):
        if train_labellist_cover [i,2] == 1:
            train_t1=train_t1+1
        else:
            train_t0=train_t0+1
    for i in range(len(test_labellist_cover)):
        if test_labellist_cover [i,2] == 1:
            test_t1=test_t1+1
        else:
            test_t0=test_t0+1    
    time_end = time.time()  #结束计时
    for epoch in range(total_epochs):
        print("epoch = " + str(epoch))
        if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
            epoch_learning_rate = epoch_learning_rate / 10

        ##打乱顺序#####
        shuffle_ix = np.random.permutation(np.arange(len(train_xx)))
        train_x = train_xx[shuffle_ix]
        train_y = train_yy[shuffle_ix] 
        train_boxes = train_labellist_cover[shuffle_ix] 
        
#        train_x = train_xx
#        train_y = train_yy
#        train_boxes = train_labellist_cover
        total_batch = int(len(train_x) / batch_size)###########################
##################################关键步骤######################################
        for step in range(total_batch):   
            batch_x = train_x[step*batch_size:(step+1)*batch_size,:]###########
            batch_y = train_y[step*batch_size:(step+1)*batch_size,:]###########
            batch_boxes = train_boxes[step*batch_size:(step+1)*batch_size,:]###
            
            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                boxes: batch_boxes,
                learning_rate: epoch_learning_rate,
                training_flag : True
            }
            _, loss_train = sess.run([train, cost], feed_dict=train_feed_dict)
            train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict=train_feed_dict)
            writer.add_summary(train_summary, global_step=epoch)
            global_step += 1
            # accuracy.eval(feed_dict=feed_dict)
            
            print("Step:",global_step,"Loss:",loss_train,"Training accuracy:",train_accuracy)
            test_feed_dict = {
                x: test_data,
                label: test_label,
                boxes: test_labellist_cover,
                learning_rate: epoch_learning_rate,
                training_flag : False
            }
            _, loss_test = sess.run([train, cost], feed_dict=train_feed_dict)
            accuracy_rates = sess.run(accuracy, feed_dict=test_feed_dict)
            print("Step:",global_step,"Loss:",loss_test,'/ Accuracy =',accuracy_rates)
            precise0.append(train_accuracy)
            precise1.append(accuracy_rates)
            loss0.append(loss_train)
            loss1.append(loss_test)  
            # writer.add_summary(test_summary, global_step=epoch)
    plt.figure(1)
    plt.xlim(0, global_step)
    plt.ylim(0, 1.1)
    plt.plot(list(range(len(precise0))),precise0,'r-',list(range(len(precise1))),precise1,'k-')
    plt.figure(2)
    plt.xlim(0, global_step)
    plt.plot(list(range(len(loss0))),loss0,'r-',list(range(len(loss1))),loss1,'k-')
    saver.save(sess=sess, save_path='./model/dense.ckpt')