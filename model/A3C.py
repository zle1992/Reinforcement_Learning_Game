import os
import numpy as np 
import tensorflow as tf
from abc import ABCMeta, abstractmethod
np.random.seed(1)
tf.set_random_seed(1)

import logging  # 引入logging模块
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置
# 由于日志基本配置中级别设置为DEBUG，所以一下打印信息将会全部显示在控制台上

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)


class ACNet(object):
    __metaclass__ = ABCMeta
    """docstring for ClassName"""
    def __init__(self, 
        n_actions,
        n_features,
        sess,
        globalAC=None,
        OPT_A =None,
        OPT_C = None,
        scope='Global_Net',
        ENTROPY_BETA=0.001,
        ):
        super(ACNet, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features

        self.sess =sess
        self.ENTROPY_BETA  = ENTROPY_BETA 


        if scope=='Global_Net':
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32,[None]+self.n_features,'s')
                _ = self._build_a_net(self.s,'actor')
                _ = self._build_c_net(self.s,'critic')
                self.a_params  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                
                 
        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32,[None]+self.n_features,'s')
                self.a =  tf.placeholder(tf.int32,[None,],'a')
                self.v_target =  tf.placeholder(tf.float32,[None,1],'v_target')

                self.a_prob = self._build_a_net(self.s,'actor')
                self.v = self._build_c_net(self.s,'critic')

                self.a_params  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

                td =tf.subtract(self.v_target, self.v, name='TD_error')

                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    #选出当前动作对应的prob
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a, self.n_actions, dtype=tf.float32), axis=1, keepdims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keepdims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)
                    # a_choose_log = tf.log(self.a_prob+1e-5) *tf.one_hot(self.a,self.n_actions,dtype=tf.float32)   #(None,n_actions)
                    # log_prob = tf.reduce_sum(a_choose_log,axis=1,keep_dims=True) #(None,1)
                    # exp_v = log_prob*tf.stop_gradient(td_error)
                    # entropy = -tf.reduce_mean(self.a_prob * tf.log(self.a_prob + 1e-5),axis=1,keep_dims=True) #(None,1)
                    # self.exp_v = self.ENTROPY_BETA * exp_v + entropy
                    # self.a_loss = tf.reduce_mean(-self.exp_v) 

                with tf.name_scope('local_gradient'):
                    self.a_grads = tf.gradients(self.a_loss,self.a_params)
                    self.c_grads = tf.gradients(self.c_loss,self.c_params)


            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op =  [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op =  [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op  =  OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op  =  OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def update_global(self,feed_dict):
        
        self.sess.run([self.update_a_op,self.update_c_op],feed_dict) 

    def pull_global(self):
        self.sess.run([self.pull_a_params_op,self.pull_c_params_op])

    def choose_action(self,s):
        prob_weights = self.sess.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action



    @abstractmethod
    def _build_a_net(self,s,scope,trainable=True):
        return NotImplementedError

    @abstractmethod
    def _build_c_net(self,s,scope,trainable=True):
        return NotImplementedError
