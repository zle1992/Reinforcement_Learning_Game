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


class DDPG(object):
    __metaclass__ = ABCMeta
    """docstring for ACNetwork"""
    def __init__(self, 
            n_actions,
            n_features,
            reward_decay,
            lr_a,
            lr_c,
            output_graph,
            log_dir,
            model_dir,
            TAU,
            a_bound,
            ):
        super(DDPG, self).__init__()
        
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma=reward_decay
        self.output_graph=output_graph
        self.lr_a =lr_a
        self.lr_c = lr_c
        self.log_dir = log_dir
        self.model_dir = model_dir 
        # total learning step
        self.learn_step_counter = 0
        self.TAU = TAU     # soft replacement
        self.a_bound = a_bound




        self.s = tf.placeholder(tf.float32,[None,self.n_features],name='s')
        self.s_next = tf.placeholder(tf.float32,[None,self.n_features],name='s_next')
        self.r = tf.placeholder(tf.float32,[None,1],name='r')


        with tf.variable_scope('Actor'):
            self.a = self._build_a_net(self.s, scope='eval', trainable=True)
            a_ = self._build_a_net(self.s_next, scope='target', trainable=False)

        with tf.variable_scope('Critic'):

            q  = self._build_c_net(self.s, self.a,scope='eval', trainable=True)
            q_  = self._build_c_net(self.s_next, a_,scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

       
          
        
        with tf.variable_scope('train_op_actor'):
            self.loss_actor = -tf.reduce_mean(q)
            self.train_op_actor = tf.train.AdamOptimizer(self.lr_a).minimize(self.loss_actor,var_list=self.ae_params)  
    
    

        
        with tf.variable_scope('train_op_critic'):
            
            q_target = self.r + self.gamma * q_ 
            self.loss_critic =tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.train_op_critic = tf.train.AdamOptimizer(self.lr_c).minimize(self.loss_critic,var_list=self.ce_params)

       

            # target net replacement
        self.soft_replace = [tf.assign(t, (1 - self.TAU) * t + self.TAU * e)
                               for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]
       
        self.sess = tf.Session()
        if self.output_graph:
            tf.summary.FileWriter(self.log_dir,self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        
        self.cost_his =[0]
        self.cost =0 

        self.saver = tf.train.Saver()

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        checkpoint = tf.train.get_checkpoint_state(self.model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print ("Loading Successfully")
            self.learn_step_counter = int(checkpoint.model_checkpoint_path.split('-')[-1]) + 1
   

    @abstractmethod
    def _build_a_net(self,x,scope,trainable):

        raise NotImplementedError
    def _build_c_net(self,x,scope,trainable):

        raise NotImplementedError
    def learn(self,data):

        # soft target replacement
        self.sess.run(self.soft_replace)
       

        batch_memory_s = data['s']
        batch_memory_a =  data['a']
        batch_memory_r = data['r']
        batch_memory_s_ = data['s_']
        
        batch_memory_r = batch_memory_r[:,np.newaxis]
        _, cost = self.sess.run(
            [self.train_op_actor, self.loss_actor],
            feed_dict={
                self.s: batch_memory_s,                      
            })

        _, cost = self.sess.run(
            [self.train_op_critic, self.loss_critic],
            feed_dict={
                self.s: batch_memory_s,
                self.a: batch_memory_a,
                self.r: batch_memory_r,
                self.s_next: batch_memory_s_,
           
            })


        
        self.cost_his.append(cost)
        self.cost =cost
        self.learn_step_counter += 1
            # save network every 100000 iteration
        if self.learn_step_counter % 10000 == 0:
            self.saver.save(self.sess,self.model_dir,global_step=self.learn_step_counter)



    def choose_action(self,s): 

        return self.sess.run(self.a, {self.s: s[np.newaxis,:]})[0]
        # s = s[np.newaxis,:]
       
        # probs = self.sess.run(self.acts_prob,feed_dict={self.s:s})
        # return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   