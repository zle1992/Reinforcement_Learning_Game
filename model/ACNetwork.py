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


class ACNetwork(object):
    __metaclass__ = ABCMeta
    """docstring for ACNetwork"""
    def __init__(self, 
            n_actions,
            n_features,
            learning_rate,
            memory_size,
            reward_decay,
            output_graph,
            log_dir,
            model_dir,
            ):
        super(ACNetwork, self).__init__()
        
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate=learning_rate
        self.gamma=reward_decay
        self.memory_size =memory_size
        self.output_graph=output_graph
        self.lr =learning_rate
        
        self.log_dir = log_dir
    
        self.model_dir = model_dir 
        # total learning step
        self.learn_step_counter = 0


        self.s = tf.placeholder(tf.float32,[None]+self.n_features,name='s')
        self.s_next = tf.placeholder(tf.float32,[None]+self.n_features,name='s_next')

        self.r = tf.placeholder(tf.float32,[None,],name='r')
        self.a = tf.placeholder(tf.int32,[None,],name='a')


        

        
        with tf.variable_scope('Critic'):

            self.v  = self._build_c_net(self.s, scope='v', trainable=True)
            self.v_  = self._build_c_net(self.s_next, scope='v_next', trainable=False)

            self.td_error =self.r + self.gamma * self.v_ - self.v
            self.loss_critic = tf.reduce_mean(tf.square(self.td_error))
            with tf.variable_scope('train'):
                self.train_op_critic = tf.train.AdamOptimizer(self.lr).minimize(self.loss_critic)

       

        with tf.variable_scope('Actor'):
            self.acts_prob = self._build_a_net(self.s, scope='actor_net', trainable=True)
            # this is negative log of chosen action
            log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.acts_prob, labels=self.a)   
            
            self.loss_actor = tf.reduce_mean(log_prob*self.td_error)
            with tf.variable_scope('train'):
                self.train_op_actor = tf.train.AdamOptimizer(self.lr).minimize(-self.loss_actor)  

       
        self.sess = tf.Session()
        if self.output_graph:
            tf.summary.FileWriter(self.log_dir,self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        
        self.cost_his =[0]


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


       

        batch_memory_s = data['s']
        batch_memory_a =  data['a']
        batch_memory_r = data['r']
        batch_memory_s_ = data['s_']
      


        _, cost = self.sess.run(
            [self.train_op_critic, self.loss_critic],
            feed_dict={
                self.s: batch_memory_s,
                self.a: batch_memory_a,
                self.r: batch_memory_r,
                self.s_next: batch_memory_s_,
           
            })

        _, cost = self.sess.run(
            [self.train_op_actor, self.loss_actor],
            feed_dict={
                self.s: batch_memory_s,
                self.a: batch_memory_a,
                self.r: batch_memory_r,
                self.s_next: batch_memory_s_,
             
            })

        
        self.cost_his.append(cost)

        self.learn_step_counter += 1
            # save network every 100000 iteration
        if self.learn_step_counter % 10000 == 0:
            self.saver.save(self.sess,self.model_dir,global_step=self.learn_step_counter)



    def choose_action(self,s): 
        s = s[np.newaxis,:]
       
        probs = self.sess.run(self.acts_prob,feed_dict={self.s:s})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   
