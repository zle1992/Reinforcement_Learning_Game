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


class DoubleDQNet(object):
    __metaclass__ = ABCMeta
    """docstring for DeepQNetwork"""
    def __init__(self, 
            n_actions,
            n_features,
            learning_rate,
            reward_decay,
            replace_target_iter,
            e_greedy,
            e_greedy_increment,
            e_greedy_max,
            output_graph,
            log_dir,
            use_doubleQ ,
            model_dir,
            ):
        super(DoubleDQNet, self).__init__()
        
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate=learning_rate
        self.gamma=reward_decay
        self.replace_target_iter=replace_target_iter
        self.epsilon=e_greedy
        self.epsilon_max=e_greedy_max
        self.epsilon_increment=e_greedy_increment
        self.output_graph=output_graph
        self.lr =learning_rate
        
        self.log_dir = log_dir
        self.use_doubleQ =use_doubleQ
        self.model_dir = model_dir 
        # total learning step
        self.learn_step_counter = 0


        self.s = tf.placeholder(tf.float32,[None]+self.n_features,name='s')
        self.s_next = tf.placeholder(tf.float32,[None]+self.n_features,name='s_next')





        self.r = tf.placeholder(tf.float32,[None,],name='r')
        self.a = tf.placeholder(tf.int32,[None,],name='a')


        self.q_eval = self._build_q_net(self.s, scope='eval_net', trainable=True)
        self.q_next = self._build_q_net(self.s_next, scope='target_net', trainable=False)
        #self.q_eval4next  = tf.stop_gradient(self._build_q_net(self.s_next, scope='eval_net4next', trainable=True))
        self.q_eval4next  = self._build_q_net(self.s_next, scope='eval_net4next', trainable=False)
        



        
   

        if self.use_doubleQ:

           
            value_i = tf.to_int32(tf.argmax(self.q_eval4next,axis=1))
            range_i = tf.range(tf.shape(self.a)[0], dtype=tf.int32)
            index_a = tf.stack([range_i, value_i], axis=1)


            maxq =  tf.gather_nd(params=self.q_next,indices=index_a)
       
        else:
            maxq =  tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )


        with tf.variable_scope('q_target'):
            #只更新最大的那一列
            self.q_target = self.r + self.gamma * maxq
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)



        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        en_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net4next')

        with tf.variable_scope("hard_replacement"):
            self.target_replace_op=[tf.assign(t,e) for t,e in zip(t_params,e_params)]

        with tf.variable_scope("hard_replacement2"):
            self.target_replace_op2=[tf.assign(t,e) for t,e in zip(en_params,e_params)]
       
        self.sess = tf.Session()
        if self.output_graph:
            tf.summary.FileWriter(self.log_dir,self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        
        self.cost_his =[0]
        self.cost = 0

        self.saver = tf.train.Saver()

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        checkpoint = tf.train.get_checkpoint_state(self.model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print ("Loading Successfully")
            self.learn_step_counter = int(checkpoint.model_checkpoint_path.split('-')[-1]) + 1
    @abstractmethod
    def _build_q_net(self,x,scope,trainable):
        raise NotImplementedError

    def learn(self,data):

        self.sess.run(self.target_replace_op2)
         # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        batch_memory_s = data['s']
        batch_memory_a =  data['a']
        batch_memory_r = data['r']
        batch_memory_s_ = data['s_']
      
    
        
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory_s,
                self.a: batch_memory_a,
                self.r: batch_memory_r,
                self.s_next: batch_memory_s_,
            
            })
        #self.cost_his.append(cost)
        self.cost = cost
        # increasing epsilon
        if self.epsilon < self.epsilon_max:
            self.epsilon += self.epsilon_increment 
        else:
            self.epsilon = self.epsilon_max



        self.learn_step_counter += 1
            # save network every 100000 iteration
        if self.learn_step_counter % 10000 == 0:
            self.saver.save(self.sess,self.model_dir,global_step=self.learn_step_counter)



    def choose_action(self,s): 
        s = s[np.newaxis,:]
        aa = np.random.uniform()
        #print("epsilon_max",self.epsilon_max)
        if aa < self.epsilon:
            action_value = self.sess.run(self.q_eval,feed_dict={self.s:s})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0,self.n_actions)
        return action
