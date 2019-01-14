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


class DeepQNetwork(object):
    __metaclass__ = ABCMeta
    """docstring for DeepQNetwork"""
    def __init__(self, 
            n_actions,
            n_features,
            learning_rate,
            reward_decay,
            e_greedy,
            replace_target_iter,
            memory_size,
            e_greedy_increment,
            output_graph,
            log_dir,
            ):
        super(DeepQNetwork, self).__init__()
        
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate=learning_rate
        self.gamma=reward_decay
        self.epsilon_max=e_greedy
        self.replace_target_iter=replace_target_iter
        self.memory_size=memory_size
        self.epsilon_increment=e_greedy_increment
        self.output_graph=output_graph
        self.lr =learning_rate
        # total learning step
        self.learn_step_counter = 0
        self.log_dir = log_dir
       
 

        self.s = tf.placeholder(tf.float32,[None]+self.n_features,name='s')
        self.s_next = tf.placeholder(tf.float32,[None]+self.n_features,name='s_next')

        self.r = tf.placeholder(tf.float32,[None,],name='r')
        self.a = tf.placeholder(tf.int32,[None,],name='a')


        self.q_eval = self._build_q_net(self.s, scope='eval_net', trainable=True)
        self.q_next = self._build_q_net(self.s_next, scope='target_net', trainable=False)



        with tf.variable_scope('q_target'):
            self.q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)



        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope("hard_replacement"):
            self.target_replace_op=[tf.assign(t,e) for t,e in zip(t_params,e_params)]


       
        self.sess = tf.Session()
        if self.output_graph:
            tf.summary.FileWriter(self.log_dir,self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        
        self.cost_his =[]

    @abstractmethod
    def _build_q_net(self,x,scope,trainable):
        raise NotImplementedError

    def learn(self,data):


         # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        batch_memory_s = data['s'], 
        batch_memory_a =  data['a'], 
        batch_memory_r = data['r'], 
        batch_memory_s_ = data['s_'], 
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory_s,
                self.a: batch_memory_a,
                self.r: batch_memory_r,
                self.s_next: batch_memory_s_,
            })
        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon_max = self.epsilon_max + self.epsilon_increment if self.epsilon_max < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1




    def choose_action(self,s): 
        s = s[np.newaxis,:]
        aa = np.random.uniform()
        #print("epsilon_max",self.epsilon_max)
        if aa < self.epsilon_max:
            action_value = self.sess.run(self.q_eval,feed_dict={self.s:s})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0,self.n_actions)
        return action
