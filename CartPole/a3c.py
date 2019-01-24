import sys
import gym
import numpy as np 
import tensorflow as tf
sys.path.append('./')
sys.path.append('model')

from util import Memory ,StateProcessor,A3CMemory
from ACNetwork import ACNetwork
from A3C import ACNet
np.random.seed(1)
tf.set_random_seed(1)

import logging  # 引入logging模块
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置
# 由于日志基本配置中级别设置为DEBUG，所以一下打印信息将会全部显示在控制台上
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)

import multiprocessing
import threading
GLOBAL_EP = 0
MAX_GLOBAL_EP = 1000
COORD = tf.train.Coordinator()
UPDATE_GLOBAL_ITER = 10
GLOBAL_RUNNING_R = []
GAMMA = 0.9

class ACNet4CartPole(ACNet):
    """docstring for ClassName"""
    def __init__(self, **kwargs):
        super(ACNet4CartPole, self).__init__(**kwargs)
    
    def _build_a_net(self,x,scope,trainable=True):
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope(scope):
            e1 = tf.layers.dense(inputs=x, 
                    units=200, 
                    bias_initializer = b_initializer,
                    kernel_initializer=w_initializer,
                    activation = tf.nn.relu6,
                    trainable=trainable)  
            q = tf.layers.dense(inputs=e1, 
                    units=self.n_actions, 
                    bias_initializer = b_initializer,
                    kernel_initializer=w_initializer,
                    activation = tf.nn.softmax,
                    trainable=trainable) 

        return q  
    
    def _build_c_net(self,x,scope,trainable=True):
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope(scope):
            e1 = tf.layers.dense(inputs=x, 
                    units=100, 
                    bias_initializer = b_initializer,
                    kernel_initializer=w_initializer,
                    activation = tf.nn.relu6,
                    trainable=trainable)  
            q = tf.layers.dense(inputs=e1, 
                    units=1, 
                    bias_initializer = b_initializer,
                    kernel_initializer=w_initializer,
                    activation =None,
                    trainable=trainable) 

        return q   

class Worker(object):
    """docstring for Worker"""
    def __init__(self,
        name,
        sess,
        ac_parms,
        globalAC,
        game_name,
         ):
        super(Worker, self).__init__()
        self.name = name
        self.sess = sess
        self.ac_parms =ac_parms
        self.globalAC = globalAC
        self.env = gym.make(game_name).unwrapped
        self.AC = ACNet4CartPole( 
            n_actions = self.ac_parms['n_actions'],
            n_features=self.ac_parms['n_features'],
            sess =self.sess,
            globalAC=globalAC,
            scope = self.name,
            OPT_A = self.ac_parms['OPT_A'],
            OPT_C =self.ac_parms['OPT_C'],
            )
        self.memory = A3CMemory()

    def work(self):
        total_step = 1
        global GLOBAL_EP
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0

            while True:
                #if self.name == 'work_0':
                self.env.render()

                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                if done: r= -5
                ep_r+=r
                self.memory.store_transition(s,a,r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done :
                    if done:
                        v_s_ =0
                    else:
                        v_s_ = self.sess.run(self.AC.v,{self.AC.s:s_[np.newaxis,:]})[0,0]

                    for r in self.memory.buffer_r[::-1]:    # reverse buffer r
                            v_s_ = r + GAMMA * v_s_

                            self.memory.buffer_v_target.append(v_s_)
                    self.memory.buffer_v_target.reverse()
                    buffer_s, buffer_a, buffer_v_target = self.memory.get_data()
                    
                    self.AC.update_global({ 
                        self.AC.s: buffer_s,
                        self.AC.a: buffer_a,
                        self.AC.v_target: buffer_v_target,
                        })
                    self.memory.clean()
                    self.AC.pull_global()
                s=s_
                total_step+=1
            
                if done:
                    if len(GLOBAL_RUNNING_R)==0:
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print( self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                              )
                    GLOBAL_EP += 1
                    break

def main():

    GAME = 'CartPole-v0'
    OUTPUT_GRAPH = True
    LOG_DIR = './log'
    N_WORKERS = multiprocessing.cpu_count()
    GLOBAL_NET_SCOPE = 'Global_Net'
    ENTROPY_BETA = 0.001
    LR_A = 0.001    # learning rate for actor
    LR_C = 0.001    # learning rate for critic
    env = gym.make(GAME)
    ac_parms ={
    'n_features': list(env.observation_space.shape),
    'n_actions':env.action_space.n,
    }
   


    sess = tf.Session()
    with tf.device('cpu/:0'):
        ac_parms['OPT_A'] = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        ac_parms['OPT_C'] = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        globalAC = ACNet4CartPole( 
        n_actions = ac_parms['n_actions'],
        n_features=ac_parms['n_features'],
        sess =sess,
        globalAC=None,
        scope = 'Global_Net',
        )  # we only need its params
        
        workers = []
        for i in range(N_WORKERS):
            i_name = 'work_%i'%i
            w = Worker(i_name,sess,ac_parms,globalAC,GAME)
            workers.append(w)

    sess.run(tf.global_variables_initializer())

    work_threads = []
    for worker in workers:
        job = lambda :worker.work()
        t = threading.Thread(target = job)
        t.start()
        work_threads.append(t)
    COORD.join(work_threads)



if __name__ == '__main__':
    main()
