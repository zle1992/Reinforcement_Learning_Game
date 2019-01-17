import sys
import gym
import numpy as np 
import tensorflow as tf
sys.path.append('./')
sys.path.append('model')

from util import Memory ,StateProcessor
from DDPG import DDPG
from ACNetwork import ACNetwork
np.random.seed(1)
tf.set_random_seed(1)

import logging  # 引入logging模块
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置
# 由于日志基本配置中级别设置为DEBUG，所以一下打印信息将会全部显示在控制台上
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)



class DDPG4Pendulum(DDPG):
    """docstring for ClassName"""
    def __init__(self, **kwargs):
        super(DDPG4Pendulum, self).__init__(**kwargs)
    
    def _build_a_net(self,s,scope,trainable):
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        #w_initializer, b_initializer = None,None
        with tf.variable_scope(scope):
            e1 = tf.layers.dense(inputs=s, 
                    units=30, 
                    bias_initializer = b_initializer,
                    kernel_initializer=w_initializer,
                    activation = tf.nn.relu,
                    trainable=trainable)  
            a = tf.layers.dense(inputs=e1, 
                    units=self.n_actions, 
                    bias_initializer = b_initializer,
                    kernel_initializer=w_initializer,
                    activation = tf.nn.tanh,
                    trainable=trainable) 

        return tf.multiply(a, self.a_bound, name='scaled_a')  
    
    def _build_c_net(self,s,a,scope,trainable):
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s',self.n_features+[n_l1],trainable=trainable)
            w1_a = tf.get_variable('w1_a',[self.n_actions,n_l1],trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            
            q = tf.layers.dense(inputs=net, 
                    units=1, 
                    bias_initializer = b_initializer,
                    kernel_initializer=w_initializer,
                    activation =None,
                    trainable=trainable) 

        return q   



batch_size = 32

memory_size  =10000
env = gym.make('Pendulum-v0') #连续


n_features= [env.observation_space.shape[0]]
n_actions= env.action_space.shape[0]
a_bound = env.action_space.high
env = env.unwrapped
MAX_EP_STEPS =200
def run():
   
    RL = DDPG4Pendulum(
        n_actions=n_actions,
        n_features=n_features,
        reward_decay=0.9,
        lr_a = 0.001,
        lr_c = 0.002,
        memory_size=memory_size,
        TAU = 0.01,
        output_graph=False,
        log_dir = 'Pendulum/log/DDPG4Pendulum/',
        a_bound =a_bound,
        model_dir = 'Pendulum/model_dir/DDPG4Pendulum/'
        )

    memory = Memory(n_actions,n_features,memory_size=memory_size)
  
    var = 3  # control exploration
    step = 0
    
    for episode in range(2000):
        # initial observation
        observation = env.reset()
        ep_r = 0

        for j in range(MAX_EP_STEPS):
            
            # RL choose action based on observation
            action = RL.choose_action(observation)
            action = np.clip(np.random.normal(action, var), -2, 2)    # add randomness to action selection for exploration
            # RL take action and get_collectiot next observation and reward
            observation_, reward, done, info=env.step(action) # take a random action
           
            #print('step:%d---episode:%d----reward:%f---action:%f'%(step,episode,reward,action))
            memory.store_transition(observation, action, reward/10, observation_)
            
            if step > memory_size:
                #env.render()
                var *= .9995    # decay the action randomness
                data = memory.sample(batch_size)
                RL.learn(data)
               
            # swap observation
            observation = observation_
            ep_r += reward
            # break while loop when end of this episode
            if(episode>200): 
                env.render()  # render on the screen
            if j == MAX_EP_STEPS-1:
                print('step: ',step,
                    'episode: ', episode,
                      'ep_r: ', round(ep_r, 2),     
                      'var:',var,   
                      #loss: ',RL.cost
                      )
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()

def main():
 
    run()



if __name__ == '__main__':
    main()
    #run2()
