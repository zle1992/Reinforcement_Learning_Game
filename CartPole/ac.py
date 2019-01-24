import sys
import gym
import numpy as np 
import tensorflow as tf
sys.path.append('./')
sys.path.append('model')

from util import Memory ,StateProcessor
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



class ACNetwork4CartPole(ACNetwork):
    """docstring for ClassName"""
    def __init__(self, **kwargs):
        super(ACNetwork4CartPole, self).__init__(**kwargs)
    
    def _build_a_net(self,x,scope,trainable):
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope(scope):
            e1 = tf.layers.dense(inputs=x, 
                    units=32, 
                    bias_initializer = b_initializer,
                    kernel_initializer=w_initializer,
                    activation = tf.nn.relu,
                    trainable=trainable)  
            q = tf.layers.dense(inputs=e1, 
                    units=self.n_actions, 
                    bias_initializer = b_initializer,
                    kernel_initializer=w_initializer,
                    activation = tf.nn.softmax,
                    trainable=trainable) 

        return q  
    
    def _build_c_net(self,x,scope,trainable):
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope(scope):
            e1 = tf.layers.dense(inputs=x, 
                    units=32, 
                    bias_initializer = b_initializer,
                    kernel_initializer=w_initializer,
                    activation = tf.nn.relu,
                    trainable=trainable)  
            q = tf.layers.dense(inputs=e1, 
                    units=1, 
                    bias_initializer = b_initializer,
                    kernel_initializer=w_initializer,
                    activation =None,
                    trainable=trainable) 

        return q   



batch_size = 32

memory_size  =100
#env = gym.make('Breakout-v0') #离散
env = gym.make('CartPole-v0') #离散


n_features= list(env.observation_space.shape)
n_actions= env.action_space.n
env = env.unwrapped

def run():
   
    RL = ACNetwork4CartPole(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=0.01,
        reward_decay=0.9,
     
        memory_size=memory_size,
    
        output_graph=True,
        log_dir = 'log/ACNetwork4CartPole/',
    
        model_dir = 'model_dir/ACNetwork4CartPole/'
        )

    memory = Memory(memory_size=memory_size)
  

    step = 0
    ep_r = 0
    for episode in range(2000):
        # initial observation
        observation = env.reset()

        while True:
            

            # RL choose action based on observation
            action = RL.choose_action(observation)
            # logging.debug('action')
            # print(action)
            # RL take action and get_collectiot next observation and reward
            observation_, reward, done, info=env.step(action) # take a random action
            
            # the smaller theta and closer to center the better
            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
            reward = r1 + r2




            memory.store_transition(observation, action, reward, observation_)
            
            
            if (step > 200) and (step % 1 == 0):
               
                data = memory.sample(batch_size)
                RL.learn(data)
                #print('step:%d----reward:%f---action:%d'%(step,reward,action))
            # swap observation
            observation = observation_
            ep_r += reward
            # break while loop when end of this episode
            if(episode>700): 
                env.render()  # render on the screen
            if done:
                print('step: ',step,
                    'episode: ', episode,
                      'ep_r: ', round(ep_r, 2),        
                      'loss: ',RL.cost_his[-1]
                      )
                ep_r = 0

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
