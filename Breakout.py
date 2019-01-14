import sys
import gym
import numpy as np 
import tensorflow as tf
sys.path.append('model')

from Memory import Memory
from DeepQNetwork import DeepQNetwork

np.random.seed(1)
tf.set_random_seed(1)

import logging  # 引入logging模块
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置
# 由于日志基本配置中级别设置为DEBUG，所以一下打印信息将会全部显示在控制台上

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)


        
class DeepQNetwork4Breakout(DeepQNetwork):
    """docstring for ClassName"""
    def __init__(self, **kwargs):
        super(DeepQNetwork4Breakout, self).__init__(**kwargs)
    
    def _build_q_net(self,x,scope,trainable):
        #w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        w_initializer, b_initializer = tf.initializers.truncated_normal(stddev=0.01), tf.constant_initializer(0.01)
     
        with tf.variable_scope(scope):
            f_conv1 = tf.layers.conv2d(
                    inputs=x,
                    filters = 32,
                    kernel_size =8,
                    strides=(4, 4),
                    padding="SAME",
                    data_format='channels_last',
                    bias_initializer = b_initializer,
                    kernel_initializer=w_initializer,
                    activation=tf.nn.relu,
                    trainable=trainable)

           
            f_pool1 = tf.layers.max_pooling2d(
                    inputs=f_conv1,
                    pool_size=(2,2),
                    strides=(2,2),
                    padding='SAME',
                    data_format='channels_last',)

            print('f_pool1',f_pool1.shape)



            f_conv2 = tf.layers.conv2d(
                    inputs=f_pool1,
                    filters = 64,
                    kernel_size =4,
                    strides=(2, 2),
                    padding="SAME",
                    data_format='channels_last',
                    bias_initializer = b_initializer,
                    kernel_initializer=w_initializer,
                    activation=tf.nn.relu,
                    trainable=trainable)

            
            f_pool2 = tf.layers.max_pooling2d(
                    inputs=f_conv2,
                    pool_size=(2,2),
                    strides=(2,2),
                    padding='SAME',
                    data_format='channels_last',)

            print('f_pool2',f_pool2.shape)



            f_conv3 = tf.layers.conv2d(
                    inputs=f_pool2,
                    filters = 64,
                    kernel_size =3,
                    strides=(1, 1),
                    padding="SAME",
                    data_format='channels_last',
                     bias_initializer = b_initializer,
                    kernel_initializer=w_initializer,
                    activation=tf.nn.relu,
                    trainable=trainable)

    


            f_pool3 = tf.layers.max_pooling2d(
                    inputs=f_conv3,
                    pool_size=(2,2),
                    strides=(2,2),
                    padding='SAME',
                    data_format='channels_last',)
            print('f_pool3',f_pool3.shape)


            f_conv3_flatten =tf.layers.flatten(f_pool3)
            print('f_conv3_flatten',f_conv3_flatten.shape)


 
            
            fc1_out = tf.layers.dense(inputs=f_conv3_flatten, 
                units=512, 
                bias_initializer = b_initializer,
                kernel_initializer=w_initializer,
                activation = tf.nn.relu,
                trainable=trainable)   




            print('fc1_out',fc1_out.shape)


            output = tf.layers.dense(inputs=fc1_out, 
                units=self.n_actions, 
                bias_initializer = b_initializer,
                kernel_initializer=w_initializer,
                trainable=trainable)   



            print('output',output.shape)
        return output


batch_size = 128

memory_size  =5000
env = gym.make('Breakout-v0') #离散
#env = gym.make('CartPole-v0') #离散


n_features= list(env.observation_space.shape)
n_actions= env.action_space.n

env = env.unwrapped

def run():
   


    RL = DeepQNetwork4Breakout(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=memory_size,
        e_greedy_increment=None,
        output_graph=True,
        log_dir = 'log/DeepQNetwork4Breakout/',
        )
    memory = Memory(n_actions,n_features,memory_size=memory_size)
  

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
            
          
            memory.store_transition(observation, action, reward, observation_)
            
            
            if (step > 200) and (step % 2 == 0):
               
                data = memory.sample(batch_size)
                RL.learn(data)
                #print('step:%d----reward:%f---action:%d'%(step,reward,action))
            # swap observation
            observation = observation_
            ep_r += reward
            # break while loop when end of this episode
            if(episode>1000): 
                env.render()  # render on the screen
            if done:
                print('episode: ', episode,
                      'ep_r: ', round(ep_r, 2),
                      ' epsilon: ', round(RL.epsilon_max, 2))
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
