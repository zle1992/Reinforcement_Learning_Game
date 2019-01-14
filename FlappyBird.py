import logging
import os, sys

import gym
from gym.wrappers import Monitor

import gym_ple
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

class DeepQNetwork4FlappyBird(DeepQNetwork):
    """docstring for ClassName"""
    def __init__(self, **kwargs):
        super(DeepQNetwork4FlappyBird, self).__init__(**kwargs)
    
    def _build_q_net(self,x,scope,trainable):
        #w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
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

        

#FlappyBird  上下
#(512, 288, 3)


# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
      
        return self.action_space.sample()

batch_size = 128
n_features= [512, 288, 3]
n_actions=2
memory_size  =2000

if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    RL = DeepQNetwork4FlappyBird(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=memory_size,
        e_greedy_increment=None,
        output_graph=True,
        log_dir = 'log/DeepQNetwork4FlappyBird/',
        )
    memory = Memory(n_actions,n_features,memory_size=memory_size)
  

    step = 0
    ep_r = 0





    env = gym.make('FlappyBird-v0' if len(sys.argv)<2 else sys.argv[1])

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = Monitor(env, directory=outdir, force=True)

    # This declaration must go *after* the monitor call, since the
    # monitor's seeding creates a new action_space instance with the
    # appropriate pseudorandom number generator.
    env.seed(0)










    agent = RandomAgent(env.action_space)
    ep_r = 0
    episode_count = 1000
    reward = 0
    done = False
    for episode in range(episode_count):
        observation = env.reset()

        while True:




            #action = agent.act(ob, reward, done)
            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)
            #print(observation_.shape)
        
            memory.store_transition(observation, action, reward, observation_)
            
            
            if (step > 200) and (step % 5 == 0):
               
                data = memory.sample(batch_size)
                RL.learn(data)
                #print('step:%d----reward:%f---action:%d'%(step,reward,action))
            # swap observation
            observation = observation_
            ep_r += reward
            # break while loop when end of this episode
            if(episode>200): 
                env.render()  # render on the screen
            if done:
                print('episode: ', episode,
                      'ep_r: ', round(ep_r, 2),
                      ' epsilon: ', round(RL.epsilon_max, 2))
                ep_r = 0

                break
            step += 1
   
    env.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
#    gym.upload(outdir)
    # Syntax for uploading has changed
