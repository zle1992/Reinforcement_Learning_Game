
import os, sys
import numpy as np 
import tensorflow as tf


sys.path.append("game/")
import wrapped_flappy_bird as game
sys.path.append('..')
sys.path.append('../model')
from util import Memory ,StateProcessor
from DeepQNetwork import DeepQNetwork
from DoubleDQNet import DoubleDQNet
np.random.seed(1)
tf.set_random_seed(1)

import logging  # 引入logging模块
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置
# 由于日志基本配置中级别设置为DEBUG，所以一下打印信息将会全部显示在控制台上

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)

class DeepQNetwork4FlappyBird(DoubleDQNet):
    """docstring for ClassName"""
    def __init__(self, **kwargs):
        super(DeepQNetwork4FlappyBird, self).__init__(**kwargs)
    
    def _build_q_net(self,x,scope,trainable):
        #w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        #w_initializer, b_initializer =tf.contrib.layers.xavier_initializer(), tf.constant_initializer(0.01)
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

           
            # f_pool1 = tf.layers.max_pooling2d(
            #         inputs=f_conv1,
            #         pool_size=(2,2),
            #         strides=(2,2),
            #         padding='SAME',
            #         data_format='channels_last',)

            # print('f_pool1',f_pool1.shape)



            f_conv2 = tf.layers.conv2d(
                    inputs=f_conv1,
                    filters = 64,
                    kernel_size =4,
                    strides=(2, 2),
                    padding="SAME",
                    data_format='channels_last',
                    bias_initializer = b_initializer,
                    kernel_initializer=w_initializer,
                    activation=tf.nn.relu,
                    trainable=trainable)

            
            # f_pool2 = tf.layers.max_pooling2d(
            #         inputs=f_conv2,
            #         pool_size=(2,2),
            #         strides=(2,2),
            #         padding='SAME',
            #         data_format='channels_last',)

            # print('f_pool2',f_pool2.shape)



            f_conv3 = tf.layers.conv2d(
                    inputs=f_conv2,
                    filters = 64,
                    kernel_size =3,
                    strides=(1, 1),
                    padding="SAME",
                    data_format='channels_last',
                     bias_initializer = b_initializer,
                    kernel_initializer=w_initializer,
                    activation=tf.nn.relu,
                    trainable=trainable)

    


            # f_pool3 = tf.layers.max_pooling2d(
            #         inputs=f_conv3,
            #         pool_size=(2,2),
            #         strides=(2,2),
            #         padding='SAME',
            #         data_format='channels_last',)
            # print('f_pool3',f_pool3.shape)


            f_conv3_flatten =tf.layers.flatten(f_conv3)
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
#


# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
      
        return self.action_space.sample()






batch_size = 32
n_features= [288, 512, 3]
n_actions= 2
memory_size  =50000 
episode_count = 1000
input_shape = [84,84,4]
OBSERVE = 100000. # timesteps to observe before training
if __name__ == '__main__':
  
    RL = DeepQNetwork4FlappyBird(
        n_actions=n_actions,
        n_features=input_shape,
        learning_rate=0.01,
        reward_decay=0.9,
        replace_target_iter=200,
        memory_size=memory_size,
        e_greedy=0.85,
        e_greedy_increment=0.0001,
        e_greedy_max=0.95,
        output_graph=True,
        log_dir = 'log/DeepQNetwork4FlappyBird/',
        use_doubleQ = True,
        model_dir = 'model_dir/DeepQNetwork4FlappyBird/'
        )
    memory = Memory(n_actions,input_shape,memory_size=memory_size)
    sp = StateProcessor(n_features,input_shape)

    

    


    step = 0
    ep_r = 0
    reward = 0
    done = False


    for episode in range(episode_count):
        game_state = game.GameState()
        a_onthot = np.zeros(n_actions)
        a_onthot[0] = 1
        o, reward, done = game_state.frame_step(a_onthot)
        o = sp.process(RL.sess, o)
        observation = np.stack([o]*input_shape[-1],axis=2)

        while True:

            action = RL.choose_action(observation)
            a_onthot = np.zeros(n_actions)
            a_onthot[action] = 1
            o_, reward, done=game_state.frame_step(a_onthot) # take a random action
            o_ = sp.process(RL.sess, o_)
            o_ = o_[:,:,np.newaxis]
            observation_ = np.append(observation[:,:,1:],o_,axis=2)
                   
            memory.store_transition(observation, action, reward, observation_)
            
            
            if (step > OBSERVE) :#and (step % 1 == 0):
               
                data = memory.sample(batch_size)
                RL.learn(data)
                #print('step:%d----reward:%f---action:%d'%(step,reward,action))
            # swap observation
            observation = observation_
            ep_r += reward
            # break while loop when end of this episode
           
            if done:
                print('step',step,
                    'episode: ', episode,
                      'ep_r: ', round(ep_r, 2),
                      ' epsilon: ', round(RL.epsilon_max, 2),
                      'loss',RL.cost_his[-1]
                      )
                ep_r = 0

                break
            step += 1
   
    env.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
#    gym.upload(outdir)
    # Syntax for uploading has changed
