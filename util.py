import numpy as np 
import tensorflow as tf
from collections import deque
import random
np.random.seed(1)
import logging  # 引入logging模块
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置
# 由于日志基本配置中级别设置为DEBUG，所以一下打印信息将会全部显示在控制台上
class Memory(object):
    """docstring for Memory"""
    def __init__(self, memory_size):
        super(Memory, self).__init__()
        self.memory_size = memory_size
        self.cnt =0 
        self.Deque = deque()
       
       
        
    def store_transition(self,s, a, r, s_):
        #logging.info('store_transition')
        self.Deque.append((s, a, r, s_))
        if self.cnt > self.memory_size:
            self.Deque.popleft() 
        self.cnt+=1

    def sample(self,n):
        #logging.info('sample')
        #assert self.cnt>=self.memory_size,'Memory has not been fulfilled'
        N = min(self.memory_size,self.cnt)
        minibatch = random.sample(self.Deque,n) 
        data ={}
        data['s'] = np.asarray([d[0] for d in minibatch])
        data['a'] = np.asarray([d[1] for d in minibatch])
        data['r'] = np.asarray([d[2] for d in minibatch])
        data['s_'] =np.asarray([d[3] for d in minibatch])

        # print('data_s',data['s'].shape)
        # print('data_a',data['a'].shape)
        # print('data_r',data['r'].shape)
        # print('data_s_',data['s_'].shape)

        return data
class A3CMemory(object):
    """docstring for ClassName"""
    def __init__(self,):
        super(A3CMemory, self).__init__()
        self.buffer_s = []
        self.buffer_a = []
        self.buffer_r = []
        self.buffer_v_target = []
    def store_transition(self,s, a, r):
        self.buffer_s.append(s)
        self.buffer_a.append(a)
        self.buffer_r.append(r)

    def clean(self):
         self.buffer_s = []
         self.buffer_a = []
         self.buffer_r = []
         self.buffer_v_target = []
    def get_data(self,fly_data = False):

        if fly_data:
            tmp= [s[np.newaxis,:,:,:] for s in self.buffer_s] 
            buffer_s =np.concatenate(tmp)
        else:
            buffer_s = np.vstack(self.buffer_s)

        buffer_a = np.array(self.buffer_a)
        buffer_v_target = np.vstack(self.buffer_v_target)
        
        #logging.error(buffer_a.shape)
        #logging.error(buffer_s.shape)
        return buffer_s, buffer_a, buffer_v_target 





    

class StateProcessor(object):
    """
    Processes a raw Atari iamges. Resizes it and converts it to grayscale.
    图片的预处理。
    """
    def __init__(self,shape=[210, 160, 3],output_shape=[84,84]):

        self.shape = shape
        self.output_shape = output_shape[:2]
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=self.shape, dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, self.output_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84, 1] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })