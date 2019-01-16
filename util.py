import numpy as np 
import tensorflow as tf
np.random.seed(1)
class Memory(object):
    """docstring for Memory"""
    def __init__(self,
            n_actions,
            n_features,
            memory_size):
        super(Memory, self).__init__()
        self.memory_size = memory_size
        self.cnt =0 

        self.s = np.zeros([memory_size]+n_features)
        self.a = np.zeros([memory_size,])
        self.r =  np.zeros([memory_size,])
        self.s_ = np.zeros([memory_size]+n_features)
        
    def store_transition(self,s, a, r, s_):
        #logging.info('store_transition')
        index = self.cnt % self.memory_size
        self.s[index] = s
        self.a[index] = a
        self.r[index] =  r
        self.s_[index] =s_
        self.cnt+=1

    def sample(self,n):
        #logging.info('sample')
        #assert self.cnt>=self.memory_size,'Memory has not been fulfilled'
        N = min(self.memory_size,self.cnt)
        indices = np.random.choice(N,size=n)
        d ={}
        d['s'] = self.s[indices]
        d['s_'] = self.s_[indices]
        d['r'] = self.r[indices]
        d['a'] = self.a[indices]
        return d


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