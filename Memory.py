import numpy as np 
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
        d['s'] = self.s[indices][0]
        d['s_'] = self.s_[indices][0]
        d['r'] = self.r[indices][0]
        d['a'] = self.a[indices][0]
        return d