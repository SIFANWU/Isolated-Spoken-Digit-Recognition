# -*- coding: utf-8 -*- 
import sys
import numpy as np
from sklearn.mixture import GaussianMixture

class HMM:
    """ A left-to-right no-skip Hidden Markov Model with Gaussian emissions

    Attributes:
        num_states: total number of HMM states (including two non-emitting states)
        states: a list of HMM states
        log_transp: transition probability matrix (log domain): a[i,j] = log_transp[i,j]
    """

    def __init__(self, num_states=3, num_mixtures=1, self_transp=0.9):
        """Performs GMM evaluation given a list of GMMs and utterances

        Args:
            num_states: number of emitting states 
            num_mixture: number of Gaussian mixture components for each state
            self_transp: initial self-transition probability for each state 初始转移概率
        """
        self.num_states = num_states
        self.states = [GaussianMixture(n_components=num_mixtures, covariance_type='diag', 
            init_params='kmeans', max_iter=10) for state_id in range(self.num_states)]
        
    
        # Initialise transition probability for a left-to-right no-skip HMM
        # For a 3-state HMM, this looks like
        #   [0.9, 0.1, 0. ],
        #   [0. , 0.9, 0.1],
        #   [0. , 0. , 0.9]
        transp = np.diag((1-self_transp)*np.ones(self.num_states-1,),1) + np.diag(self_transp*np.ones(self.num_states,))
        
        
        
        self.log_transp = np.log(transp)
        #这个为转移概率矩阵
        # np.diag(v,k) 
        #
        # np.diag(np.ones(2,)*0.1,0):
        #array([[0.1, 0. ],
        #      [0. , 0.1]])
        #
        #np.diag(np.ones(2,)*0.1,1)
        #array([[0. , 0.1, 0. ],
        #       [0. , 0. , 0.1],
        #      [0. , 0. , 0. ]])
        #
    def forward(self,obs):
        T = obs.shape[0]
        log_outp = np.array([self.states[state_id].score_samples(obs).T for state_id in range(self.num_states)])
        
        #Initial state probs PIs
        initial_dist = np.zeros(self.num_states)
        initial_dist[1:] = -float('inf')
        
        alpha = np.zeros((self.num_states, T), dtype='int')
        alpha[:,0] = initial_dist[:] + log_outp[:,0] 
        
        for t in range(1,T):
            for n in range(self.num_states):
                alpha[n,t] = np.sum(alpha[:,t-1]+self.log_transp[:,n])+log_outp[n,t]
        
        prob=np.sum(alpha[:,T-1])
        return prob, alpha
                

    def viterbi_decoding(self, obs):
        """Performs Viterbi decoding

        Args:
            obs: a sequence of observations [T x dim]
            
        Returns:
            log_prob: log-likelihood
            state_seq: most probable state sequence
        """       

        # Length of obs sequence
        T = obs.shape[0]
        # Precompute log output probabilities [num_states x T] log_outp输出矩阵
        log_outp = np.array([self.states[state_id].score_samples(obs).T for state_id in range(self.num_states)])
        
     
        # Initial state probs PI
        initial_dist = np.zeros(self.num_states) # prior prob = log(1) for the first state
        initial_dist[1:] = -float('inf') # prior prob = log(0) for all the other states
        #等同于Pi
        # Back-tracing matrix [num_states x T]
        #= beta
        back_pointers = np.zeros((self.num_states, T), dtype='int')
        # -----------------------------------------------------------------
        # INITIALISATION 初始化
        # YOU MAY WANT TO DEFINE THE DELTA VARIABLE AS A MATRIX INSTEAD 
        # OF AN ARRAY FOR AN EASIER IMPLEMENTATION.使用矩阵代替array
        # -----------------------------------------------------------------
        # Initialise the Delta probability
        probs = log_outp[:,0] + initial_dist 
        #显示矩阵第一行
        # -----------------------------------------------------------------
        # RECURSION 递推
        # -----------------------------------------------------------------
        for t in range(1, T):
            probs= probs + self.log_transp.T
            # STEP 1. Add all transitions to previous best probs
            temp = np.amax(probs,axis=1)
            # STEP 2. Select the previous best state from all transitions into a state
            #temp = np.amax(probs, axis=0) 列取 axis=1 横取
            back_pointers[:, t] = np.argmax(probs, axis=1)
            # STEP 3. Record back-trace information in +back_pointers
            probs = temp + log_outp[:, t]
            # STEP 4. Add output probs to previous best probs
            '''
            Comments:
            why use self.log_transp.T? 
                                                    T         
                     State1 State2 State3 State4   >>>       State1 State2 State3 State4
        (obs1)State1  0.9    0.1                      State1  0.9
        (obs2)State2         0.9    0.1               State2  0.1    0.9
        (obs3)State3                0.9    0.1        State3         0.1   0.9
        (obs4)State4                       0.9        State4               0.1     0.9
                                                             (obs1) (obs2) (obs3) (obs4)
            Vertical correspondence becomes horizontal correspondence
            '''
        
        # -----------------------------------------------------------------
        # SAVE THE GLOBAL LOG LIKELIHOOD IN log_prob AS A RETURN VALUE.
        # THE GLOBAL LOG LIKELIHOOD WILL BE THE VALUE FROM THE LAST STATE 
        # AT TIME T (THE LAST FRAME).
        # -----------------------------------------------------------------
        log_prob = probs[-1]
        #Choose the biggest one

        state_seq = np.empty((T,), dtype='int')
        state_seq[T-1] = self.num_states - 1
        
        
        #from back to begin step=-1 
        #The last one is done  state_seq[T-1] = self.num_states - 1
        for t in range(T, 1,-1):
            state_seq[t-2] = back_pointers[state_seq[t-1],t-1]
        
        # -----------------------------------------------------------------
        # RETURN THE OVERAL LOG LIKELIHOOD log_prob AND THE MOST PROBABLE
        # STATE SEQUENCE state_seq
        # YOU MAY WANT TO CHECK IF THE STATE SEQUENCE LOOKS REASONABLE HERE
        # E.G. FOR A 5-STATE HMM IT SHOULD LOOK SOMETHING LIKE
        #     0 0 0 0 1 1 1 2 2 2 2 3 3 3 3 3 3 3 4 4
        # -----------------------------------------------------------------
        return log_prob, state_seq


