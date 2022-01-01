import random
import argparse
from copy import deepcopy
#from datetime import datetime
#from tensorboard_logger import configure, log_value
import numpy as np

class WoLFPHCAgent():
    def __init__(self,n_actions, n_states, learningRate, discountFactor, winDelta=0.01,
                 loseDelta=0.1, epsilon=0.0):

        self.alpha = learningRate
        self._gamma = discountFactor
        self._winDelta = winDelta
        self._loseDelta = loseDelta

        self._n_actions = np.arange(n_actions)
        self._n_states = n_states

        self._avg_pi = (1 / n_actions) * np.ones(shape=(n_states, n_actions))
        self._pi = (1 / n_actions) * np.ones(shape=(n_states, n_actions))
        self._Q = np.zeros(shape=(n_states, n_actions))
        self._C = np.zeros(shape=(n_states,))

        self._epsilon = 1
        self._min_epsilon = epsilon

    def learn(self, s, r, a, nextObs):
        self._Q[s, a] = (1 - self.alpha) * self._Q[s, a] + self.alpha * (r + self._gamma * np.max(self._Q[nextObs, :]))


    def chooseAction(self, s):
        '''
        Choose action based on e greedy policy
        '''
        if np.random.random() > self._epsilon:  # Choose greedy action
            # a = self.possibleActions[self.Q(self._s1).argmax()]
            # a = self.possibleActions[np.argmax(self.pi(self._s1))]
            a = np.random.choice(self._n_actions, p=self._pi[s])
        else:  # Choose randomly
            a = np.random.choice(self._n_actions)
        return a

    def calculateAveragePolicyUpdate(self, s):
        self._C[s] += 1
        C = self._C[s]
        for i, a in enumerate(self._n_actions):
            self._avg_pi[s][i] += (1 / C) * (self._pi[s][i] - self._avg_pi[s][i])

    def calculatePolicyUpdate(self, s, a):
        # Find the suboptimal actions
        argmax = self._Q[s].argmax()
        # Decide which lr to use
        qs = self._Q[s]
        sum_avg = self._avg_pi[s].dot(qs)
        sum_norm = self._pi[s].dot(qs)
        delta = self._winDelta if sum_norm > sum_avg else self._loseDelta

        # Update probability of suboptimal actions
        for i, a in enumerate(self._n_actions):
            if a == argmax:
                self._pi[s][i] += delta
                self._pi[s][i] = min(1,self._pi[s][i])
            else :
                self._pi[s][i] -= delta/(len(self._n_actions)-1)
                self._pi[s][i] = max(0,self._pi[s][i])


    def update(self,s , a , r, nextObs):
        self.learn( s, r, a, nextObs)
        self.calculateAveragePolicyUpdate(s)
        self.calculatePolicyUpdate(s, a)

        self._epsilon = max(self._epsilon*0.999,self._min_epsilon)