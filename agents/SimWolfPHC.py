import random
import argparse
from copy import deepcopy
#from datetime import datetime
#from tensorboard_logger import configure, log_value
import numpy as np

class WoLFPHCAgent():
    def __init__(self,n_actions, n_states, learningRate, discountFactor, winDelta=0.01,
                 loseDelta=0.1, epsilon=0.0):

        self._lr = learningRate
        self._gamma = discountFactor
        self._winDelta = winDelta
        self._loseDelta = loseDelta

        self._n_actions = np.arange(n_actions)
        self._n_states = n_states

        self._avg_pi = {}
        self._pi = {}
        self._Q = {}
        self._n = {}
        self._C = {}

        # Setting max and mins
        self._max_winDelta = 0.9
        self._min_winDelta = winDelta

        self._max_loseDelta = 0.9
        self._min_loseDelta = loseDelta

        self._epsilon = 1

        self._max_epsilon = 0.9
        self._min_epsilon = epsilon

        self._max_lr = 0.9
        self._min_lr = learningRate

    def setExperience(self, state, action, reward, status, nextState):
        self._s1 = state
        self._a = action
        self._r = reward
        self._d = status
        self._s2 = nextState

    def learn(self):
        Q = self.Q(self._s1, self._a)
        Q_max = max([self.Q(self._s2, a2) for a2 in self._n_actions])

        TD_target = self._r + self._gamma * Q_max
        TD_delta = TD_target - Q

        self._Q[self._tuple([self._s1, self._a])] = Q + self._lr * TD_delta
        return TD_delta * self._lr  # Return the delta in Q

    def avg_pi(self, s):
        key = self._tuple(s)
        if key not in self._avg_pi.keys():
            self._avg_pi[key] = [1 / len(self._n_actions)] * len(self._n_actions)
        return self._avg_pi[key]

    def pi(self, s):
        key = self._tuple(s)
        if key not in self._pi.keys():
            self._pi[key] = [1 / len(self._n_actions)] * len(self._n_actions)
        return self._pi[key]

    def Q(self, s, a):
        key = self._tuple([s, a])
        if key not in self._Q.keys():
            self._Q[key] = 0.0
            return 0.0
        return self._Q[key]

    def C(self, s):
        key = self._tuple(s)
        if key not in self._C.keys():
            self._C[key] = 0.0
            return 0.0
        return self._C[key]

    def chooseAction(self, s):
        '''
        Choose action based on e greedy policy
        '''
        if np.random.random() > self._epsilon:  # Choose greedy action
            # a = self.possibleActions[self.Q(self._s1).argmax()]
            # a = self.possibleActions[np.argmax(self.pi(self._s1))]
            a = np.random.choice(self._n_actions, p=self.pi(s))
        else:  # Choose randomly
            a = np.random.choice(self._n_actions)
        return a

    def calculateAveragePolicyUpdate(self):
        self._C[self._tuple(self._s1)] = self.C(self._s1) + 1
        C = self.C(self._s1)
        for i, a in enumerate(self._n_actions):
            k = self._tuple(self._s1)
            self._avg_pi[k][i] = self.avg_pi(k)[i] + 1 / C * (self.pi(k)[i] - self.avg_pi(k)[i])
        return self.avg_pi(k)  # The avg policy in current state

    def calculatePolicyUpdate(self):
        # Find the suboptimal actions
        Q_max = max([self.Q(self._s1, a) for a in self._n_actions])
        actions_sub = [a for a in self._n_actions if
                       self.Q(self._s1, a) < Q_max]  # TODO check this maybe it has an error
        assert len(actions_sub) != len(self._n_actions)

        # Decide which lr to use
        qs = [self.Q(self._s1, a) for a in self._n_actions]
        sum_avg = np.dot(self.avg_pi(self._s1), qs)
        sum_norm = np.dot(self.pi(self._s1), qs)
        delta = self._winDelta if sum_norm >= sum_avg else self._loseDelta

        # Update probability of suboptimal actions
        p_moved = 0.0
        for i, a in enumerate(self._n_actions):
            pi = self.pi(self._s1)
            if a in actions_sub:
                p_moved = p_moved + min([delta / len(actions_sub), pi[i]])
                self._pi[self._tuple(self._s1)][i] = pi[i] - min([delta / len(actions_sub), pi[i]])

            # Update prob of optimal actions
        for i, a in enumerate(self._n_actions):
            pi = self.pi(self._s1)
            if a not in actions_sub:
                self._pi[self._tuple(self._s1)][i] = pi[i] + p_moved / \
                                                     (len(self._n_actions) - len(actions_sub))

        return self.pi(self._s1)  # The policy for the current state

    def _tuple(self, args):
        if type(args) == list or type(args) == tuple:
            t = ()
            for x in args:
                t = t + self._tuple(x)
            return t
        else:
            return tuple([args])


    def update(self, s):
        self.learn()
        self.calculateAveragePolicyUpdate()
        self.calculatePolicyUpdate()

        self._epsilon = max(self._epsilon*0.999,self._min_epsilon)