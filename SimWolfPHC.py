import numpy as np



class WoLFPHCAgent :

    def __init__(self, s_ideal = None, n_actions = 2, n_states = 20, delta_win = 0.05, delta_lose = 0.02,epsilon = 1,decay = 0.999,epsilon_min = 0.0001):
        self.s_ideal = s_ideal
        self.n_actions = n_actions
        self.n_states = n_states
        self.Q = np.zeros(shape=(n_states,n_actions))
        self.H = (1/n_actions) * np.ones(shape=(n_states,n_actions))
        self.H_AVG = np.zeros(shape=(n_states,n_actions)) #(1/n_actions)*np.ones(shape=(n_states,n_actions))
        self.C = np.zeros(shape=(n_states))
        self.alpha = 0.9
        self.delta_win = delta_win # how much we update the policy  if we are winning
        self.delta_loose = delta_lose #how much we update the policy  if we are loosing
        self.gamma = 0.9
        self.epsilon = epsilon
        self.decay = decay
        self.epsilon_min = epsilon_min


    """
    def get_reward(self, s, s_ideal):
        #HARD VERSION
        if np.round(s) == np.round(s_ideal) :
            return 1
        else: 
            return 0
    """

    def chooseAction(self,s):
        u = np.random.uniform()
        if u <= self.epsilon :
            action = np.random.choice(range(self.n_actions))
        else:
            action = np.argmax(self.Q[s])
        return action

    def update_Q(self,s, a, r, s_next):
        self.Q[s,a] = (1 - self.alpha) * self.Q[s,a] + self.alpha * (r + self.gamma * np.max(self.Q[s_next, :]))

    def update_H_AVG(self , s):
        self.C[s] += 1
        self.H_AVG[s, :] += (1/self.C[s]) * (self.H[s, :] - self.H_AVG[s , :])

    def update_H(self, s, a):
        #Calculate deltas
        if self.H[s, :].dot(self.Q[s, :]) > self.H_AVG[s, :].dot(self.Q[s, :]):
            delta = self.delta_win
        else :
            delta = self.delta_loose


        if a == np.argmax(self.Q[s, :]) :
            self.H[s, a] += delta #argamax a => increase probabilty
            self.H[s, a] = min(1,self.H[s, a])
        else:
            self.H[s, a] -= delta/(self.n_actions-1)  # argamax a => increase probabilty
            self.H[s, a] = max(0, self.H[s, a])


    def update(self, s, r, a, s_next):
        self.update_H_AVG(s)
        self.update_Q(s, a, r, s_next)
        self.update_H(s, a)

        self.epsilon = max(self.epsilon * self.decay, self.epsilon_min)




