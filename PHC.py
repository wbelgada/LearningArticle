import numpy as np



class PHCAgent:

    def __init__(self,n_actions = 2, n_states = 20,delta = 0.02,epsilon = 1,decay = 0.999,epsilon_min = 0.0001):
        self.n_actions = n_actions
        self.n_states = n_states
        self.Q = np.zeros(shape=(n_states, n_actions))
        self.H = (1 / n_actions) * np.ones(shape=(n_states, n_actions))
        self.alpha = 0.9
        self.delta = delta  # how much we update the policy  if we are winning
        self.gamma = 0.9
        self.epsilon = epsilon
        self.decay = decay
        self.epsilon_min = epsilon_min


    def chooseAction(self, s):

        if np.random.random() < self.epsilon:
            action = np.random.randint(0,self.n_actions)
        else :
            action = np.random.choice(self.n_actions, p = self.H[s])

        return action

    def update(self, s, r, a, nextObs):
        self.Q[s, a] = (1 - self.alpha)*self.Q[s, a] + self.alpha*(r + self.gamma * np.max(self.Q[nextObs,:]))


        for a in range(self.n_actions):
            if a == np.argmax(self.Q[s, :]) :
                self.H[s, a] += self.delta #argamax a => increase probabilty
                self.H[s, a] = min(1,self.H[s, a])
            else:
                self.H[s, a] -= self.delta/(self.n_actions-1)  # argamax a => increase probabilty
                self.H[s, a] = max(0, self.H[s, a])

        self.epsilon = max(self.epsilon * self.decay, self.epsilon_min)