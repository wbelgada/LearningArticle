import numpy as np

class WoLFPHCAgent():
    def __init__(self, n_actions=2, n_states=20, deltaWin=0.0001, deltaLose = 0.001, epsilon=1):
        self.n_actions = n_actions
        self.n_states = n_states
        self.Q = np.zeros(shape=(n_states, n_actions))
        self.H = (1 / n_actions) * np.ones(shape=(n_states, n_actions))
        self.H_AVG = np.zeros(shape=(n_states, n_actions))
        self.C = np.zeros(shape=(n_states,))
        self.alpha = 0.9
        self.deltaWin = deltaWin
        self.deltaLose = deltaLose # how much we update the policy  if we are winning
        self.gamma = 0.9
        self.epsilon = epsilon

    def chooseAction(self, s):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action = np.random.choice(self.n_actions, p=self.H[s])

        return action

    def update(self, s, r, a, nextObs):
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * (r + self.gamma * np.max(self.Q[nextObs, :]))

        self.C[s] += 1

        for a in range(self.n_actions):
            self.H_AVG[s,a] += (1/self.C[s]) * (self.H[s,a] - self.H_AVG[s,a])

        avgSum = self.H_AVG[s].dot(self.Q[s])
        nomSum = self.H[s].dot(self.Q[s])
        delta = self.deltaWin if nomSum > avgSum else self.deltaLose

        for a in range(self.n_actions):
            if a == np.argmax(self.Q[s, :]):
                self.H[s, a] += delta  # argamax a => increase probabilty
                self.H[s, a] = min(1, self.H[s, a])
            else:
                self.H[s, a] -= delta / (self.n_actions - 1)  # argamax a => increase probabilty
                self.H[s, a] = max(0, self.H[s, a])

        self.epsilon = max(self.epsilon * 0.999, 0.0001)