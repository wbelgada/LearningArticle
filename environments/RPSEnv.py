import numpy as np

class RPSEnv:

    def __init__(self):
        self.rewards = np.array([
            [0,-1,1],
            [1,0,-1],
            [-1,1,0]
        ])

    def step(self, action1, action2):
        reward1 = self.rewards[action1][action2]
        reward2 = -reward1

        return reward1, reward2, True