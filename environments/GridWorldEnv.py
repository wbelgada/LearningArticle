import numpy as np

class GridWorldEnv:

    def __init__(self):
        self.agent1 = 6
        self.agent2 = 8

    def step(self, action1, action2):
        reward1 = 0
        reward2 = 0
        done = False
        new1 = self.actOne(action1, 1)
        new2 = self.actOne(action2, 2)

        if new1 != new2:
            if 0 <= new1 <= 8:
                self.agent1 = new1
                if self.agent1 == 1:
                    reward1 = 1
                    done = True
            if 0 <= new2 <= 8:
                self.agent2 = new2
                if self.agent2 == 1:
                    reward2 = 1
                    done = True

        return self.agent1, self.agent2, reward1, reward2, done

    def actOne(self, action, agent):

        if(agent == 1):
            currentAgent = self.agent1
        else:
            currentAgent = self.agent2
        if (currentAgent == 6 or currentAgent == 8) and action == 0: #north
            if np.random.uniform(0, 1) < 0.5:
                currentAgent -= 3
        elif action == 0: #north
            currentAgent-=3
        elif action == 1: #south
            currentAgent+=3
        elif action == 2: #east
            currentAgent+=1
        elif action == 3: #west
            currentAgent -=1

        return currentAgent
