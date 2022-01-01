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
            reward1 = self.rew(1, new1)
            if reward1 is None:
                print("old : " ,self.agent1, " new : ", new1)
                reward1 = 0
            reward2 = self.rew(2, new2)
            if reward2 is None:
                print("old : ", self.agent2, " new : ", new2)
                reward2 = 0
            if 0 <= new1 <= 8 and reward1 != -1:
                self.agent1 = new1
            if 0 <= new2 <= 8 and reward2 != -1:
                self.agent2 = new2
        if self.agent1 == 1 or self.agent2 == 1:
            done=True

        return self.agent1, self.agent2, reward1, reward2, done

    def actOne(self, action, agent):

        if(agent == 1):
            currentAgent = self.agent1
        else:
            currentAgent = self.agent2
        if (currentAgent == 6 or currentAgent == 8) and action == 0: #north
            if np.random.random() < 0.5:
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

    def reset(self):
        self.__init__()
        return self.agent1, self.agent2

    def rew(self,cur, pos):

        if cur == 1:
            return self.getRew(self.agent1, pos)
        else :
            return self.getRew(self.agent2, pos)

    def getRew(self, old, new):
        # border checks
        if old == 0 or old == 2: # top border
            if new == -3 or new == -1:
                return -1
        if old == 0 or old == 3 or old == 6: # left border
            if new == -1 or new == 2 or new == 5:
                return -1
        if old == 6 or old == 7 or old == 8: # bottom border
            if new == 9 or new == 10 or new == 11:
                return -1
        if old == 2 or old == 5 or old == 8:
            if new == 3 or old == 6 or old == 9:
                return -1


        if old == 6 or old == 8:
            if new == 3 or new == 5 or new == 7:
                return 0.3
        elif old == 3 or old == 5 or old == 7:
            if new == 6 or new == 8:
                return -0.3
            elif new == 4 or new == 0 or new == 2:
                return 0.6
        elif old == 4 or old == 0 or old == 2:
            if new == 1:
                return 1
            elif new == 3 or new == 5 or new == 7:
                return -0.8

        if old == 6 or old == 8:
            if new == 6 or new == 8:
                return 0.15

