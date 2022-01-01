from agents.SimWolfPHC import WoLFPHCAgent
import numpy as np

from environments.GridWorldEnv import GridWorldEnv


def run_episode(env, agents, train):

    done = False
    obs1, obs2 = env.reset()

    t = 0
    while not done:
        action1 = agents[0].chooseAction(obs1)
        action2 = agents[1].chooseAction(obs2)
        new_obs1, new_obs2, rew1, rew2, done = env.step(action1, action2)

        if train :
            agents[0].setExperience(obs1, action1, rew1, None, new_obs1)
            agents[0].update()
            agents[1].setExperience(obs2,action2,rew2,None, new_obs2)
            agents[1].update()
        t+=1
        obs1 = new_obs1
        obs2 = new_obs2
        """print("state1 : ", obs1)
        print("state2 : ", obs2)"""
    return agents[0]._pi[(6,)][0], agents[0]._pi[(6,)][2], agents[1]._pi[(8,)][0], agents[1]._pi[(8,)][3] #TODO return les proba c mieux

def train(env, num_episodes, evaluate_every, num_evaluation_episodes):
    returns = np.zeros(shape=((num_episodes // evaluate_every), num_evaluation_episodes, 2))
    agents = [WoLFPHCAgent(4, 9, 0.1, 0.999, 0.01, 0.1, 0.0), WoLFPHCAgent(4, 9, 0.1, 0.999, 0.01, 0.1, 0.0)]
    probsNorth1 = []
    probsNorth2 = []
    probsWest2 = []
    probsEast1 = []
    for i in range(num_episodes):
        run_episode(env, agents, True)

        if(i+1) % evaluate_every == 0:
            print("Training for", i+1, "/", num_episodes)
            for episode in range(num_evaluation_episodes):
                probNorth1, probEast1, probNorth2, probWest2 = run_episode(env, agents, False)
                probsNorth1.append(probNorth1)
                probsNorth2.append(probNorth2)
                probsEast1.append(probEast1)
                probsWest2.append(probWest2)


    return agents, probsNorth1, probsEast1, probsNorth2, probsWest2


if __name__ == "__main__":
    env = GridWorldEnv()
    agents, probsNorth1, probsEast1, probsNorth2, probsWest2 = train(env, 1000, 1, 1)

    print("-------Interesting probs for agent 1--------")
    print(probsNorth1)
    print(probsEast1)

    print("-------Interesting probs for agent 2--------")
    print(probsNorth2)
    print(probsWest2)

