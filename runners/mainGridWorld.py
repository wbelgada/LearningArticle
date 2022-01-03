from agents.SimWolfPHC import WoLFPHCAgent
import numpy as np
import matplotlib.pyplot as plt

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
            agents[0].update(obs1)
            agents[1].setExperience(obs2, action2, rew2, None, new_obs2)
            agents[1].update(obs2)
        t+=1
        obs1 = new_obs1
        obs2 = new_obs2
        """print("state1 : ", obs1)
        print("state2 : ", obs2)"""
    return agents[0]._pi[(6,)][0], agents[0]._pi[(6,)][2], agents[1]._pi[(8,)][0], agents[1]._pi[(8,)][3]

def train(env, num_episodes, evaluate_every, num_evaluation_episodes):
    returns = np.zeros(shape=((num_episodes // evaluate_every), num_evaluation_episodes, 2))
    agents = [WoLFPHCAgent(4, 9, 0.1, 0.999, 0.0025, 0.01, 0.0), WoLFPHCAgent(4, 9, 0.1, 0.999, 0.0025, 0.01, 0.0)]
    probsNorth1 = []
    probsNorth2 = []
    probsWest2 = []
    probsEast1 = []
    for i in range(num_episodes):
        run_episode(env, agents, True)

        if(i+1) % evaluate_every == 0:
            print("Training for", i+1, "/", num_episodes)
            for episode in range(num_evaluation_episodes):
                probsNorth1.append(agents[0]._pi[(6,)][0])
                probsNorth2.append(agents[1]._pi[(8,)][0])
                probsEast1.append(agents[0]._pi[(6,)][2])
                probsWest2.append(agents[1]._pi[(8,)][3])


    return agents, probsNorth1, probsEast1, probsNorth2, probsWest2


if __name__ == "__main__":
    env = GridWorldEnv()
    agents, probsNorth1, probsEast1, probsNorth2, probsWest2 = train(env, 10000, 1, 1)

    print("-------Interesting probs for agent 1--------")
    print(probsNorth1)
    print(probsEast1)

    print("-------Interesting probs for agent 2--------")
    print(probsNorth2)
    print(probsWest2)

    fig, ax = plt.subplots()

    plt.plot(probsNorth1, probsEast1, label="Training average")

    plt.plot(probsNorth2, probsWest2, label="Training average")
    ax.set_xlim(0,1)
    plt.show()