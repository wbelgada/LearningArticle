from agents.SimWolfPHC import WoLFPHCAgent
from agents.PHC import PHCAgent
import numpy as np
from matplotlib import pyplot as plt

from environments.RPSEnv import RPSEnv


def run_episode(env, agents, train):
    a1 = agents[0].chooseAction(0)
    a2 = agents[1].chooseAction(0)

    r1,r2, done = env.step(a1,a2)

    if train :
        agents[0].setExperience(0, a1, r1, None, 0)
        agents[0].update(0)
        agents[1].setExperience(0,a2,r2,None, 0)
        agents[1].update(0)

    return a1,a2

def train(env: RPSEnv, num_episodes: int, evaluate_every: int, num_evaluation_episodes: int, epsilon: int,  decay: int, epsilon_min: int) :
    """
        Training loop.
        :param env: The gym environment.
        :param gamma: The discount factor.
        :param num_episodes: Number of episodes to train.
        :param evaluate_every: Evaluation frequency.
        :param num_evaluation_episodes: Number of episodes for evaluation.
        :param alpha: Learning rate.
        :param epsilon_max: The maximum epsilon of epsilon-greedy.
        :param epsilon_min: The minimum epsilon of epsilon-greedy.
        :param epsilon_decay: The decay factor of epsilon-greedy.
        :return: Tuple containing the agent, the returns of all training episodes and averaged evaluation return of
                each evaluation.
        """

    returns = np.zeros(shape=((num_episodes//evaluate_every),num_evaluation_episodes,2))
    agents = [WoLFPHCAgent(3,1,0.1,0.999,0.01,0.1,0.0),WoLFPHCAgent(3,1,0.1,0.999,0.01,0.1,0.0)]
    current_eval = 0
    probs1 = []
    probs2 = []
    for i in range(num_episodes):
        run_episode(env,agents,True)
        probs1.append(agents[1]._pi[(0,)][1])
        probs2.append(agents[1]._pi[(0,)][0])
        if ((i+1) % (evaluate_every*10000) == 0) :
            print("Training for", i+1 , "/", num_episodes)

            print("Probability for this evaluation  is for agent 1 :", agents[0]._pi[(0,)][0])
            print("Probability for this evaluation  is for agent 2 :", agents[1]._pi[(0,)][0], agents[1]._pi[(0,)][1])



    return agents,returns,probs1,probs2




if __name__ == "__main__" :

    env = RPSEnv()

    num_episodes = 1000
    evaluate_every = 1
    num_evaluation_episodes = 100
    epsilon = 1
    decay = 0.999
    epsilon_min = 0.00001

    agents,returns,probs1,probs2 = train(env,num_episodes,evaluate_every,num_evaluation_episodes,epsilon,decay,epsilon_min)

    print(probs1)
    print(probs2)


    fig, ax = plt.subplots()

    labels = [str((i + 1)) for i in range(0, num_episodes, evaluate_every)]
    x_train_pos = np.arange(0, num_episodes, evaluate_every)


    plt.plot(probs1, probs2, label="Training average")

    #plt.plot(x_train_pos, probs2, label="Training average")

    plt.show()