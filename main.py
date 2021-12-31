from SimWolfPHC import WoLFPHCAgent
from PHC import PHCAgent
import numpy as np
from matplotlib import pyplot as plt

def get_payoff(mat , a1, a2):
    return mat[a1][a2],mat[a1][a2]*(-1)

def run_episode(agents,mat, train):
    a1 = agents[0].chooseAction(0)
    a2 = agents[1].chooseAction(0)

    r1,r2 = get_payoff(mat, a1, a2)

    if train :
        agents[0].update(0, r1, a1, 0)
        agents[1].setExperience(0,a2,r2,None, 0)
        agents[1].update(0)

    return a1,a2

def train(num_episodes: int, evaluate_every: int, num_evaluation_episodes: int, epsilon: int,  decay: int, epsilon_min: int) :
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
    agents = [PHCAgent(2, 1, 0.05,epsilon,decay,epsilon_min),WoLFPHCAgent(2,1,0.1,0.999,0.01,0.1,0.0)]
    mat = np.array([[1,-1],
                    [-1,1]])
    current_eval = 0
    probs1 = []
    probs2 = []
    for i in range(num_episodes):
        run_episode(agents,mat,True)

        if ((i+1) % evaluate_every == 0) :
            print("Training for", i+1 , "/", num_episodes)
            for episode in range(num_evaluation_episodes):
                actions = run_episode(agents,mat,False)
                returns[current_eval][episode] = actions
            count1 = 0
            count2 = 0
            for elem in returns[current_eval]:
                if elem[0] == 0 :
                    count1 += 1
                if elem[1] == 0:
                    count2 += 1
            current_eval += 1
            print("Probability for this evaluation  is for agent 1 :", count1 /num_evaluation_episodes)
            print("Probability for this evaluation  is for agent 2 :", count2 / num_evaluation_episodes)

            probs1.append(count1/num_evaluation_episodes)
            probs2.append(count2 / num_evaluation_episodes)

    return agents,returns,probs1,probs2




if __name__ == "__main__" :

    num_episodes = 100000
    evaluate_every = 10000
    num_evaluation_episodes = 100
    epsilon = 1
    decay = 0.999
    epsilon_min = 0.00001

    agents,returns,probs1,probs2 = train(num_episodes,evaluate_every,num_evaluation_episodes,epsilon,decay,epsilon_min)

    print(probs1)
    print(probs2)


    fig, ax = plt.subplots()

    labels = [str((i + 1)) for i in range(0, num_episodes, evaluate_every)]
    x_train_pos = np.arange(0, num_episodes, evaluate_every)


    plt.plot(x_train_pos, probs1, label="Training average")

    plt.plot(x_train_pos, probs2, label="Training average")

    plt.show()