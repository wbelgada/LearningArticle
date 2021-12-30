from SimWolfPHC import WoLFPHCAgent
from PHC import PHCAgent
import numpy as np
from matplotlib import pyplot as plt

def get_payoff(mat , a1, a2):
    return mat[a1][a2],mat[a1][a2]*(-1)

def run_episode(agents,mat, train ):
    a1 = agents[0].chooseAction(0)
    a2 = agents[1].chooseAction(0)
    #a2 = np.random.randint(0,1)
    r1,r2 = get_payoff(mat, a1, a2)

    if train :
        agents[0].update(0, a1, r1, 0)
        agents[1].update(0, a2, r2, 0)

    return a1,a2

def train(num_episodes: int, evaluate_every: int, num_evaluation_episodes: int) :
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

    mat = np.array([[1, -1],
                    [-1, 1]])

    n_states = 1
    n_action = len(mat[0])

    delta_win = 0.001
    delta_lose = 0.1

    alpha = 0.1
    gamma = 0.999


    agents = [PHCAgent(n_action, n_states,delta_win,alpha,gamma,epsilon,decay,epsilon_min),WoLFPHCAgent(n_action,n_states,delta_win,delta_lose)]

    probs1 = []
    probs2 = []
    for i in range(num_episodes):
        run_episode(agents,mat,True)

        if ((i+1) % evaluate_every == 0) :
            print("Training for", i+1 , "/", num_episodes)
            print("Probability for this evaluation  is for agent 1 :", agents[0].H[0][0])
            print("Probability for this evaluation  is for agent 2 :", agents[1].H[0][0])

            probs1.append(agents[0].H[0][0])
            probs2.append(agents[1].H[0][0])

    return agents,returns,probs1,probs2




if __name__ == "__main__" :

    num_episodes = 1000000
    evaluate_every = 10000
    num_evaluation_episodes = 32
    epsilon = 1
    decay = 0.9999999
    epsilon_min = 0.000000001
    nb_runs = 1


    tot_probs1 = np.zeros(shape=(num_episodes//evaluate_every))
    tot_probs2 = np.zeros(shape=(num_episodes//evaluate_every))
    for i in range(nb_runs) :
        agents, returns, probs1, probs2 = train(num_episodes,evaluate_every,num_evaluation_episodes)
        tot_probs1 += probs1
        tot_probs2 += probs2

    tot_probs1 /= nb_runs
    tot_probs2 /= nb_runs

    fig, ax = plt.subplots()

    labels = [str((i + 1)) for i in range(0, num_episodes, evaluate_every)]
    x_train_pos = np.arange(0, num_episodes, evaluate_every)


    plt.plot(x_train_pos, probs1, label="PHC")

    #plt.plot(x_train_pos, probs2, label="WoLF-PHC")
    plt.legend()
    plt.show()