import numpy as np
import math
import random

true_means = [0.9, 3.2, 1.9]
true_std = [0.1, 2.2, 0.9]

cuml_rewards = [0.0, 0.0, 0.0]
cuml_actions = [0.0, 0.0, 0.0]
cuml_values = [0.0, 0.0, 0.0]
total_runs = 0
c = 2

def bandit(a):
    reward_1 = np.random.normal(true_means[a], true_std[a])
    return reward_1

def reward_adder(cuml_reward, cuml_action_attempt, overall_value, tr, reward_1):
    tr += 1
    cuml_reward += reward_1
    cuml_action_attempt += 1
    overall_value = overall_value + (1 / cuml_action_attempt) * (reward_1 - overall_value)
    return cuml_reward, cuml_action_attempt, overall_value, tr

def UCB_algo(cv, ca, tr: float, constant):
    UCB_algo_values = []
    if tr == 0 or cuml_actions[0] == 0 or cuml_actions[1] == 0 or cuml_actions[2] == 0:
        UCB_algo_values.append(random.uniform(0, 1))
        UCB_algo_values.append(random.uniform(0, 3))
        UCB_algo_values.append(random.uniform(0, 1))
    else:
        for i in range(3):
            algo = cv[i] + constant * math.sqrt(math.log(tr) / ca[i])
            UCB_algo_values.append(algo)
    return UCB_algo_values

def maximizer(vals):
    return int(np.argmax(vals))

def execution(tr2):
    algo_values = UCB_algo(cuml_values, cuml_actions, total_runs, c)
    index = maximizer(algo_values)
    turn_reward = bandit(index)
    cr, ca, cv, tr = reward_adder(cuml_rewards[index], cuml_actions[index], cuml_values[index], total_runs, turn_reward)
    cuml_rewards[index] = cr
    cuml_actions[index] = ca
    cuml_values[index] = cv
    tr2 = tr
    print(cuml_values)
    return tr2

for _ in range(10000000):
    total_runs = execution(total_runs)
    