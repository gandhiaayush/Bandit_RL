import numpy as np
import random

true_means = [0.9, 3.2, 1.9]
true_std = [0.1, 2.2, 0.9]

cuml_rewards = [0.0, 0.0, 0.0]
cuml_actions = [0, 0, 0]
cuml_values = [0.0, 0.0, 0.0]
explore_rate = 0

def bandit(a):
    reward_1 = np.random.normal(true_means[a], true_std[a])
    return reward_1

def reward(cuml_reward, cuml_action_attempt, overall_value, reward_1):
    cuml_reward += reward_1
    cuml_action_attempt += 1
    overall_value = overall_value + (1 / cuml_action_attempt) * (reward_1 - overall_value)
    return cuml_reward, cuml_action_attempt, overall_value

def greatest(vals, explore_rate):
    explore_rate += 1
    if explore_rate % 10 == 0 and explore_rate < 20000:
        return random.randint(0, 2), explore_rate
    else:
        return int(np.argmax(vals)), explore_rate

def execution(explore_rate):
    index, explore_rate = greatest(cuml_values, explore_rate)
    r = bandit(index)
    cr, ca, cv = reward(cuml_rewards[ab], cuml_actions[ab], cuml_values[ab], r)
    cuml_rewards[ab] = cr
    cuml_actions[ab] = ca
    cuml_values[ab] = cv
    print(cuml_values)
    print(explore_rate)
    return explore_rate

for _ in range(100000):
    explore_rate = execution(explore_rate)
