import numpy as np
import random
import matplotlib.pyplot as plt

# --- Variables
true_values = np.zeros(10)
true_std = np.ones(10)
step_counter = 0 

cuml_reward = np.zeros(10)
cuml_actions = np.zeros(10)
cuml_values = np.zeros(10)

explore_rate = 0 

optimal_action_array = np.empty((2,0))
overall_steps = 0
overall_success = 0

# --- Functions
def random_walks(step_counter, true_values):
    step_counter += 1 
    if step_counter % 100 == 0:
        for i in range(10):
            true_values[i] = random.uniform(0, 5)
    else:
        true_values = true_values
    return step_counter, true_values

def bandit(index):
    return np.random.normal(true_values[index], true_std[index])

def reward_tracker(cuml_reward, cuml_actions, cuml_values, bandit_reward):
    cuml_reward += bandit_reward
    cuml_actions += 1
    cuml_values = cuml_values + (2) * (bandit_reward - cuml_values)
    return cuml_reward, cuml_actions, cuml_values

def greatest_selection(explore_rate, cuml_values):
    explore_rate += 1
    if explore_rate % 10 == 0:
        index = random.randint(0, 9)
    else: 
        index = int(np.argmax(cuml_values))
    return explore_rate, index

def optimal_action_plotter(index, true_values, overall_steps, overall_success, optimal_action_array):
    overall_steps += 1
    if index == np.argmax(true_values):
        overall_success += 1
    else:
        overall_success += 0
    optimal_rate = 100 * overall_success / overall_steps 
    new_col = np.array(([overall_steps], [optimal_rate]))
    optimal_action_array = np.column_stack((optimal_action_array, new_col))
    return overall_steps, overall_success, optimal_action_array    

def execution(step_counter, true_values, cuml_reward, cuml_actions, cuml_values, explore_rate, overall_steps, overall_success, optimal_action_array):
    step_counter, true_values = random_walks(step_counter, true_values)
    explore_rate, index = greatest_selection(explore_rate, cuml_values)
    bandit_reward = bandit(index)
    cr, ca, cv = reward_tracker(cuml_reward[index], cuml_actions[index], cuml_values[index], bandit_reward)
    overall_steps, overall_success, optimal_action_array = optimal_action_plotter(index, true_values, overall_steps, overall_success, optimal_action_array)
    return cr, ca, cv, optimal_action_array, explore_rate, overall_steps, overall_success, index, step_counter, true_values

for _ in range(10000):
    a, b, c, optimal_action_array, explore_rate, overall_steps, overall_success, index, step_counter, true_values = execution(step_counter, true_values, cuml_reward, cuml_actions, cuml_values, explore_rate, overall_steps, overall_success, optimal_action_array)
    cuml_reward[index] = a
    cuml_actions[index] = b
    cuml_values[index] = c
    print(optimal_action_array)

# --- Plot 
plt.plot(optimal_action_array[0], optimal_action_array[1])
plt.xlabel("Overall_Steps")
plt.ylabel("Overall_Success")
plt.title("Optimal Action Plot")
plt.grid(True)
plt.show()