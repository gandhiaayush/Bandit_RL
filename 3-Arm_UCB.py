import numpy as np
import random
import matplotlib.pyplot as plt

# --- environment (non-stationary every 10 steps) ---
true_values = np.zeros(10)
true_std    = np.ones(10)

# --- agent state ---
cuml_reward  = np.zeros(10, dtype=float)
cuml_actions = np.zeros(10, dtype=int)      # counts as integers
cuml_values  = np.zeros(10, dtype=float)

# --- trackers ---
step_counter = 0
explore_rate = 0
overall_steps = 0
overall_success = 0
optimal_action_array = np.empty((2, 0), dtype=float)  # [ [step...], [rate...] ]

def random_walks(step_counter, true_values):
    """Every 10 steps, reset the true means to U[0,5]."""
    step_counter += 1
    if step_counter % 10 == 0:
        for i in range(10):
            true_values[i] = random.uniform(0, 5)
    return step_counter, true_values

def bandit(index, true_values, true_std):
    return np.random.normal(true_values[index], true_std[index])

def reward_tracker(c_r, c_a, c_v, r):
    """Sample-average update for a single arm (scalar in, scalar out)."""
    c_r += r
    c_a += 1
    c_v = c_v + (r - c_v) / c_a
    return c_r, c_a, c_v

def greatest_selection(explore_rate, q_values):
    """Periodic exploration every 10th decision; otherwise greedy."""
    explore_rate += 1
    if explore_rate % 10 == 0:
        index = random.randint(0, 9)
    else:
        # break ties randomly to avoid bias toward low index
        max_val = np.max(q_values)
        candidates = np.flatnonzero(q_values == max_val)
        index = int(np.random.choice(candidates))
    return explore_rate, index

def optimal_action_plotter(index, true_values, overall_steps, overall_success, optimal_action_array):
    overall_steps += 1
    if index == int(np.argmax(true_values)):
        overall_success += 1
    optimal_rate = 100.0 * overall_success / overall_steps
    new_col = np.array([[overall_steps], [optimal_rate]])
    optimal_action_array = np.column_stack((optimal_action_array, new_col))
    return overall_steps, overall_success, optimal_action_array

def execution(step_counter, true_values, cuml_reward, cuml_actions, cuml_values,
              explore_rate, overall_steps, overall_success, optimal_action_array):
    # update environment
    step_counter, true_values = random_walks(step_counter, true_values)

    # choose action
    explore_rate, index = greatest_selection(explore_rate, cuml_values)

    # pull arm and update that arm's stats
    r = bandit(index, true_values, true_std)
    cr, ca, cv = reward_tracker(cuml_reward[index], cuml_actions[index], cuml_values[index], r)

    # record optimal-action metric
    overall_steps, overall_success, optimal_action_array = optimal_action_plotter(
        index, true_values, overall_steps, overall_success, optimal_action_array
    )

    # write back the updated scalars for the chosen arm
    cuml_reward[index]  = cr
    cuml_actions[index] = ca
    cuml_values[index]  = cv

    return (step_counter, true_values, cuml_reward, cuml_actions, cuml_values,
            explore_rate, overall_steps, overall_success, optimal_action_array)

# --- run ---
for _ in range(100):
    (step_counter, true_values, cuml_reward, cuml_actions, cuml_values,
     explore_rate, overall_steps, overall_success, optimal_action_array) = execution(
        step_counter, true_values, cuml_reward, cuml_actions, cuml_values,
        explore_rate, overall_steps, overall_success, optimal_action_array
    )

# --- plot ---
plt.plot(optimal_action_array[0], optimal_action_array[1])
plt.xlabel("Steps")
plt.ylabel("% Optimal action")
plt.title("Optimal Action Plot")
plt.grid(True)
plt.tight_layout()
plt.show()
