###
# Group Members
# Group member names and student numbers
# Neo Nkosi:2437872
# Joshua Moorhead:2489197
# Naomi Muzamani:2456718
# PraiseGod Emenike:2428608
###
import numpy as np
from environments.gridworld import GridworldEnv
import time
import matplotlib.pyplot as plt

def generate_random_trajectory(env, max_steps=100):
    state = env.reset()
    trajectory = []
    done = False
    step = 0

    while not done and step < max_steps:
        action = env.action_space.sample()  # Choose a random action
        next_state, reward, done, _ = env.step(action)
        trajectory.append((state, action))
        state = next_state
        step += 1

    return trajectory

def print_trajectory_grid(trajectory, shape):
    action_symbols = ['U', 'R', 'D', 'L']  # UP, RIGHT, DOWN, LEFT
    grid = [['o' for _ in range(shape[1])] for _ in range(shape[0])]

    for state, action in trajectory:
        row, col = state // shape[1], state % shape[1]
        grid[row][col] = action_symbols[action]

    print("Trajectory (actions taken at each state):")
    for row in grid:
        print(' '.join(row))

    # Print legend
    print("\nContext:")
    print("U: Up, R: Right, D: Down, L: Left")
    print("o: Unvisited state")
def policy_evaluation(env, policy, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    """
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V



def policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=1.0):
    """
    Iteratively evaluates and improves a policy until an optimal policy is found.
    """
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    while True:
        V = policy_evaluation_fn(env, policy, discount_factor)
        policy_stable = True
        for s in range(env.observation_space.n):
            chosen_action = np.argmax(policy[s])
            action_values = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            best_action = np.argmax(action_values)
            if chosen_action != best_action:
                policy_stable = False
            policy[s] = np.eye(env.action_space.n)[best_action]
        if policy_stable:
            return policy, V

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        raise NotImplementedError

    raise NotImplementedError


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all actions in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        action_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for probability, next_state, reward, done in env.P[state][action]:
                action_values[action] += probability * (reward + discount_factor * V[next_state])
        return action_values

    Values = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for states in range(env.observation_space.n):
            v = Values[states]
            Values[states] = np.max(one_step_lookahead(states, Values))
            delta = max(delta, np.abs(v - Values[states]))
        if delta < theta:
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.observation_space.n, env.action_space.n])
    for states in range(env.observation_space.n):
        action_values = one_step_lookahead(states, Values)
        best_action = np.argmax(action_values)
        policy[states, best_action] = 1.0

    return policy, Values


def main():
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(shape=[5, 5], terminal_states=[
                       24], terminal_reward=0, step_reward=-1)
    state = env.reset()
    print("")
    env.render()
    print("")

    trajectory = generate_random_trajectory(env)

    #debugging for trajectory
    """print("action at each state for first t timesteps:")
    for state, action in trajectory:
        print(f"State: {state}, Action: {action}")"""

    #print direction of the action taken at each state, shaped as the grid
    print_trajectory_grid(trajectory, env.shape)



    print("-----------------------------------")

    # TODO: generate random policy
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    random_policy = np.ones([num_states, num_actions]) / num_actions

    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")

    v = policy_evaluation(env, random_policy)

    print("State values (Policy Evaluation):")
    print(v.reshape(env.shape))

    # Test: Make sure the evaluated policy is what we expected
    expected_v = np.array([-106.81, -104.81, -101.37, -97.62, -95.07,
                           -104.81, -102.25, -97.69, -92.40, -88.52,
                           -101.37, -97.69, -90.74, -81.78, -74.10,
                           -97.62, -92.40, -81.78, -65.89, -47.99,
                           -95.07, -88.52, -74.10, -47.99, 0.0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")

    policy, v = policy_iteration(env)

    # Print out best action for each state in grid shape
    print("Optimal Policy (Policy Iteration):")
    optimal_policy_grid = np.array([np.argmax(policy[s]) for s in range(env.observation_space.n)])
    print(optimal_policy_grid.reshape(env.shape))

    print("State values (Policy Iteration):")
    print(v.reshape(env.shape))

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")

    policy, v = value_iteration(env)

    # print out best action for each state in grid shape
    print("Optimal Policy (Value Iteration):")
    optimal_policy_grid = np.array([np.argmax(policy[s]) for s in range(env.observation_space.n)])
    print(optimal_policy_grid.reshape(env.shape))
    # print state value for each state, as grid shape
    print("State values (Value Iteration):")
    print(v.reshape(env.shape))

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    # Timing Analysis and Plotting
    discount_rates = np.logspace(-0.2, 0, num=30)
    policy_iteration_times = []
    value_iteration_times = []

    for discount_rate in discount_rates:
        policy_iteration_time = 0
        for _ in range(10):
            start_time = time.time()
            policy_iteration(env, discount_factor=discount_rate)
            policy_iteration_time += time.time() - start_time
        policy_iteration_times.append(policy_iteration_time / 10)

        value_iteration_time = 0
        for _ in range(10):
            start_time = time.time()
            value_iteration(env, discount_factor=discount_rate)
            value_iteration_time += time.time() - start_time
        value_iteration_times.append(value_iteration_time / 10)

    plt.figure(figsize=(10, 6))
    plt.plot(discount_rates, policy_iteration_times, label='Policy Iteration')
    plt.plot(discount_rates, value_iteration_times, label='Value Iteration')
    plt.xlabel('Discount Rate (γ)')
    plt.ylabel('Average Time (seconds)')
    plt.title('Average Running Time vs Discount Rate')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
