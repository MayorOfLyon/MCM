import numpy as np

def validate_transition_matrix(transition_matrix):
    # Validate that the probabilities sum up to 1.
    if not np.allclose(np.sum(transition_matrix, axis=1), 1):
        raise ValueError("Invalid transition matrix. Probabilities should sum up to 1.")

def markov_chain_simulation(states, transition_matrix, initial_state, days):
    # Validate input parameters
    if len(states) != len(transition_matrix) or len(states) != len(transition_matrix[0]):
        raise ValueError("Inconsistent dimensions in states or transition_matrix.")

    validate_transition_matrix(transition_matrix)

    # Initialize activity list with the initial state
    activity_today = initial_state
    activity_list = [activity_today]

    # To calculate the probability of the activity list
    prob = 1

    for _ in range(days):
        state_index = states.index(activity_today)
        change = np.random.choice(states, replace=True, p=transition_matrix[state_index])

        # Update probability and activity list
        prob *= transition_matrix[state_index][states.index(change)]
        activity_today = change
        activity_list.append(activity_today)

    return activity_list, prob

def monte_carlo_simulation(states, transition_matrix, target_start_state, target_end_state, iterations):
    cnt_start_from_target = 0
    cnt_start_from_target_and_end_at_target = 0

    for _ in range(iterations):
        init_state = np.random.choice(states, replace=True, p=[1/len(states)] * len(states))
        activity_list, _ = markov_chain_simulation(states, transition_matrix, init_state, 2)

        if activity_list[0] == target_start_state:
            cnt_start_from_target += 1
            if activity_list[2] == target_end_state:
                cnt_start_from_target_and_end_at_target += 1

    # Calculate the probability
    if cnt_start_from_target > 0:
        probability = (cnt_start_from_target_and_end_at_target / cnt_start_from_target) * 100
        print(f"The probability of starting at state '{target_start_state}' and ending at state '{target_end_state}' = {probability:.2f}%")
    else:
        print(f"No instances found starting from state '{target_start_state}' in the simulations.")

# The statespace
states = ["Sleep","Run","Icecream"]
# Probabilities matrix (transition matrix)
transitionMatrix = [[0.2, 0.6, 0.2], [0.1, 0.6, 0.3], [0.2, 0.7, 0.1]]

# User-specified parameters
custom_target_start_state = "Run"
custom_target_end_state = "Run"
custom_iterations = 10000

# Perform Monte Carlo simulation
monte_carlo_simulation(states, transitionMatrix, custom_target_start_state, custom_target_end_state, custom_iterations)
