"""
RL Lab Assignment 3
GridWorld – Policy Evaluation and Value Iteration

Tasks implemented:
    Task 1 - Formulating the GridWorld Environment
    Task 2 - Policy Evaluation
    Task 3 - Value Iteration and Optimal Policy Generation
    Task 4 - Optimal Path Analysis

Environment conventions are based on the Neuromatch GridWorld tutorial:
    - Four actions: Left, Right, Down, Up
    - Walls are impassable
    - Attempting to move into a wall keeps the agent in the same state
    - Reaching the goal gives reward 1
    - Other transitions give reward 0

Because the instructor did not provide a specific layout/policy,
a compact 5-state GridWorld has been defined explicitly here.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

GAMMA = 0.9
THETA = 1e-6

# Fixed seed makes the Random Policy experiment reproducible.
RANDOM_SEED = 42

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# TASK 1 - GRIDWORLD DEFINITION
# ============================================================

# Grid representation:
#
#       C1      C2
#    ┌───────┬───────┐
# R1 │  S1   │  S2   │
#    ├───────┼───────┤
# R2 │  S3   │  S4   │
#    ├───────┼───────┤
# R3 │  S5   │  WALL │
#    └───────┴───────┘
#
# S1 = Start
# S5 = Goal / Terminal
#
# '*' = wall
# ' ' = valid state
# 'g' = goal

WORLD_SPEC = np.array([
    ['*', '*', '*', '*'],
    ['*', ' ', ' ', '*'],
    ['*', ' ', ' ', '*'],
    ['*', 'g', '*', '*'],
    ['*', '*', '*', '*']
])


# State names and their coordinates.
STATES = {
    "S1": (1, 1),
    "S2": (1, 2),
    "S3": (2, 1),
    "S4": (2, 2),
    "S5": (3, 1),
}

COORD_TO_STATE = {
    coordinate: state
    for state, coordinate in STATES.items()
}

START_STATE = "S1"
GOAL_STATE = "S5"


# ============================================================
# ACTION DEFINITIONS
# ============================================================

# Actions follow the Neuromatch tutorial notation:
#
# < = Left
# > = Right
# v = Down
# ^ = Up

ACTIONS = ["<", ">", "v", "^"]

ACTION_NAMES = {
    "<": "Left",
    ">": "Right",
    "v": "Down",
    "^": "Up",
}

ACTION_EFFECTS = {
    "<": (0, -1),
    ">": (0, 1),
    "v": (1, 0),
    "^": (-1, 0),
}


# ============================================================
# GRIDWORLD FUNCTIONS
# ============================================================

def is_valid_cell(position):
    """
    Return True if a position is a valid non-wall cell.
    """

    row, col = position

    rows, cols = WORLD_SPEC.shape

    if row < 0 or row >= rows:
        return False

    if col < 0 or col >= cols:
        return False

    return WORLD_SPEC[row, col] != "*"


def get_neighbours(state):
    """
    Return the resulting state for each possible action.

    If an action would hit a wall or boundary, the agent
    remains in the current state.

    Returns
    -------
    dict
        Example:
        {
            '<': 'S1',
            '>': 'S2',
            'v': 'S3',
            '^': 'S1'
        }
    """

    if state == GOAL_STATE:
        return {action: GOAL_STATE for action in ACTIONS}

    current_position = STATES[state]

    neighbours = {}

    for action in ACTIONS:

        delta_row, delta_col = ACTION_EFFECTS[action]

        new_position = (
            current_position[0] + delta_row,
            current_position[1] + delta_col
        )

        if is_valid_cell(new_position):

            neighbours[action] = COORD_TO_STATE[new_position]

        else:

            # Hit wall/boundary:
            # remain in current state.
            neighbours[action] = state

    return neighbours


def step(state, action):
    """
    Execute one action.

    Returns
    -------
    next_state : str
    reward : float
    """

    if state == GOAL_STATE:
        return GOAL_STATE, 0.0

    neighbours = get_neighbours(state)

    next_state = neighbours[action]

    # Reward of 1 only when the goal is reached.
    if next_state == GOAL_STATE:
        reward = 1.0
    else:
        reward = 0.0

    return next_state, reward


# ============================================================
# DISPLAY ENVIRONMENT
# ============================================================

def display_grid(policy=None, values=None):
    """
    Display the GridWorld in the terminal.

    If policy is provided, actions are shown.
    If values are provided, state values are shown.
    """

    print()

    for row in range(WORLD_SPEC.shape[0]):

        line = ""

        for col in range(WORLD_SPEC.shape[1]):

            cell = WORLD_SPEC[row, col]

            if cell == "*":

                line += "##########"

            elif cell == "g":

                line += "   S5 ⭐  "

            else:

                state = COORD_TO_STATE[(row, col)]

                if policy is not None and state in policy:

                    action = policy[state]

                    if isinstance(action, list):

                        symbols = "/".join(action)
                        text = f"{state} {symbols}"

                    else:

                        text = f"{state} {action}"

                elif values is not None and state in values:

                    text = f"{state} {values[state]:.3f}"

                else:

                    text = f"  {state}  "

                line += f"{text:^10}"

        print(line)


# ============================================================
# TASK 2 - POLICY EVALUATION
# ============================================================

# Fixed policy supplied for this experiment.
#
# S1 -> Right
# S2 -> Down
# S3 -> Down
# S4 -> Left
# S5 -> Terminal
#
# Therefore:
#
# S1 -> S2 -> S4 -> S3 -> S5

EVALUATED_POLICY = {
    "S1": ">",
    "S2": "v",
    "S3": "v",
    "S4": "<",
}


def policy_evaluation(theta=THETA):
    """
    Evaluate the fixed policy using the Bellman
    expectation equation.

    V(s) = R + gamma * V(s')

    Returns
    -------
    values : dict
        Final state values.

    history : list
        Iteration information.
    """

    values = {
        state: 0.0
        for state in STATES
    }

    history = []

    iteration = 0

    while True:

        iteration += 1

        new_values = values.copy()

        for state in STATES:

            # Terminal state.
            if state == GOAL_STATE:

                new_values[state] = 0.0
                continue

            action = EVALUATED_POLICY[state]

            next_state, reward = step(state, action)

            new_values[state] = (
                reward
                + GAMMA * values[next_state]
            )

        # Maximum change.
        delta = max(
            abs(
                new_values[state]
                - values[state]
            )
            for state in STATES
        )

        history.append({
            "iteration": iteration,
            "delta": delta,
            "values": new_values.copy()
        })

        values = new_values

        if delta < theta:
            break

    return values, history


def print_policy_evaluation_results(values, history):
    """
    Print Task 2 results.
    """

    print("\n")
    print("=" * 70)
    print("TASK 2 - POLICY EVALUATION")
    print("=" * 70)

    print("\nFixed Policy:")

    for state in EVALUATED_POLICY:

        action = EVALUATED_POLICY[state]

        print(
            f"  {state} -> "
            f"{ACTION_NAMES[action]} ({action})"
        )

    print("\nPolicy Evaluation Iterations")
    print("-" * 70)

    print(
        f"{'Iteration':<12}"
        f"{'Maximum Change (Δ)':<25}"
        f"{'Convergence Status':<20}"
    )

    print("-" * 70)

    # Assignment asks us to record iterations 1-5.
    for record in history[:5]:

        iteration = record["iteration"]
        delta = record["delta"]

        status = (
            "Converged"
            if delta < THETA
            else "Not Converged"
        )

        print(
            f"{iteration:<12}"
            f"{delta:<25.6f}"
            f"{status:<20}"
        )

    print("\nFinal State Values")
    print("-" * 40)

    for state in STATES:

        print(
            f"{state}: "
            f"{values[state]:.6f}"
        )


def plot_policy_evaluation(history):
    """
    Plot maximum change versus iteration.
    """

    iterations = [
        record["iteration"]
        for record in history
    ]

    deltas = [
        record["delta"]
        for record in history
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        iterations,
        deltas,
        marker="o"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Maximum Change (Δ)")
    plt.title("Policy Evaluation Convergence")

    plt.grid(True)

    plt.tight_layout()

    filename = os.path.join(
        OUTPUT_DIR,
        "policy_evaluation_convergence.png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()

    print(
        f"\nSaved plot: {filename}"
    )


# ============================================================
# TASK 3 - VALUE ITERATION
# ============================================================

def value_iteration(theta=THETA):
    """
    Perform Value Iteration using the Bellman
    Optimality Equation.

    V(s) = max_a [R + gamma * V(s')]

    Returns
    -------
    values : dict
        Optimal state values.

    optimal_policy : dict
        Optimal actions.

    history : list
        Iteration information.
    """

    values = {
        state: 0.0
        for state in STATES
    }

    history = []

    iteration = 0

    while True:

        iteration += 1

        new_values = values.copy()

        for state in STATES:

            # Terminal state.
            if state == GOAL_STATE:

                new_values[state] = 0.0
                continue

            action_values = {}

            for action in ACTIONS:

                next_state, reward = step(
                    state,
                    action
                )

                action_values[action] = (
                    reward
                    + GAMMA * values[next_state]
                )

            # Bellman Optimality Equation.
            new_values[state] = max(
                action_values.values()
            )

        delta = max(
            abs(
                new_values[state]
                - values[state]
            )
            for state in STATES
        )

        history.append({
            "iteration": iteration,
            "delta": delta,
            "values": new_values.copy()
        })

        values = new_values

        if delta < theta:
            break

    # --------------------------------------------------------
    # Extract optimal policy from converged values.
    # --------------------------------------------------------

    optimal_policy = {}

    for state in STATES:

        if state == GOAL_STATE:

            optimal_policy[state] = []

            continue

        action_values = {}

        for action in ACTIONS:

            next_state, reward = step(
                state,
                action
            )

            action_values[action] = (
                reward
                + GAMMA * values[next_state]
            )

        best_value = max(
            action_values.values()
        )

        best_actions = [
            action
            for action, value in action_values.items()
            if np.isclose(
                value,
                best_value
            )
        ]

        optimal_policy[state] = best_actions

    return values, optimal_policy, history


def print_value_iteration_results(
    values,
    optimal_policy,
    history
):
    """
    Print Task 3 results.
    """

    print("\n")
    print("=" * 70)
    print("TASK 3 - VALUE ITERATION")
    print("=" * 70)

    print(
        f"\n{'State':<10}"
        f"{'Optimal Action':<25}"
        f"{'State Value':<15}"
    )

    print("-" * 55)

    for state in STATES:

        if state == GOAL_STATE:

            action_text = "Terminal"

        else:

            action_text = ", ".join(
                f"{ACTION_NAMES[action]} ({action})"
                for action in optimal_policy[state]
            )

        print(
            f"{state:<10}"
            f"{action_text:<25}"
            f"{values[state]:<15.6f}"
        )

    print("\nValue Iteration Convergence")
    print("-" * 50)

    print(
        f"Iterations required: "
        f"{len(history)}"
    )

    print(
        f"Final Δ: "
        f"{history[-1]['delta']:.10f}"
    )


def plot_value_iteration(history):
    """
    Plot Value Iteration convergence.
    """

    iterations = [
        record["iteration"]
        for record in history
    ]

    deltas = [
        record["delta"]
        for record in history
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        iterations,
        deltas,
        marker="o"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Maximum Change (Δ)")
    plt.title("Value Iteration Convergence")

    plt.grid(True)

    plt.tight_layout()

    filename = os.path.join(
        OUTPUT_DIR,
        "value_iteration_convergence.png"
    )

    plt.savefig(
        filename,
        dpi=300
    )


    plt.close()

    print(
        f"\nSaved plot: {filename}"
    )


# ============================================================
# POLICY EXECUTION
# ============================================================

def choose_random_action(rng):
    """
    Select a random action with equal probability.
    """

    return rng.choice(ACTIONS)


def choose_policy_action(
    state,
    policy,
    rng=None
):
    """
    Select an action according to the requested policy.
    """

    if policy == "random":

        return choose_random_action(rng)

    elif policy == "evaluated":

        return EVALUATED_POLICY[state]

    elif policy == "optimal":

        # If multiple optimal actions exist,
        # choose the first one for reproducibility.
        actions = OPTIMAL_POLICY[state]

        return actions[0]

    else:

        raise ValueError(
            f"Unknown policy: {policy}"
        )


def run_policy(
    policy,
    seed=RANDOM_SEED,
    max_steps=100
):
    """
    Run an agent from S1 until it reaches the goal
    or max_steps is exceeded.

    Returns
    -------
    result : dict
    """

    rng = np.random.default_rng(seed)

    state = START_STATE

    path = [state]

    rewards = []

    total_reward = 0.0

    for step_number in range(1, max_steps + 1):

        if state == GOAL_STATE:

            break

        action = choose_policy_action(
            state,
            policy,
            rng
        )

        next_state, reward = step(
            state,
            action
        )

        rewards.append(reward)

        total_reward += reward

        state = next_state

        path.append(state)

        if state == GOAL_STATE:

            break

    goal_reached = (
        state == GOAL_STATE
    )

    return {
        "policy": policy,
        "path": path,
        "actions": len(rewards),
        "rewards": rewards,
        "total_reward": total_reward,
        "goal_reached": goal_reached
    }


# ============================================================
# TASK 4 - POLICY COMPARISON
# ============================================================

def print_policy_execution(result):
    """
    Print one policy's trajectory.
    """

    policy_name = {
        "random": "Random Policy",
        "evaluated": "Evaluated Policy",
        "optimal": "Optimal Policy"
    }[result["policy"]]

    print(f"\n{policy_name}")
    print("-" * 50)

    print(
        "Path: "
        + " -> ".join(result["path"])
    )

    print(
        f"Path Length: "
        f"{result['actions']}"
    )

    print(
        f"Rewards: "
        f"{result['rewards']}"
    )

    print(
        f"Total Reward: "
        f"{result['total_reward']:.3f}"
    )

    print(
        f"Goal Reached: "
        f"{'Yes' if result['goal_reached'] else 'No'}"
    )


def compare_policies():
    """
    Execute Random, Evaluated, and Optimal policies.
    """

    print("\n")
    print("=" * 70)
    print("TASK 4 - OPTIMAL PATH ANALYSIS")
    print("=" * 70)

    random_result = run_policy(
        "random",
        seed=RANDOM_SEED
    )

    evaluated_result = run_policy(
        "evaluated"
    )

    optimal_result = run_policy(
        "optimal"
    )

    results = [
        random_result,
        evaluated_result,
        optimal_result
    ]

    for result in results:

        print_policy_execution(result)

    print("\nComparison Table")
    print("-" * 75)

    print(
        f"{'Policy Type':<22}"
        f"{'Path Length':<15}"
        f"{'Total Reward':<15}"
        f"{'Goal Reached':<15}"
    )

    print("-" * 75)

    for result in results:

        policy_name = {
            "random": "Random Policy",
            "evaluated": "Evaluated Policy",
            "optimal": "Optimal Policy"
        }[result["policy"]]

        print(
            f"{policy_name:<22}"
            f"{result['actions']:<15}"
            f"{result['total_reward']:<15.3f}"
            f"{'Yes' if result['goal_reached'] else 'No':<15}"
        )

    return results


# ============================================================
# OPTIONAL: DISPLAY POLICY ON GRID
# ============================================================

def display_policy_grid(policy, title):
    """
    Print a simple representation of a policy.
    """

    print("\n" + title)
    print("-" * len(title))

    for row in range(WORLD_SPEC.shape[0]):

        line = ""

        for col in range(WORLD_SPEC.shape[1]):

            cell = WORLD_SPEC[row, col]

            if cell == "*":

                line += "  ████  "

            elif cell == "g":

                line += "  GOAL  "

            else:

                state = COORD_TO_STATE[
                    (row, col)
                ]

                actions = policy.get(
                    state,
                    []
                )

                if isinstance(actions, list):

                    if len(actions) == 0:

                        symbol = "·"

                    else:

                        symbol = "/".join(
                            actions
                        )

                else:

                    symbol = actions

                line += f" {state}:{symbol:^3} "

        print(line)


# ============================================================
# SAVE NUMERICAL RESULTS TO TEXT FILE
# ============================================================

def save_results(
    evaluated_values,
    evaluation_history,
    optimal_values,
    optimal_policy,
    policy_results
):
    """
    Save important numerical results to a text file.
    """

    filename = os.path.join(
        OUTPUT_DIR,
        "lab_results.txt"
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "RL LAB ASSIGNMENT 3\n"
        )

        file.write(
            "5-State GridWorld\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        # ----------------------------------------------------
        # Environment
        # ----------------------------------------------------

        file.write(
            "ENVIRONMENT\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            "States: S1, S2, S3, S4, S5\n"
        )

        file.write(
            "Start State: S1\n"
        )

        file.write(
            "Goal/Terminal State: S5\n"
        )

        file.write(
            "Actions: Left, Right, Down, Up\n"
        )

        file.write(
            "Goal Reward: 1\n"
        )

        file.write(
            "Other Rewards: 0\n"
        )

        file.write(
            f"Discount Factor: {GAMMA}\n\n"
        )

        # ----------------------------------------------------
        # Policy Evaluation
        # ----------------------------------------------------

        file.write(
            "POLICY EVALUATION\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        for record in evaluation_history:

            file.write(
                f"Iteration {record['iteration']}: "
                f"Delta = {record['delta']:.10f}\n"
            )

        file.write("\nFinal Values:\n")

        for state in STATES:

            file.write(
                f"{state}: "
                f"{evaluated_values[state]:.6f}\n"
            )

        # ----------------------------------------------------
        # Value Iteration
        # ----------------------------------------------------

        file.write(
            "\nVALUE ITERATION\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        for state in STATES:

            if state == GOAL_STATE:

                actions = "Terminal"

            else:

                actions = ", ".join(
                    optimal_policy[state]
                )

            file.write(
                f"{state}: "
                f"Action = {actions}, "
                f"Value = "
                f"{optimal_values[state]:.6f}\n"
            )

        # ----------------------------------------------------
        # Policy comparison
        # ----------------------------------------------------

        file.write(
            "\nPOLICY COMPARISON\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        for result in policy_results:

            file.write(
                f"\n{result['policy']}\n"
            )

            file.write(
                "Path: "
                + " -> ".join(result["path"])
                + "\n"
            )

            file.write(
                f"Path Length: "
                f"{result['actions']}\n"
            )

            file.write(
                f"Total Reward: "
                f"{result['total_reward']:.3f}\n"
            )

            file.write(
                f"Goal Reached: "
                f"{result['goal_reached']}\n"
            )

    print(
        f"\nResults saved to: {filename}"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("REINFORCEMENT LEARNING LAB ASSIGNMENT 3")
    print("GRIDWORLD - POLICY EVALUATION AND VALUE ITERATION")
    print("=" * 70)

    print("\nEnvironment Parameters")
    print("-" * 40)

    print(f"States: {list(STATES.keys())}")
    print(f"Start State: {START_STATE}")
    print(f"Goal State: {GOAL_STATE}")
    print(f"Discount Factor γ: {GAMMA}")
    print(f"Convergence Threshold θ: {THETA}")

    # --------------------------------------------------------
    # TASK 1
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TASK 1 - GRIDWORLD ENVIRONMENT")
    print("=" * 70)

    display_grid()

    print("\nState Mapping:")

    for state, position in STATES.items():

        print(
            f"  {state} -> "
            f"Grid position {position}"
        )

    print("\nAvailable Actions:")

    for action in ACTIONS:

        print(
            f"  {action} = "
            f"{ACTION_NAMES[action]}"
        )

    print("\nTransition Examples:")

    examples = [
        ("S1", ">", "S2"),
        ("S1", "v", "S3"),
        ("S3", "v", "S5"),
        ("S1", "^", "S1"),
        ("S1", "<", "S1")
    ]

    for state, action, expected in examples:

        next_state, reward = step(
            state,
            action
        )

        print(
            f"  {state} --{action}--> "
            f"{next_state}, "
            f"Reward = {reward}"
        )

    # --------------------------------------------------------
    # TASK 2
    # --------------------------------------------------------

    evaluated_values, evaluation_history = (
        policy_evaluation()
    )

    print_policy_evaluation_results(
        evaluated_values,
        evaluation_history
    )

    display_policy_grid(
        EVALUATED_POLICY,
        "Evaluated Policy"
    )

    plot_policy_evaluation(
        evaluation_history
    )

    # --------------------------------------------------------
    # TASK 3
    # --------------------------------------------------------

    global OPTIMAL_POLICY

    (
        optimal_values,
        OPTIMAL_POLICY,
        value_history
    ) = value_iteration()

    print_value_iteration_results(
        optimal_values,
        OPTIMAL_POLICY,
        value_history
    )

    display_policy_grid(
        OPTIMAL_POLICY,
        "Optimal Policy"
    )

    plot_value_iteration(
        value_history
    )

    # --------------------------------------------------------
    # TASK 4
    # --------------------------------------------------------

    policy_results = compare_policies()

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    save_results(
        evaluated_values,
        evaluation_history,
        optimal_values,
        OPTIMAL_POLICY,
        policy_results
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("LAB EXECUTION COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")

    print(
        "  outputs/policy_evaluation_convergence.png"
    )

    print(
        "  outputs/value_iteration_convergence.png"
    )

    print(
        "  outputs/lab_results.txt"
    )

    print("\nUse the terminal output and lab_results.txt")
    print("to fill the observation tables in the assignment.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()