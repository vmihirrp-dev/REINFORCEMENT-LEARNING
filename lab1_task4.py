import gymnasium as gym

# Create the CartPole environment
env = gym.make("CartPole-v1")

# Reset the environment
observation, info = env.reset()

# Initialize counters
step = 0
total_reward = 0

print("Starting Random Agent...\n")

# Run one episode
while True:

    # Select a random action
    action = env.action_space.sample()

    # Execute the action
    observation, reward, terminated, truncated, info = env.step(action)

    # Update counters
    step += 1
    total_reward += reward

    # Print step details
    print(f"Step: {step}")
    print(f"Action: {action}")
    print(f"Observation: {observation}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print("-" * 40)

    # End the episode if finished
    if terminated or truncated:
        break

print("\nEpisode Finished!")
print(f"Total Steps: {step}")
print(f"Total Reward: {total_reward}")

# Close the environment
env.close()