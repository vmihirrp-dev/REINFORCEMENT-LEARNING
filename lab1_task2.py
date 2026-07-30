import gymnasium as gym


env = gym.make("CartPole-v1")


observation, info = env.reset()

print("Initial Observation:")
print(observation)

print("\nEnvironment Info:")
print(info)