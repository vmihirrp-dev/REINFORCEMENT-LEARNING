import gymnasium as gym

env = gym.make("CartPole-v1")

observation, info = env.reset()

print("Initial Observation:")
print(observation)

print("\nEnvironment Info:")
print(info)

print("\nObservation Space:")
print(env.observation_space)

print("\nAction Space:")
print(env.action_space)

print("\nObservation Space Type:")
print(type(env.observation_space))

print("\nNumber of Possible Actions:")
print(env.action_space.n)