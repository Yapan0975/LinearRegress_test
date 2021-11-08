from os import initgroups
import numpy as np

import matplotlib.pyplot as plt

np.random.seed(42)

observations = 1000

x = np.random.uniform(low=-10, high=10, size= (observations, 1))

print(x.shape)

noise = np.random.uniform(-1, 1, (observations,1))
targets = 13*x+2 +noise
print(targets.shape)

plt.plot(x,targets)
plt.ylabel("Targets")
plt.xlabel("Input")
plt.title("Data")
plt.show()

init_range = 0.1
weights = np.random.uniform(low= -init_range, high=init_range, size = (1, 1))
biases = np.random.uniform(low= -init_range, high= init_range, size = 1)

learning_rate = 0.02

losses = []

for i in range(100):
    outputs = np.dot(x, weights)+biases
    deltas = outputs-targets
    loss = np.sum(deltas**2) / 2 / observations

    print(loss)

    losses.append(loss)

    deltas_scaled = deltas / observations

    weights = weights-learning_rate*np.dot(x.T, deltas_scaled)
    biases = biases -learning_rate*np.sum(deltas_scaled)

plt.plot(range(100), losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Training")
plt.show()

print(weights, biases)