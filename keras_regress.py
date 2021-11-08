

import array
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import float32



np.random.seed(42)

observations = 1000

x = np.random.uniform(low=-10, high=10, size= (observations, 1))


print(x.shape)
np.sort(x)

noise = np.random.uniform(-1, 1, (observations,1))
targets = 13*x+2 +noise
print(targets.shape)

plt.plot(x,targets)
plt.ylabel("Targets")
plt.xlabel("Input")
plt.title("Data")
# plt.show()


np.savez('TF_intro', inputs = x, targets = targets)

training_data = np.load('TF_intro.npz')

input_size = 1
output_size = 1



model = tf.keras.Sequential([
    tf.keras.layers.Input(shape = (input_size, ) ),
    tf.keras.layers.Dense(output_size,
        kernel_initializer = tf.random_uniform_initializer(minval = -0.1, maxval = 0.1),
        bias_initializer = tf.random_uniform_initializer(minval= -0.1, maxval = 0.1)
    )
])

model.summary()

custom_optimizer = tf.keras.optimizers.SGD(learning_rate = 0.02)

model.compile(optimizer = custom_optimizer, loss = 'mse')

model.fit(training_data['inputs'], training_data['targets'], epochs = 100, verbose = 2)

model.layers[0].get_weights()

weights = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]
bias, weights
