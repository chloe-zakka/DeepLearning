# Keras is a deep-learning framework for Python that provides a convenient way to define
#       and train almost any kind of deep-learning model

# Keras provides high-level building blocks for developing deep-learning models
# Does not handle low-level operations such as tensor manipulation and differentiation
# Relies on a specialized library – its backend engine — to perform these operations
    # 3 choices: TensorFlow, Theano, and CNTK

# Developing with Keras typically involves the following steps:
# 1. Define your training data: input tensors and target tensors
# 2. Define a network of layers (or model) that maps your inputs to your targets
# 3. Configure the learning process by choosing a loss function, an optimizer, and some metrics to monitor
# 4. Iterate on your training data by calling the fit() method of your model

# There are 2 ways to define a model:
#    1. Sequential class: For linear stack of layers (most common)
#    2. Functional API : For directed acyclic graphs of layers (arbitrary architectures)

# E.g of Sequential class for two layer model:
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(784,)))
model.add(layers.Dense(10, activation='softmax'))

# E.g of Functional API for the same two layer model:
input_tensor = layers.Input(shape=(784,))
x = layers.Dense(32, activation='relu')(input_tensor)
output_tensor = layers.Dense(10, activation='softmax')(x)
model = models.Model(inputs=input_tensor, outputs=output_tensor)

# For the next steps, they are all the same regardless of which way you choose to define your model
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='mse',
              metrics=['accuracy'])

# Finally, the learning process consists of passing Numpy arrays of input data and corresponding target data to the
# model via the fit() method

# model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)
