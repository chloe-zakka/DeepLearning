from keras import layers
from keras import models

# The notion of layer compatibility refers to the fact that every layer will only
# accept input tensors of a certain shape and will return output tensors of a certain shape.
# layer = layers.Dense(32, input_shape=(784,)) --> will only take 2D tensor of shape (784, ) as input
# The layer will also return a tensor of shape (32, )

model = models.Sequential()
model.add(layers.Dense(32, input_shape=(784,)))
model.add(layers.Dense(32)) # will automatically infer its input shape as that of the layer that came before

# The topology of a network defines a hypothesis space.
# By choosing a network topology, you constrain your space of possibilities (hypothesis space)
# to a specific series of tensor operations, mapping input data to output data.
# You then search for a good set of values for the weight tensors of the tensor operations

# Loss functions and optimizers: keys to configuring the learning process
# Once you define the network architecture, you still have to choose 2 more things:
# 1. Loss function (objective function) - the quantity that will be minimized during training(measure of success)
# 2. Optimizer - Determines how the network will be updated based on the loss function.
#                Implements a specific variant of stochastic gradient descent (SGD)

# A neural network that has multiple outputs may have multiple loss functions (one per output)
# Gradient descent process must be based on a SINGLE scalar loss value (for multiloss networks all losses are combined)

# Choosing the right objective function for the right problem is extremely important:
# Network will take any shortcut it can to minimize loss; if the objective doesn’t fully correlate with success for the
#   task at hand, your network will end up doing things you may not have wanted.

