from keras import layers
from keras import models

# The notion of layer compatibility refers to the fact that every layer will only
# accept input tensors of a certain shape and will return output tensors of a certain shape.
# layer = layers.Dense(32, input_shape=(784,)) --> will only take 2D tensor of shape (784, ) as input
# The layer will also return a tensor of shape (32, )

model = models.Sequential()
model.add(layers.Dense(32, input_shape=(784,)))
model.add(layers.Dense(32)) # will automatically infer its input shape as that of the layer that came before
