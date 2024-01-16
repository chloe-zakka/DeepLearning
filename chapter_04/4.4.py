# Overfitting and underfitting:

# A model can quickly start to overfit to the training data
# Fundamental issue of ML: tension between optimization and generalization
# Optimization: process of adjusting a model to get the best performance possible on the training data
# Generalization: how well the trained model performs on data it has never seen before
# Goal of ML: get good generalization, but you don't control generalization directly

# At first, optimization and generalization are correlated: model is underfit (still has room to improve)
# But after a certain number of iterations on the training data, generalization stops improving
# and validation metrics stall & begin to degrade: model is overfitting (learns patterns too specific to training data)
# To prevent overfitting, you can:
# 1. Get more training data
# 2. Reduce capacity of network

# Reducing network capacity:
# A model with fewer parameters has less "memorization capacity", and therefore is forced to optimize
# more, and generalize better --> harder to overfit
# On the other hand, if the network has limited memorization resources, it won't be able to learn the mapping as easily
# -> Make sure you have enough params to prevent underfitiitng

# Example

# Original model:
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation='sigmoid'))

# Version with lower capacity:
model = models.Sequential()
model.add(layers.Dense(4, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(4, activation="relu"))
model.add(layers.Dense(1, activation='sigmoid'))

# Smaller model starts overfitting later than the reference one and its performance degrades more slowly
# once it starts overfitting

# Version with higher capacity:
model = models.Sequential()
model.add(layers.Dense(512, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(1, activation='sigmoid'))

# Bigger network starts overfitting almost immediately, and overfits much more severely.
# Its validation loss is also noisier

# Adding weight regularization:
# Simpler models are less likely to overfit than complex ones
# Simple model: less params
# Common way to mitigate overfitting is to put constraints on the complexity of a network by forcing its weights
# to only take small values, which makes the distribution of weight values more "regular"
# L1 regularization: cost added is proportional to the absolute value of the weights coefficients
# L2 regularization (weight decay): cost added is proportional to the square of the value of the weights coefficients

# Adding L2 weight regularization to the model:
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation="relu", input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# l2(0.001) --> every coef in the weight matrix of the layer will add 0.001 * weight_coefficient_value to the total loss
# of the network. This is only added in training time, not in testing time
# Model with L2 regularization is much more resistant to overfitting than the reference model

# Different weight regularizers available in Keras:
from keras import regularizers

regularizers.l1(0.001)  # L1 regularization
regularizers.l1_l2(l1=0.001, l2=0.001)  # L1 and L2 regularization combined

# Adding dropout:
# Dropout is one of the most effective and most commonly used regularization techniques for neural networks
# Consists of randomly "dropping out" (setting to zero) a number of output features of the layer during training
# Droupout rate: fraction of the features that are zeroed out (between 0.2 and 0.5)
# at test time, no units are dropped out ; they are scaled down by a factor = to droupout rate to balance for the fact
# that more units are active than at training time

layer_output *= np.random.randint(0, high=2, size=layer_output.shape)  # At training time, drops 50% of units in output
layer_output *= 0.5  # At test time, scales down the output by 0.5

# Process can be implemented by doing both operations at training time and leaving the output unchanged at test time:
layer_output *= np.random.randint(0, high=2, size=layer_output.shape)  # At training time, drops 50% of units in output
layer_output /= 0.5  # At test time, scales up the output by 0.5

# You can also have droupout applied to an activation matrix at training time, with rescaling happening during training.
# At test time, activation matrix is unchanged.

# Adding dropout to the IMDB network:
model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))



# RECAP:
# To prevent overfitting:
# Get more training data
# Reduce capacity of network
# Add weight regularization
# Add dropout