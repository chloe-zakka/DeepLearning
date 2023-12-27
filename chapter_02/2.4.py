import keras
# Each neural layer from the first network example transforms its input data as follows:
# output = relu(dot(W, input) + b)
# W and b are tensors that are weights (of Kernel and bias respectively)
# Initially random - although not useful, is a good starting point.
# Gradual adjustments made through training loops in order to reduce the loss function

# How to find the right values for the weights and biases?
# Take advantage of the fact that all operations used in the network are differentiable
# The gradient function is finding the points for which the loss function is minimal --> where its derivative is 0

# Use four-step algorithm:
# 1- Draw a batch of training samples x and corresponding targets y
# 2- Run the network on x to obtain predictions y_pred
# 3- Compute the loss of the network on the batch, a measure of the mismatch between y_pred and y,
# and compute the gradient of the loss with regard to the network's parameters
# 4- Move the parameters a little in the opposite direction from the gradient


# Global minimus vs local minimums
past_velocity = 0.
momentum = 0.1
while loss > 0.01:  # This is a toy example
    w, loss, gradient = get_current_parameters()
    velocity = past_velocity * momentum - learning_rate * gradient
    w = w + momentum * velocity - learning_rate * gradient
    past_velocity = velocity
    update_parameter(w)

# Don't want the ball to lose momentum and get stuck in local minimums
