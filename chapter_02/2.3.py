import keras
import numpy as np
# What does a Keras layer look like?
keras.layers.Dense(512, activation='relu')


# Layer is represented as a function which takes as input a 2D tensor and returns another 2D tensor
# The function is as follows: output = relu(dot(W, input) + b)

# Element-wise operations


def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()  # Avoid overwriting the input tensor
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x

    # For addition


def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x

# Broadcasting
    # Consists of two steps:
    # Axes are added to match the ndim of the larger tensor
    # The smaller tensor is repeated alongside these new axes to match the full shape of the larger tensor

def naive_add_matrix_and_vector(x,y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x
# Tensor dot (z = x . y) - only vectors with the same number of elements are compatible for a dot product


def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape == y.shape
    z = 0
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z

    # Matrix dot product
    # x.shape = (a,b) and y.shape = (b,c) then z.shape = (a,c)

    # Tensor reshaping: rearranging rows and columns to match a target shape.
    # Reshaped tensor has the same total number of coefficients as the initial tensor
    # Usually used to transpose (exchange rows and columns)


x = np.zeros((300, 20))
x = np.transpose(x)
print(x.shape)  # (20, 300)

