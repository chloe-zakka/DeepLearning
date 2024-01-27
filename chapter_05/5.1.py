# Instantiate a small convnet for MNIST classification
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

print(model.summary())

# The code show what a basic convnet looks like. It's a stack of Conv2D and MaxPooling2D layers.
# The output of every Conv2D and MaxPooling2D layer is a 3D tensor of shape (height, width, channels).
# The width and height dimensions tend to shrink as you go deeper in the network.
# The number of channels is controlled by the first argument passed to the Conv2D layers (32 or 64).


# Now, we add a classifier on top of the convnet

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

print(model.summary())

# The (3, 3, 64) outputs are flattened into vectors of shape (576, 1) before going through two Dense layers.

# Now, we train the convnet on the MNIST digits

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000 ,28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Evaluating the model on the test data:
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)

# The test accuracy is 99.2%. Comparing this to the densely connected network from chapter 2 (accuracy of 97.8%), we see
# that the error rate decreased by 68%.

# Main difference between a densely connected layer and a convolutional layer:
# Dense layers learn global patterns in their input feature space (for example, for a MNIST digit, patterns involving
# all pixels)
# Convolution layers learn local patterns: in the case of images, patterns found in small 2D windows

# More detailed explanation I made for myself:
# Densely Connected Layer (Fully Connected Layer):
    # Every neuron in this layer is connected to every neuron in the previous layer.
    # It learns global patterns in the data.
    # Typically used at the end of a neural network for classification tasks.
# Convolution Layer:
    # Neurons in this layer are not connected to every neuron in the previous layer but only to a small region of it.
    # It learns local patterns, like textures and shapes in images.
    # Commonly used in image processing tasks for feature extraction

# Convolutional layer to extract features from an image, breaking them up into local patterns (edges, textures etc)
# This characteristic gives convnets two interesting properties:
    # The patterns they learn are translation invariant: After learning a certain pattern in the lower-right corner of a
    # picture, a convnet can recognize it anywhere

    # They can learn spatial hierarchies of patterns: A first convolution layer will learn small local patterns, a
    # second convolution layer will learn larger patterns made of the features of the first layers and so on.
    # More efficient for learning complex and abstarct visuals


# Convolutions are defined by two key parameters:
    # Size of the patches extracted from the inputs (typically 3x3 or 5x5)
    # Depth of the output feature map (number of filters computed by the convolution)

# ** Intermission**
# Here I got a little confused because the concept of depth is used a lot but not really explained**
# After researching, this is the easiest explanation I was able to put together:
# Depth of an output is referring to how many different types of features it can detect in an image
# Think of each "feature" as a specific detail in the image, like an edge, a corner, or a color spot.
# The "depth" is like having a set of different glasses, each showing you a different kind of detail.
    # A depth of 32 means you have 32 different types of glasses, each highlighting a unique detail in the image.
    # When you increasing the depth helps the neural network understand and analyze the image better.

# ** End of Intermission **

# How does the convolution operate?
# 1. Slide the convolution window (3x3 or 5x5) across the input feature map
# 2. Stop at every possible location and extract patch of pixels
# 3. Transform patch into 1D vector (dot product between the weights of the convolutional layer and the patch)
# 4. Vectors spatially reassembled to form 3D output feature map

# Convolution may cause output to be different size than input. This is because of border effects and/or strides.

# Understanding border effects and padding:
# Border effect refers to how the network handles the edges of the input data during the convolution process.
# BUT, border effect may be problematic because it shrinks the size of the output. How do we fix this? Padding!
# Padding consists of adding an appropriate number of rows and columns on each side of the input feature map to fit
# center convolution windows
# Super easy to code: in Conv2D, add padding='same' to add padding to the input feature map

# Understanding strides:
# Stride is the distance between two successive windows. By default, it's 1.
# It's possible to have strides greater than 1. This can be a problem:
# Filter moves two pixels over (instead of one) each time it scans the image. This results in an output feature map
# that is half the width and half the height of the original image. So, the image gets smaller.
# Strided convolutions rarely ever used

# To downsample feature maps, use max-pooling instead of strides.
# Max-pooling: Aggressively downsample feature maps by transforming extracted input them via hardcoded max tensor ops.
# Done w 2x2 windows and a stride of 2

# Why max-pooling? What's the point? Let's see!
# No max-pooling:
model_no_max_pool = models.Sequential()
model_no_max_pool.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model_no_max_pool.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_no_max_pool.add(layers.Conv2D(64, (3, 3), activation='relu'))
print(model_no_max_pool.summary())

# What's wrong with this set up?
# 1. Using only small windows (like 3x3) in convolutional layers limits the network's ability to learn complex,
# high-level patterns, as it restricts the view to very small portions of the input, making it challenging to recognize
# larger structures like digits
# 2. The final feature map has 22 × 22 × 64 = 30,976 total coefficients per sample. This is huge. Will overfit.

# What's the point of downsampling?
# In short, the reason to use downsampling is to reduce the number of feature-map coefficients to process, as well as
# to induce spatial-filter hierarchies by making successive convolution layers look at increasingly large windows