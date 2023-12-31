from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# A layer is a filter for data — data goes in, and comes out in its most useful form.
# Specifically, layers extract representations out of the data fed into them
# — hopefully, representations that are more meaningful for the problem at hand.

# Network made up of 2 Dense layers
# The second is a 10-way softmax layer — it will return an array of 10 probability scores (summing to 1).
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# Compilation
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Preparing the image data - reshaping and scaling so values are in [0,1] instead of [0,255]
# (Since they were stored as 8-bit integers)
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Preparing the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Train the network — we fit the model to its training data
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Check that the model performs well on the test set, too:
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

# Test accuracy ≠ training accuracy which is an indication of overfitting
# Overfitting: the fact that machine-learning models tend to perform worse on new data than on their training data.
