from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Loading the MNIST dataset in Keras

# training data
print(train_images.shape)
print(len(train_labels))
print(train_labels)

# test data
print(test_images.shape)
print(len(test_labels))
print(test_labels)
