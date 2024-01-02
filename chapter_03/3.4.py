# Binary classification is the most widely applied kind of machine-learning problem.

# You should never test a machine-learning model on the same data that you used to train it!
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# num_words=10000 means you’ll only keep the top 10,000 most frequently occurring words in the training data.
# Rare words will be discarded. This allows you to work with vector data of manageable size.
# train_data and test_data are lists of reviews; each review is a list of word indices (encoding a sequence of words).
# train_labels and test_labels are lists of 0s and 1s, where 0 stands for negative and 1 stands for positive:

print(train_data[0])
print(train_labels[0])

# Because you’re restricting yourself to the top 10,000 most frequent words, no word index will exceed 10,000:
print(max([max(sequence) for sequence in train_data]))  # 9999

# How to decode back to English?
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ''.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# Preparing your data
# You can’t feed lists of integers into a neural network. You have to turn your lists into tensors.
# There are two ways to do that:
#   Pad your lists so that they all have the same len, turn them into an int tensor of shape (samples, word_indices),
#   and then use as the first layer in your network a layer capable of handling such int tensors (the Embedding layer).
#   OR, one-hot encode your lists to turn them into vectors of 0s and 1s. For e.g., turn sequence [3, 5] into a 10,000-
#   dimensional vector that would be all 0s except for indices 3 and 5, which would be 1s.
#   Then you could use as the first layer in your network a Dense layer, capable of handling floating-point vector data.

# Encoding the integer sequences into a binary matrix
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # Creates an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        # Sets specific indices of results[i] to 1s
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)  # Vectorized training data
x_test = vectorize_sequences(test_data)  # Vectorized test data
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# A type of network that performs well on such a problem is a simple stack of fully connected (Dense) layers with relu
# activations: Dense(16, activation='relu'). The argument being passed to each Dense layer (16) is the number of hidden
# units of the layer (dimension in the representation space of the layer)
# 16 hidden units means that the weight matrix W will have shape (input_dimension, 16)

# There are 2 key architecture decisions to be made about such a stack of Dense layers:
#   How many layers to use
#   How many hidden units to choose for each layer

# Following architure is used generally:
#   2 intermediate layers with 16 hidden units each
#   A 3rd layer that will output the scalar prediction regarding the sentiment of the current review

# Model definition:
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compilation
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Configuring the optimizer
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Using custom losses and metrics
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

# Validating your approach: by training accuracy od the model on data it has never seen before,
# you'll create a validation set by setting apart 10,000 samples from the original training data.
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Train the model for 20 epochs, in mini-batches of 512 samples. At the same time, monitor loss and accuracy on the
# 10,000 samples that you set apart. (Pass the validation data as the validation_data argument.)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512,
                    validation_data=(x_val, y_val))
history_dict = history.history
print(history_dict.keys())

# Plotting the training and validation loss
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,len(loss_values)+1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Plotting the training and validation accuracy

plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Retraining a model from scratch
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

