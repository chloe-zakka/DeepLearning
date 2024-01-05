# Evaluating a model boils down to splitting the data into 3 sets:
# Training, Validation and Test.
import data
# Hyperparams: number of layers or the size of the layers.
# Params: the weights of the layers.

# Validation is necessary - Form of learning(search for a good configuration of hyperparams):
# Tuning by using as a feedback signal the performance of the model on the validation data
# Tuning model based on its performance in the validation set can lead to overfitting to the validation set
# Every time you tune hyperparams based on the validation set, some info about the validation data leaks into the model
# Remains reliable if you do this only once for one parameter, but too much info leaked into model if repeated

# 3 classic evaluation recipes:
# Simple hold-out validation, K-fold validation, Iterated K-fold validation with shuffling

# Simple hold-out validation: set apart some fraction of the data as a test set, train on remaining data,
# and evaluate on the test set.


import numpy as np
from keras.src.dtensor.integration_test_utils import get_model

num_validation_samples = 10000
# Always important to shuffle data:
np.random.shuffle(data)
# Define validation set:
validation_data = data[:num_validation_samples]
data = data[num_validation_samples:]
# Define training set:
training_data = data[:]
# Train model on training data, and evaluate it on validation data:
model = get_model()
model.train(training_data)
validation_score = model.evaluate(validation_data)

# Model can be tuned at this point, retrain it, evaluate it, tune it again, etc.
# Common to train final model from scratch on all non-test data available after tuning hyperparams:
model = get_model()
model.train(np.concatenate([training_data, validation_data]))

test_score = model.evaluate(test_data)

# K-fold validation: split data into K partitions of equal size.
    # For each partition i, train a model on the remaining K-1 partitions, and evaluate it on partition i.
    # Final score is the averages of the K scores obtained.
    # Helpful for when model shows significant variance in performance based on data it's trained on.

k = 4
num_validation_samples = len(data) // k
np.random.shuffle(data)
validation_scores = []
for fold in range(k):
    # Select validation data partition:
    validation_data = data[num_validation_samples * fold: num_validation_samples * (fold + 1)]
    # Use remainder of data as training data:
    training_data = data[:num_validation_samples * fold] + data[num_validation_samples * (fold + 1):]
    # Create a brand-new instance of the model (untrained):
    model = get_model()
    model.train(training_data)
    validation_score = model.evaluate(validation_data)
    validation_scores.append(validation_score)
# Average of validation scores of the k folds:
validation_score = np.average(validation_scores)
# Train final model on all non-test data available:
model = get_model()
model.train(data)
test_score = model.evaluate(test_data)

# Iterated K-fold validation with shuffling: same as K-fold, but repeats process multiple times,
# shuffling data every time

# When choosing evaluation protocol, keep in mind:
# Data representativeness: data should be representative of the data at hand
# The arrow of time: if you're trying to predict the future given the past, you should not randomly shuffle data before
    # splitting it - will create temporary leak. Always make sure all data in test set is posterior to the data in the
    # training set.
# Redundancy in data: if some data points appear twice in your data, then shuffling the data and splitting it into a
    # training set and a validation set will result in redundancy between the training and validation sets.
    # Make sure training and validation sets are disjoint.