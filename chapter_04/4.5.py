# Defining the problem and assembling a dataset
# 1. What will your input be? What are you trying to predict? Make sure you have available training data
# 2. What type of problem are you facing? Identify the problem type to guide choice of model architecture

# Hypothesis: Outputs can be predicted given your inputs, and that your available data is informative enough to learn
# the relationship between inputs and outputs

# Not all problems can be solved.
# ex: nonstationary problems.
    # Say you're trying to build a recommendation engine for clothing, and you're training it on data from the month of
    # August. If you want to generate recommendations for the winter, you cannot. Buying clothes is a nonstationary
    # phenomenon over the scake of a few months.

# Choosing measure of success
# You must define what you mean by success. Accuracy? Precision? Customer-retention rate?
# Measure of success will serve as guide for choice of loss function, i.e, what the model will optimize

# Deciding on an evaluation protocol
# How do you measure current progress?
    # Maintaining a hold-out validation set: Use when you have a lot of data
    # Doing K-fold cross-validation: Use when you have too few samples for hold-out validation to be reliable
    # Doing iterated K-fold validation: Use for small datasets

# Preparing your data
# Data should be formatted as tensors, scaled to small values, and normalized (maybe feature engineering as well)

# Developing a model that does better than a baseline
# Goal: achieve statistical power: a model that outperforms a dumb baseline.
# If statistical power is not achieved after tyring multiple reasonable architectures, then hypothesis are false

# Assuming you have a working model that achieves statistical power, then you need to make 3 key choices to build model:
# 1. Last-layer activation: determines range that model's output will lie in / a constraint on output (sigmoid for IMDB)
# 2. Loss function: determines how learning will proceed (binary crossentropy for IMDB)
# 3. Optimization configuration: determines specifics of gradient descent (choice of optimizer, learning rate...)

# ----------------------------------------------------------------------------------------------------------------------
# Problem type                      Last-layer activation             Loss function
# ----------------------------------------------------------------------------------------------------------------------
# Binary classification             sigmoid                           binary_crossentropy
# Multiclass, single-label          softmax                           categorical_crossentropy
# Multiclass, multilabel            sigmoid                           binary_crossentropy
# Regression to arbitrary values    None                              mse
# Regression to values between      0 and 1 sigmoid                     mse or binary_crossentropy
# ----------------------------------------------------------------------------------------------------------------------

# Scaling up: Is the model powerful enough? enough layers and params?
# Develop a model that overfits (add layers, make them bigger, train for more epochs)
# Monitor training loss and validation loss, as well as training and validation values for any metrics you care about
# Once you overfit, regularize and tune the model (add dropout, add/remove layers, add L1/L2 regularization, try diff
# hyperparams, iterate on feature engineering) to get close to a model that neither overfits nor underfits

# Be careful; everytime you use feedback from your model to tune it, you leak information about the validation set into

# Once model configuration is satisfactory, train final product on all the available data and evaluate one last time on
# the test set
# If the performance is significantly worse than that of validation data, then either validation procedure wasn't
# reliable or that model was overfitting to validation data (switch to K-fold validation)