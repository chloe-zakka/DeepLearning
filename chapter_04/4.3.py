import  numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
import matplotlib.pyplot as plt
# Data preprocessing aims at making raw data more amenable to neural networks.
# This includes vectorization, normalization, handling missing values, and feature extraction

    # Vectorization: transforming data into tensors
    # All inputs and targets in a neural network must be tensors of floating-point data

    # Value normalization:
    # Take small values: typically most values should be in the 0-1 range
    # Be homogenous: all features should take values in roughly the same range
    # Additionally, these are stricter normalization practices that aren't necessary, but helful
    # Normalize each feature independently to have a mean of 0
    # Normalize each feature independently to have a standard deviation of 1

x-= x.mean(axis=0)  # Assuming x is a 2d data matrix of shape (samples, features)
x/= x.std(axis=0)

    # Handling missing values:
    # If you have missing vals for data set, safe to input 0 w the condition that 0 is not already a meaningful value
    # Network will learn from exposure to the data that 0 means missing data and will ignore 0s

    # Feature engineering: Process of using your own knowledge about the data and about the machine learning algorithm
    # to make the algorithm better by applying hardcoded transformations to the data before it goes into the model
    # That was too complicated of a defintion, so here is one put into simpler terms:
    # Essentially boils down to creating new input features from existing ones to enable the machine learning algorithms
    # to uncover and exploit patterns more effectively
    # Example: Reading time on a clock:
    # Raw data is pixel grid depicting clock
    # You can engineer a feature that encodes the clock hands' coordinates
    # You can engineer a feature that encodes the angle of the clock hand



