# The Four Branches of Machine Learning

#   Supervised Learning: Goal is to learn the relationship between input and training targets
    # It's the most common type of machine learning
    # It's often used to predict the future and consists mostly of classification and regression. There are more exotic
    # variants as well, such as:
    # - Sequence generation: Given a picture, predict a caption describing it.
    # - Syntax tree prediction: Given a sentence, predict its decomposition into a syntax tree.
    # - Object detection: Given a picture, draw a bounding box around certain objects inside the picture. \
    # - Image segmentation: Given a picture, draw a pixel-level mask on a specific object.

#   Unsupervised Learning: Goal is to find interesting transformations of the input without the help of any targets
    # Used to better understand the correlations present in the data at hand
    # Diminsionality reduction and clustering are well-known categories of unsupervised learning

#   Self-supervised Learning
    # Specific instance of supervised learning ; ≠enough to warrant its own category; doesn't use human-annotated labels
    # No humans in the loop. Labels still involved, but they're generated from the input data,
    # typically using a heuristic algorithm

    # The distinction between supervised, self-supervised, and unsupervised learning can be blurry sometimes
    # —these categories are more of a continuum without solid borders. Self-supervised learning can be reinterpreted as
    # either supervised or unsupervised learning, depending on whether you pay attention to the learning mechanism or to
    # the context of its application.

#   Reinforcement Learning
    # Agent receives info about its environment and learns to choose actions that will maximize some reward.

#   Glossary
    # Sample or input: One data point that goes into your model.
    # Prediction or output: What comes out of your model.
    # Target: The truth. What your model should ideally have predicted, according to an external source of data.
    # Prediction error or loss value: A measure of the distance between your model's prediction and the target.
    # Classes: A set of possible labels to choose from in a classification problem.
    # Label: A specific instance of a class annotation in a classification problem.
    # Ground-truth or annotations: All targets for a dataset, typically collected by humans.
    # Binary classification: Classification task - each input sample should be categorized into 2 exclusive categories.
    # Multiclass classification: Classification task - each input sample categorized into more than 2 categories.
    # Multilabel classification: Classification task - each input sample can be assigned multiple labels.
    # Scalar regression: Regression where the target is a continuous scalar value.
    # Vector regression: Regression where the target is a set of continuous values: a vector.
    # Mini-batch or batch: Small set of samples (between 8 & 128) that are processed simultaneously by the model.

