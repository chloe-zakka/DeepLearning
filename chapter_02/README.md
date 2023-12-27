# Chapter 2 Overview
This chapter lays the groundwork for understanding the operational dynamics of neural networks, emphasizing the critical roles played by loss functions and optimization techniques in the training journey.
## Key Concepts

* **Core of Neural Network Learning**: he fundamental aspect of learning in neural networks is to optimize model parameters in a way that minimizes the loss function. This function is pivotal as it quantifies the model's performance on the provided training data and their corresponding targets.
* **Mechanics of Learning**: The learning process unfolds through:
  * Selecting random subsets of data and their targets.
  * Calculating the gradient with respect to the model parameters based on the loss function for these subsets.
  * Modifying the parameters slightly in the reverse direction of the gradient, guided by a parameter known as the learning rate.
* **Differentiable Operations and Chain Rule**: The backbone of neural networks consists of tensor operations that are differentiable. This characteristic enables the application of the chain rule in calculus for efficient gradient computation, a crucial step in the learning algorithm.
* **Key Components - Loss and Optimizers**: Two vital elements in the training process are:
     * **Loss Function**: It acts as a success metric for the task at hand and is the target for minimization during training.
    * **Optimizer**: These define the methodology for utilizing the loss gradients to update the model parameters. Examples include optimizers like RMSProp and SGD with momentum.

