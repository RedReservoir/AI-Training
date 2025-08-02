# AI-Training

## Deep Learning

### Session 1: Introduction to PyTorch

In this session (notebook `dl_introduction.ipynb`) we learn the basics of PyTorch by optimizing a 1-parameter model on a set of noisy points. Topics covered:

  - What are the loss function and loss landscape.
  - How the Gradient Descent method works.
  - What are the `torch.Tensor` and `torch.nn.Parameter` classes.
  - Basics of PyTorch Optimizers.
  - How are gradients computed, stored and utilized in PyTorch
  - Basics of PyTorch Learning Rate Schedulers.
  - Sensibility of model training wrt. learning rate.
  - The training loop in detail, with basic metric logging and visualization.

Afterwards, an exercice (`dl_introduction_exercice.ipynb`) is proposed where the session steps must be reproduced to optimize a 3-parameter model.

### Session 2: Deep Learning with the MNIST Dataset

In this session (notebook `dl_training_loop.ipynb`) we explore how to load and feed data to our models in Pytorch. Topics covered:

  - Basics of `numpy` and `matplotlib` to load and view array data.
  - What is the `torch.nn.Module` class.
  - How to create our own PyTorch models.
  - What is the `torch.utils.data.Dataset` class.
  - How to create our own PyTorch datasets and how to split them.
  - What is the `torch.utils.data.DataLoader` class.
  - How to control dataset sampling options in PyTorch.
  - The train and evaluation loops.
  - Loss and metric monitoring. Early stopping.
  - The Cross-Entropy loss.

## Machine Learning

### Session 1: PCA - An example of Linear Algebra application

In this session (notebook `ml_pca_linalg.ipynb`) we see an example of dimensionality reduction as application of linear algebra. Topics covered:

  - A quick review on linear algebra.
  - First contact with a ML dataset.
  - PCA from the inside step by step.
  - First contact with a ML regressor.
  - Applying PCA to an example in real life and assessing its effects on final performance.

