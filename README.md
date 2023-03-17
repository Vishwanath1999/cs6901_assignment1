# Feedforward Neural Network
This is an implementation of a feedforward neural network using Numpy. The network is built using customizable parameters such as network size, layer activation functions, initialization methods, learning rate, optimizer, regularization, batch size, and more.

## Prerequisites
- Python 3.x
- Numpy
- Matplotlib
- scikit-learn
- Seaborn
- Keras

## How to Use
The Jupyter Notebook allows you to tune the hparam, and train.py allows you to run the code with the provided hyperparameters.

```
python train.py --wandb_entity myname --wandb_project myprojectname
```
| Name | Default Value | Description |
| -------- | -------- | -------- |
|wandb_project|myprojectname|Project name used to track experiments in Weights & Biases dashboard|
|wandb_entity|myname|Wandb Entity used to track experiments in the Weights & Biases dashboard|
|dataset|fashion_mnist|choices: ["mnist", "fashion_mnist"]|
|epochs|10|Number of epochs to train neural network|
|batch_size|32|Batch size used to train neural network|
|loss_fn|'mse'|choices: ["mse", "cross_ent"]|
|optim_algo|'nadam'|choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]|
|lr|1e-3|Learning rate used to optimize model parameters|
|momentum|0.9|Momentum used by momentum and nag optimizers|
|beta|0.999|Beta used by rmsprop optimizer|
|beta1|0.9|Beta1 used by adam and nadam optimizers|
|beta2|0.9999|Beta2 used by adam and nadam optimizers|
|epsilon|0.000001|Epsilon used by optimizers|
|weight_decay|0|Weight decay used by optimizers|
|weight_init|'xavier_uniform'|choices: ["random", "xavier_uniform"]|
|num_layers|5|Number of hidden layers used in feedforward neural network|
|hidden_size|128|Number of hidden neurons in a feedforward layer|
|activation|sigmoid|choices: ["identity", "sigmoid", "tanh", "ReLU"]|

