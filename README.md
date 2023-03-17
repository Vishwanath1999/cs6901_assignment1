# Feedforward Neural Network
This is an implementation of a feedforward neural network using Numpy. The network is built using customizable parameters such as network size, layer activation functions, initialization methods, learning rate, optimizer, regularization, batch size, and more.

## Prerequisites
- Python 3.x
- Numpy
- Matplotlib
- scikit-learn
- Seaborn
- Keras

## Data
The network has been trainied with a wide range of hyperparameters listed below on fashion MNIST data. The Keras package has been used to import the data. Other than that keras serves no purpose here. The images are of type 8 bit integers (uint8) between 0 and 255. They are normalize between 0 and 1 and converted to float data type.

![alt text](https://github.com/Vishwanath1999/cs6901_assignment1/blob/master/fashion_mnist.png)

```
from keras.datasets fashion_mnist,mnist
```

## How to Use
The Jupyter Notebook allows you to tune the hyper-parameters, and train.py allows you to run the code with the provided hyperparameters.

```
jupyter My_mnist.ipynb
```
## Wandb sweep config
```
sweep_config = {
    'method':'bayes',
    'metric':{
    'name':'val_acc',
    'goal':'maximize'
    },
    'parameters':{
    'n_epochs':{
    'values':[5,10]
    },
    'n_hidden':{
    'values':[3,4,5]
    },
    'n_hidden_units':{
    'values':[32,64,128]
    },
    'l2_coeff':{
    'values':[0,5e-4,5e-1]
    },
    'lr':{
    'values':[1e-3,1e-4]
    },
    'optim_algo':{
    'values':['sgd','sgdm','rmsprop','adam','nadam','nag']
    },
    'batch_size':{
    'values':[16,32,64]
    },
    'weights_init':{
    'values':['random','xavier_uniform']
    },
    'act_func':{
    'values':['relu','sigmoid','tanh','identity']
    },
    'loss_func':{
    'values':['cross_ent','mse']
    },
    'relu_param':{
    'values':[0,1e-1,1e-2,1e-3]
    }
    }
}

```
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
|beta2|0.999|Beta2 used by adam and nadam optimizers|
|epsilon|0.000001|Epsilon used by optimizers|
|weight_decay|0|Weight decay used by optimizers|
|weight_init|'xavier_uniform'|choices: ["random", "xavier_uniform"]|
|num_layers|5|Number of hidden layers used in feedforward neural network|
|hidden_size|128|Number of hidden neurons in a feedforward layer|
|activation|sigmoid|choices: ["identity", "sigmoid", "tanh", "ReLU"]|

## Confusion matrix for fashion MNIST
![alt text](https://github.com/Vishwanath1999/cs6901_assignment1/blob/master/confusion_matrix.png)

## References
- https://ml-cheatsheet.readthedocs.io/en/latest/index.html
- [Sentdex - Neural Network from scratch](https://youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)
- [CS7015- Deep Learning](https://youtube.com/playlist?list=PLyqSpQzTE6M9gCgajvQbc68Hk_JKGBAYT)
- https://www.deeplearningbook.org/
