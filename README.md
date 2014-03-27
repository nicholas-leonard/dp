# dp Package Reference Manual#

__dp__ is a deep learning framework based on the [Torch7](http://torch.ch) distribution and 
inspired by pylearn2/Theano. It provides common datasets like MNIST, CIFAR-10 and CIFAR-100, 
preprocessing like Zero-Component Analysis, Global Contrast Normalization, Lecunn Local Contrast Normalization  and 
facilities for interfacing your own. Additionally, it provides a higher-level framework that 
abstracts away common usage patterns of the [nn](https://github.com/torch/nn/blob/master/README.md) 
and [torch7](https://github.com/torch/torch7/blob/master/README.md) packages such as [early stopping](http://en.wikipedia.org/wiki/Early_stopping), dataset, and allows for more
flexible representations. The framework includes hyperparameter optimization facilities for 
sampling and running experiment from the cmd-line or prior hyper-parameter distributions.
It provides facilites for storing and analysing experimental hyperpameters and results using
a PostgreSQL database backend, which facilitates running many experiments on different machines. 

## Neural Network Example ##
A quick example to show what can be expected by this package.

## Data and preprocessing ##
DataTensor, DataSet, DataSource, Samplers and Preprocessing.
DataSource is made up of 3 DataSet instances : train, valid and test.

## Models and states ##
Model, Sequential, etc.

## Experiments ##
Experiment, Propagator, etc.

## Hyperparameter optimization ##
Big section. Starts with an example. MLPBuilder, etc. 

## Install ##
```shell
sudo apt-get install libpq-dev
sudo luarocks install luasql-postgres PGSQL_INCDIR=/usr/include/postgresql
sudo luarocks install fs
sudo luarocks install underscore
sudo luarocks install nnx
sudo apt-get install liblapack-dev
```