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

<a name="NeuralNetworkExample"/>
## Neural Network Example ##
We begin with a simple [neural network example](examples/neuralnetwork.lua). The first line loads 
the __dp__ package, whose first matter of business is to load its dependencies (see [dp/init.lua](dp/init.lua)):
```lua
require 'dp'
```
Note : package `underscore` in imported as `_`. So `_` shouldn't be used for dummy variables, instead 
use the much more annoying `__`, or whatnot. 

Lets define some hyper-parameters and store them in a table. We will need them later:
```lua
--[[hyperparameters]]--
opt = {
   nHidden = 100, --number of hidden units
   learningRate = 0.1, --training learning rate
   momentum = 0.9, --momentum factor to use for training
   maxOutNorm = 1, --maximum norm allowed for outgoing weights
   batchSize = 128, --number of examples per mini-batch
   maxTries = 100, --maximum number of epochs without reduction in validation error.
   maxEpoch = 1000 --maximum number of epochs of training
}
```
We intend to build and train a neural network so we need some data, 
which we encapsulate in a [DataSource](data/datasource.lua)
object. __dp__ provides the option of training on different datasets, 
notably [MNIST](data/mnist.lua), [NotMNIST](data/notmnist.lua), 
[CIFAR-10](data/cifar10) or [CIFAR-100](data/cifar100.lua), but for this
tutorial we will be using the archtypical MNIST (don't leave home without it):
```lua
--[[data]]--
datasource = dp.Mnist{input_preprocess = dp.Standardize()}
```
A `datasource` contains up to three [Datasets](data/dataset.lua): 
`train`, `valid` and `test`. The first if for training the model. 
The second is used for [early-stopping](observer/earlystopper.lua).
The third is used for publishing papers and comparing different models.
  
Although not really necessary, we [Standardize](preprocess/standardize.lua) 
the datasource, which subtracts the mean and divides 
by the standard deviation. Both statistics (mean and standard deviation) are 
measured on the `train` set only. This is common pattern when preprocessing. 
When statistics need to be measured accross different examples 
(as in [ZCA](preprocess/zca.lua) and [LecunLCN](preprocess/lecunlcn.lua) preprocesses), 
we fit the preprocessor on the `train` set and apply it to all sets (`train`, `valid` and `test`). 
However, some preprocesses require that statistics be measured
only on each example (as in [global constrast normalization](preprocess/gcn.lua)). 

Ok so we have a `datasource`, now we need a model:
```lua
--[[Model]]--
mlp = dp.Sequential{
   models = {
      dp.Neural{
         input_size = datasource:featureSize(), 
         output_size = opt.nHidden, 
         transfer = nn.Tanh()
      },
      dp.Neural{
         input_size = opt.nHidden, 
         output_size = #(datasource:classes()),
         transfer = nn.LogSoftMax()
      }
   }
}
```



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