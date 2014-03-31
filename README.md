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
the __dp__ package, whose first matter of business is to load its dependencies (see [dp/init.lua]):
```lua
require 'dp'
```
Note : package `underscore` in imported as `_`. So `_` shouldn't be used for dummy variables, instead 
use the much more annoying `__`, or whatnot. 

Then we make some hyper-parameters and other options available to the user via the command line:
```lua
--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST MLP Training/Optimization')
cmd:text('Example:')
cmd:text('$> th neuralnetwork.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--numHidden', 200, 'number of hidden units')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--type', 'double', 'type: double | float | cuda')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dropout', false, 'apply dropout on hidden neurons, requires "nnx" luarock')
cmd:option('--dataset', 'Mnist', 'which dataset to use : Mnist | NotMnist | Cifar10 | Cifar100')
cmd:option('--standardize', false, 'apply Standardize preprocessing')
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')
cmd:text()
opt = cmd:parse(arg or {})
print(opt)
```
We will come back to these later. For now, all you need to know is that `opt` is a table containing attributes
like `learningRate` and `numHidden`, and that these can be specified via the command line switches 
`--learningRate [number]` and `--numHidden [number]`, respectively. 

We are going to build and train a neural network so we need some data, 
which we encapsulate in a [DataSource](data/datasource.lua)
object. We provide the option of training on different datasets, 
notably [MNIST](data/mnist.lua), [NotMNIST](data/notmnist.lua), 
[CIFAR-10](data/cifar10) or [CIFAR-100](data/cifar100.lua):

```lua
--[[preprocessing]]--
local input_preprocess = {}
if opt.standardize then
   table.insert(input_preprocess, dp.Standardize())
end
if opt.zca then
   table.insert(input_preprocess, dp.ZCA())
end

--[[data]]--
local datasource
if opt.dataset == 'Mnist' then
   dp.Mnist{input_preprocess = input_preprocess}
elseif opt.dataset == 'NotMnist' then
   dp.NotMnist{input_preprocess = input_preprocess}
elseif opt.dataset == 'Cifar10' then
   dp.Cifar10{input_preprocess = input_preprocess}
elseif opt.dataset == 'Cifar100' then
   dp.Cifar100{input_preprocess = input_preprocess}
else
    error("Unknown Dataset")
end
```
Ok so we have a dataset, now we need a model:
```lua
--[[Model]]--
local dropout
if opt.dropout then
   require 'nnx'
   dropout = nn.Dropout()
end

mlp = dp.Sequential{
   models = {
      dp.Neural{
         input_size = datasource._feature_size, 
         output_size = opt.numHidden, 
         transfer = nn.Tanh()
      },
      dp.Neural{
         input_size = opt.numHidden, 
         output_size = #(datasource._classes),
         transfer = nn.LogSoftMax(),
         dropout = dropout
      }
   }
}

--[[GPU or CPU]]--
if opt.type == 'cuda' then
   require 'cutorch'
   require 'cunn'
   mlp:cuda()
end
```

An `EIDGenerator` is used for generating unique ids. It is initialized with a key that shouldn't be used 
by another experiments running in parallel. It is used to differentiate experiments and their associated logs.


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