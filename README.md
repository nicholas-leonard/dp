# dp Package Reference Manual#

__dp__ is a <b>d</b>ee<b>p</b> learning library designed for streamlining 
research and development using the [Torch7](http://torch.ch) distribution, with an emphasis on flexibility through the elegant use of object-oriented [design patterns](http://en.wikipedia.org/wiki/Design_Patterns).

Inspired by pylearn2/Theano, it provides common datasets like MNIST, CIFAR-10 and CIFAR-100, 
preprocessing like Zero-Component Analysis whitening, Global Contrast Normalization, 
Lecunn's Local Contrast Normalization  and facilities for interfacing your own. 
Additionally, it provides a high-level framework that abstracts away common usage patterns of the [nn](https://github.com/torch/nn/blob/master/README.md) 
and [torch7](https://github.com/torch/torch7/blob/master/README.md) package such as 
loading datasets and [early stopping](http://en.wikipedia.org/wiki/Early_stopping). 
The library includes hyperparameter optimization facilities for sampling and running 
experiments from the command-line or prior hyper-parameter distributions.

We are currently working on providing facilities for distributing the models over different machines.

Finally, we optionally provide facilites for storing and analysing hyperpameters and results using
a PostgreSQL database backend which facilitates distributing experiments over different machines. 

<a name="dp.tutorials"/>
## Tutorials and Examples ##
In order to help you get up and running we provide a quick [neural network tutorial](doc/neuralnetworktutorial.md) which explains step-by-step the contents of this [example script](examples/neuralnetwork_tutorial.lua). For a more flexible option that allows input from the command-line specifying different datasources and preprocesses, using dropout, running the code on a GPU/CPU, please consult this [script](examples/neuralnetwork.lua).

<a name="dp.packages"/>
## dp Packages ##
	
  * Data Library
    * [Data](doc/data.md) defines objects used for loading data.
    * [Preprocess](doc/preprocess.md) defines objects used for preprocessing data.
  * Model Library
    * Model defines objects encapsulating nn.Modules (doc TODO).
    * Container defines objects encapsulating Models (doc TODO).
  * Experiment Library
    * Propagator defines objects used to propagate (forward/backward) DataSets through models (doc TODO).
  * Hyperparameter Library
    * Hyperoptimizer (doc TODO)
    * DatasourceFactory (doc TODO)
    * ExperimentFactory (doc TODO)


<a name="dp.install"/>
## Install ##
To use this library, we will require some lua rocks:
```shell
$> sudo luarocks install fs
$> sudo luarocks install moses
$> sudo luarocks install nnx
```
We require the latest version of the `optim` package:
```
$> git clone https://github.com/torch/optim.git
$> cd optim
$> sudo luarocks make optim-1.0.4-0.rockspec 
```
Optional:
```shell
$> sudo apt-get install libpq-dev
$> sudo luarocks install luasql-postgres PGSQL_INCDIR=/usr/include/postgresql
$> sudo apt-get install liblapack-dev
```

If you are encountering problems related to BLAS, please refer to Torch7's [BLAS and LAPACK installation manual] (https://github.com/torch/torch7-distro/blob/master/pkg/dok/dokinstall/blas.dok)
