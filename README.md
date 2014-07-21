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

Finally, we optionally provide facilites for storing and analysing hyperpameters and results using
a PostgreSQL database backend which facilitates distributing experiments over different machines. 

<a name="dp.tutorials"/>
## Tutorials and Examples ##
In order to help you get up and running we provide a quick [neural network tutorial](doc/neuralnetworktutorial.md) which explains step-by-step the contents of this [example script](examples/neuralnetwork_tutorial.lua). For a more flexible option that allows input from the command-line specifying different datasources and preprocesses, using dropout, running the code on a GPU/CPU, please consult this [script](examples/neuralnetwork.lua).

<a name="dp.packages"/>
## dp Packages ##
	
  * Data Library
    * [View](doc/view.md) : Tensor containers like [DataView](doc/view.md#dp.DataView), [ImageView](doc/view.md#dp.ImageView) and [ClassView](doc/view.md#dp.ClassView);
    * [Data](doc/data.md) : View containers like like [Batch](doc/data.md#dp.Batch) and [DataSet](doc/data.md#dp.DataSet), and [DataSources](doc/data.md#dp.DataSource) like [Mnist](doc/data.md#dp.Mnist) and [BillionWords](doc/data.md#dp.BillionWords);
    * [Preprocess](doc/preprocess.md) : data preprocessing like [ZCA](doc/preprocess.md#dp.ZCA) and [Standardize](doc/preprocess.md#dp.Standardize);
  * Node Library
    * [Node](doc/node.md) : abstract class that defines Model and Loss commonalities;
    * [Model](doc/model.md) : parameterized Nodes like [Neural](doc/model.md#dp.Neural) and [Convolution2D](doc/model.md#dp.Convolution2d) that adapt [Modules](https://github.com/torch/nn/blob/master/doc/module.md#module) to [Model](doc/model.md#dp.Model);
    * [Loss](doc/loss.md) : non-parameterized Nodes like [NLL](doc/loss.md#dp.NLL) that adapt [Criterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.Criterion) to [Loss](doc/loss.md#dp.Loss);
  * Experiment Library
    * Experiment : trains a Model using a DataSource and a Loss;
    * Propagator : [Propagators](propagator/propagator.lua) that forward DataSets and backpropagates a Loss through a Model;
    * Visitor : visits Models after a backward pass to update parameters, statistics or gradients;
  * Hyperparameter Library
    * Hyperoptimizer : explores different experiment configurations;
    * DatasourceFactory : builds a datasource;
    * ExperimentFactory : builds an experiment and Model.


<a name="dp.install"/>
## Install ##
To use this library, clone and make it:
```shell
$> git clone git@github.com:nicholas-leonard/dp.git
$> cd dp
$> sudo luarocks make dp-scm-1.rockspec 
```

Optional:
```shell
$> sudo apt-get install libpq-dev
$> sudo luarocks install luasql-postgres PGSQL_INCDIR=/usr/include/postgresql
$> sudo apt-get install liblapack-dev
```
