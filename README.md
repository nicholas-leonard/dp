# dp Package Reference Manual#

__dp__ is a <b>d</b>ee<b>p</b> learning library designed for streamlining 
research and development using the [Torch7](http://torch.ch) distribution. 
It emphasizes flexibility through the elegant use of object-oriented 
[design patterns](http://en.wikipedia.org/wiki/Design_Patterns).

During my time in the LISA Lab as an apprentice of Yoshua Bengio and Aaron Courville,
I was inspired by pylearn2 and Theano to build a framework better suited to 
my needs and style.

Among other things, this package includes : 
 * common datasets like MNIST, CIFAR-10 and CIFAR-100, reprocessing like Zero-Component Analysis whitening, Global Contrast Normalization, Lecun's Local Contrast Normalization, and facilities for interfacing your own.
 * a high-level framework that abstracts away common usage patterns of the [nn](https://github.com/torch/nn/blob/master/README.md) and [torch7](https://github.com/torch/torch7/blob/master/README.md) package such as loading datasets and [early stopping](http://en.wikipedia.org/wiki/Early_stopping). 
 * hyperparameter optimization facilities for sampling and running experiments from the command-line or prior hyper-parameter distributions.
 * facilites for storing and analysing hyperpameters and results using a PostgreSQL database backend which facilitates distributing experiments over different machines.

<a name="dp.tutorials"/>
## Tutorials and Examples ##
In order to help you get up and running we provide a quick [neural network tutorial](doc/neuralnetworktutorial.md) which explains step-by-step the contents of this [example script](examples/neuralnetwork_tutorial.lua). For a more flexible option that allows input from the command-line specifying different datasources and preprocesses, using dropout, running the code on a GPU/CPU, please consult this [script](examples/neuralnetwork.lua).

A [Facial Keypoints tutorial](doc/facialkeypointstutorial.md) involving the case study of a Kaggle Challenge is also available. It provides an overview of the steps required for extending and using  __dp__ in the context of the challenge. And even provides the script so that you can generate your own Kaggle submissions.

<a name="dp.packages"/>
## dp Packages ##
	
  * Data Library
    * [View](doc/view.md) : Tensor containers like [DataView](doc/view.md#dp.DataView), [ImageView](doc/view.md#dp.ImageView) and [ClassView](doc/view.md#dp.ClassView);
    * [Data](doc/data.md) : View containers like like [Batch](doc/data.md#dp.Batch), and [DataSources](doc/data.md#dp.DataSource) like [Mnist](doc/data.md#dp.Mnist) and [BillionWords](doc/data.md#dp.BillionWords);
    * [Preprocess](doc/preprocess.md) : data preprocessing like [ZCA](doc/preprocess.md#dp.ZCA) and [Standardize](doc/preprocess.md#dp.Standardize);
  * Node Library
    * [Node](doc/node.md) : abstract class that defines Model and Loss commonalities;
    * [Model](doc/model.md) : parameterized Nodes like [Neural](doc/model.md#dp.Neural) and [Convolution2D](doc/model.md#dp.Convolution2D) that adapt [Modules](https://github.com/torch/nn/blob/master/doc/module.md#module) to [Model](doc/model.md#dp.Model);
    * [Loss](doc/loss.md) : non-parameterized Nodes like [NLL](doc/loss.md#dp.NLL) that adapt [Criterions](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.Criterion);
  * Experiment Library
    * [Experiment](doc/experiment.md) : trains a Model using a DataSource and a Loss;
    * [Propagator](doc/propagator.md) : propagates a DataSet through a Model and Loss;
    * [Visitor](doc/visitor.md) : visits Models after a backward pass to update parameters, statistics or gradients;
  * Extension Library
    * [Feedback](doc/feedback.md) : provides I/O feedback given the Model output, input and targets;
    * [Observer](doc/observer.md) : plugins that can be appended to objects as extensions;
    * [Mediator](doc/mediator.md) : singleton to which objects can publish and subscribe Channels;
  * Hyperparameter Library
    * Hyperoptimizer : explores different experiment configurations;
    * DatasourceFactory : builds a datasource;
    * ExperimentFactory : builds an experiment and Model.


<a name="dp.install"/>
## Install ##
To use this library, install it globally via luarocks:
```shell
$> sudo luarocks install dp
```
or install it locally:
```shell
$> luarocks install dp --local
```
or clone and make it:
```shell
$> git clone git@github.com:nicholas-leonard/dp.git
$> cd dp
$> sudo luarocks make dp-scm-1.rockspec 
```

### Optional Dependencies ###
For CUDA:
```shell
$> sudo luarocks install cunnx
```
For PostgresSQL
```shell
$> sudo apt-get install libpq-dev
$> sudo luarocks install luasql-postgres PGSQL_INCDIR=/usr/include/postgresql
$> sudo apt-get install liblapack-dev
```

## Contributions ##

We appreciate [issues](https://github.com/nicholas-leonard/dp/issues) and [pull requests](https://github.com/nicholas-leonard/dp/pulls?q=is%3Apr+is%3Aclosed) of all kind.
