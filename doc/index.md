# dp Package Reference Manual#

[![Join the chat at https://gitter.im/nicholas-leonard/dp](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/nicholas-leonard/dp?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

__dp__ is a <b>d</b>ee<b>p</b> learning library designed for streamlining 
research and development using the [Torch7](http://torch.ch) distribution. 
It emphasizes flexibility through the elegant use of object-oriented 
[design patterns](http://en.wikipedia.org/wiki/Design_Patterns).

During my time in the LISA Lab as an apprentice of Yoshua Bengio and Aaron Courville,
I was inspired by pylearn2 and Theano to build a framework better suited to 
my needs and style. I have been using it since November 2013, and it has 
recently undergone major refactoring to allow for a more seemless integration 
with [nn](https://github.com/torch/nn/blob/master/README.md) through 
[dpnn](https://github.com/nicholas-leonard/dpnn/blob/master/README.md).

Among other things, this package includes : 

  * common datasets like MNIST, CIFAR-10 and CIFAR-100, preprocessing like Zero-Component Analysis whitening, Global Contrast Normalization, Lecun's Local Contrast Normalization, and facilities for interfacing your own.
  * a framework that abstracts away common usage patterns of the [nn](https://github.com/torch/nn/blob/master/README.md) and [torch7](https://github.com/torch/torch7/blob/master/README.md) package such as loading datasets and [early stopping](http://en.wikipedia.org/wiki/Early_stopping). 
  * documentation, examples and tutorials;
  
<a name="dp.tutorials"/>
[]()
## Tutorials and Examples ##
In order to help you get up and running we provide a quick [neural network tutorial](neuralnetworktutorial.md) which explains step-by-step the contents of this [example script](https://github.com/nicholas-leonard/dp/blob/master/examples/neuralnetwork.lua). For a more flexible option that allows input from the command-line specifying different datasources and preprocesses, using dropout, running the code on a GPU/CPU, please consult this [script](https://github.com/nicholas-leonard/dp/blob/master/examples/neuralnetwork.lua).

A [Facial Keypoints tutorial](facialkeypointstutorial.md) involving the case study of a Kaggle Challenge is also available. It provides an overview of the steps required for extending and using  __dp__ in the context of the challenge. And even provides the script so that you can generate your own Kaggle submissions.

The [Language Model tutorial](languagemodeltutorial.md) examines the implementation of a neural network language model trained on the Billion Words dataset.

In the [Data tutorial](datatutorial.md), we explore different ways to wrap your own data into a DataSource so that it may be used with dp.

A brief (1 hours) overview of Torch7, which includes some details about __dp__, is available via this [NVIDIA GTC Webinar video](http://on-demand.gputechconf.com/gtc/2015/webinar/torch7-applied-deep-learning-for-vision-natural-language.mp4). In any case, this presentation gives a nice overview of Logistic Regression, Multi-Layer Perceptrons, Convolutional Neural Networks and Recurrent Neural Networks using Torch7.

<a name="dp.packages"/>
[]()
## dp Packages ##
	
  * Data Library
    * [View](view.md) : Tensor containers like [DataView](view.md#dp.DataView), [ImageView](view.md#dp.ImageView) and [ClassView](view.md#dp.ClassView);
    * [BaseSet](data.md#dp.BaseSet) : View containers like [Batch](data.md#dp.Batch) and [DataSet](data.md#dp.DataSet);
    * [DataSource](data.md#dp.DataSource) : BaseSet containers like [Mnist](data.md#dp.Mnist) and [BillionWords](data.md#dp.BillionWords);
    * [Preprocess](preprocess.md) : data preprocessing like [ZCA](preprocess.md#dp.ZCA) and [Standardize](preprocess.md#dp.Standardize);
    * [Sampler](data.md#dp.Sampler) : DataSet iterators like [ShuffleSampler](data.md#dp.ShuffleSampler) and [SentenceSampler](data.md#dp.SentenceSampler);
  * Experiment Library
    * [Experiment](experiment.md) : trains a Model using a DataSource and a Loss;
    * [Propagator](propagator.md) : propagates a DataSet through a Model and Loss;
  * Extension Library
    * [Feedback](feedback.md) : provides I/O feedback given the Model output, input and targets;
    * [Observer](observer.md) : plugins that can be appended to objects as extensions;
    * [Mediator](mediator.md) : singleton to which objects can publish and subscribe Channels;


<a name="dp.install"/>
[]()
## Install ##
To use this library, install it via luarocks:
```shell
$> luarocks install dp
```
or clone and make it (recommended):
```shell
$> git clone git@github.com:nicholas-leonard/dp.git
$> cd dp
$> luarocks make rocks/dp-scm-1.rockspec 
```

### Optional Dependencies ###

For CUDA:
```shell
$> sudo luarocks install cunnx
```

## Other packages 

These are some related package :
 
  * [rnn](https://github.com/Element-Research/rnn) : Recurrent neural network library ;
  * [hypero](https://github.com/Element-Research/hypero) : Simple distributed hyper-optimization library.

## Contributions ##

We appreciate [issues](https://github.com/nicholas-leonard/dp/issues) and [pull requests](https://github.com/nicholas-leonard/dp/pulls?q=is%3Apr+is%3Aclosed) of all kind.
