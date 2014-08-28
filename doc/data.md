# Elementary, Dear Data #
One of the most important aspects of any machine learning problem is the data. The _dp_ library provides the following data-related facilities:
  * [BaseSet](#dp.BaseSet) : abstract class;
     * [DataSet](#dp.DataSet) : a dataset for input and target [Views](view.md#dp.View);
      * [SentenceSet](#dp.SentenceSet) : container of sentences (used for language modeling);
     * [Batch](#dp.Batch) : a mini-batch of inputs and targets;
  * [DataSource](#dp.DataSource) : a container of train, valid and test DataSets;
    * [Mnist](#dp.Mnist) : the ubiquitous MNIST dataset;
    * [NotMnist](#dp.NotMnist) : the lesser known NotMNIST dataset;
    * [Cifar10](#dp.Cifar10) : the CIFAR-10 dataset;
    * [Cifar100](#dp.Cifar100) : the very difficult to generalize CIFAR-100 dataset;
    * [BillionWords](#dp.BillionWords) : the Google 1-Billion Words language model dataset;
    * [SVHN](#dp.SVHN) : the Google Street View House Numbers dataset;
  * [Sampler](#dp.Sampler) : dataset iterator;
    * [ShuffleSampler](#dp.ShuffleSampler) : shuffled dataset iterator;

<a name="dp.BaseSet"/>
## BaseSet ##
This is the base (abstract) class inherited by subclasses like [DataSet](#dp.DataSet),
[SentenceSet](#dp.SentenceSet) and [Batch](#dp.Batch). It is used for training or evaluating a model. 
It supports multiple-input and multiple-output datasets using [ListView](#dp.ListView).
In the case of multiple targets, it is useful for multi-task learning, 
or learning from hints . In the case of multiple inputs, richer inputs representations could 
be created allowing, for example, images to be combined with 
tags, text with images, etc. Multi-input/target facilities could be used with nn.ParallelTable and 
nn.ConcatTable. If the BaseSet is used for unsupervised learning, only inputs need to be provided.

<a name='dp.BaseSet.__init'/>
### dp.BaseSet{inputs, [targets, which_set]} ###
Constructs a dataset from inputs and targets.
Arguments should be specified as key-value pairs. 
 * `inputs` is an instance of [View](#dp.View) or a table of these. In the latter case, they will be automatically encapsulated by a [ListView](#dp.ListView). These are used as inputs to a [Model](model.md#dp.Model).
 * `targets` is an instance of `View` or a table of these. In the latter case, they will be automatically encapsulated by a `ListView`. These are used as targets for training a `Model`. The indices of examples in `targets` must be aligned with those in `inputs`. 
 * `which_set` is a string identifying the purpose of the dataset. Valid values are 
   * *train* for training, i.e. for fitting a model to a dataset; 
   * *valid* for cross-validation, i.e. for early-stopping and hyper-optimization; 
   * *test* for testing, i.e. comparing your model to the current state-of-the-art and such.

<a name="dp.BaseSet.preprocess"/>
### preprocess([input_preprocess, target_preprocess, can_fit]) ###
Preprocesses the BaseSet.
 * `input_preprocess` is [Preprocess](preprocess.md#dp.Preprocess) to be applied to the input [View](view.md#dp.View) of the [BaseSet](data.md#dp.BaseSet).
 * `target_preprocess` is [Preprocess](preprocess.md#dp.Preprocess) to be applied to the target [View](view.md#dp.View) of the [BaseSet](data.md#dp.BaseSet).
 * `can_fit` is a boolean. When true, allows measuring of statistics on the View of BaseSet to initialize the Preprocess. Should normally only be done on the training set. Default is to fit the training set.

<a name="dp.BaseSet.inputs"/>
### [inputs] inputs() ###
Returns inputs [View](view.md#dp.View).

<a name="dp.BaseSet.targets"/>
### [targets] targets() ###
Returns targets [View](view.md#dp.View).

<a name="dp.DataSet"/>
## DataSet ##
A subclass of [BaseSet](#dp.BaseSet). Contains input and optional target [Views](view.md#dp.View) used for training or evaluating [Models](model.md#dp.Model).

<a name='dp.DataSet.batch'/>
### batch(batch_size) ###
A factory method that builds a [Batch](#dp.Batch) of size `batch_size`. It effectively 
calls [sub](#dp.DataSet.sub) with arguments `start=1` and `stop=batch_size`. This method 
reuses the DataSet's inputs and targets, such that these shouldn't be modified, unless the 
intent is to modify the original DataSet.

<a name='dp.DataSet.sub'/>
### sub(start, stop, [new]) ###
A factory method that builds a [Batch](#dp.Batch) by calling [sub](view.md#dp.View.sub) 
with argument `start` and `stop` on the DataSet's inputs and targets.
This method reuses the DataSet's inputs and targets, such that these shouldn't be modified, unless the 
intent is to modify the original DataSet.

<a name='dp.DataSet.index'/>
### index([batch,] indices) ###

<a name="dp.SentenceSet"/>
## SentenceSet ##
A subclass of [DataSet](#dp.DataSet) used for language modeling. 
Takes a sequence of words stored as a tensor of word IDs and a [Tensor](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor) 
holding the start index of the sentence of its commensurate word id (the one at the same index).
Unlike DataSets, for memory efficiency reasons, this class does not store its data in [Views](view.md#dp.View).
However, the outputs of factory methods [batch](#dp.DataSet.batch), [sub](#dp.DataSet.sub), and
[index](#dp.DataSet.index) are [Batches](#dp.Batch) containing input and target [ClassViews](view.md#dp.ClassView).
The returned [batch:inputs()](#dp.BaseSet.inputs) are filled according to [Google 1-Billion Words guidelines](https://code.google.com/p/1-billion-word-language-modeling-benchmark/source/browse/trunk/README.perplexity_and_such).

<a name="dp.Batch"/>
## Batch ##
A subclass of [BaseSet](#dp.BaseSet). A mini-batch of input and target [Views](view.md#dp.View) 
to be fed into a [Model](model.md#dp.Model) and [Loss](loss.md#dp.Loss). The batch of examples is usually sampled 
from a [DataSet](#dp.DataSet) via a [Sampler](#dp.Sampler) iterator by calling the DataSet's different factory methods : [batch](#dp.DataSet.batch), [sub](#dp.DataSet.sub), and [index](#dp.DataSet.index). A batch is also the original generator of the `carry` table passed through the computation graph using a propagation.

<a name="dp.DataSource"/>
## DataSource ##
Abstract class used to generate up to 3 [DataSets](#dp.DataSet) : *train*, *valid* and *test*:
 * *train* for training, i.e. for fitting a model to a dataset; 
 * *valid* for cross-validation, i.e. for early-stopping and hyper-optimization; 
 * *test* for testing, i.e. comparing your model to the current state-of-the-art and such.
It can also perform preprocessing using [Preprocess](preprocess.md#dp.Preprocess) on all DataSets by fitting only the training set.

### dp.DataSource{...} ###
DataSource constructor. Arguments should be specified as key-value pairs. 
 * `train_set` is an optional [DataSet](#dp.DataSet) used for training, i.e. optimizing a [Model](model.md#dp.Model) to minimize a [Loss](loss.md#dp.Loss)
 * `valid_set` is an optional DataSet used for cross-validation, i.e. for early-stopping and hyper-optimization
 * `test_set` is an optional DataSet used to evaluate generalization performance after training (e.g. to compare different models)
 * `input_preprocess` is a [Preprocess](preprocess.md#dp.Preprocess) that will be applied to the inputs. Statistics are measured (fitted) on the `train_set` only, and then reused to preprocess all provided sets. This argument may also be provided as a list (table) of Preprocesses, in which case, they will be wrapped in the composite [Pipeline](preprocess.md#dp.Pipeline) Preprocess.
 * `target_preprocess` is like `input_preprocess`, but for preprocessing the targets.

At least one of the 3 `set` arguments should be specified.

### preprocess() ###
If they exist, applies the `input_preprocess` and `target_preprocess` [Preprocess](preprocess.md#dp.Preprocess) 
attributed specified in the constructor or via the `set[Input,Target]Preprocess` methods 
to the inputs and targets, respectively.  Statistics are measured (fitted) on the `train_set` only, 
and then reused to preprocess all contained [DataSets](#dp.DataSet). This method is invoked by the constructor.

### [path] getDataPath{name, url, data_dir, decompress_file} ###
A static function (not to be called via method operator `:`) that 
looks for a file `data_dir/name/decompress_file`, and, if it is missing, downloads it. 
Returns the `path` to the resulting data file. 
 * `name` is a string specifying the name of the DataSource (e.g. "Mnist", "BillionWords", etc). Also the name of the directory where the file should be located. A directory with this name is created within `data_directory` to contain the downloaded files. Or is expected to find the data files in this directory.
 * `url` is a string specifying the URL from which data can be downloaded in case it is not found in the path.
 * `data_dir` is a string specifying the path to the directory containing directory `name`, which is expected to contain the data, or where it will be downloaded.
 * `decompress_file` is a string that when non-nil, decompresses the downloaded data if `data_dir/name/decompress_file` is not found. In which case, returns `data_dir/name/decompress_file`.
 
<a name="dp.Mnist"/>
## Mnist ##
A [DataSource](#dp.DataSource) subclass wrapping the simple but widely used handwritten digits 
classification problem (see [MNIST](http://yann.lecun.com/exdb/mnist/)). The images are of size `28x28x1`. The classes are : 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.

<a name="dp.NotMnist"/>
## NotMnist ##
A [DataSource](#dp.DataSource) subclass wrapping the much larger alternative to MNIST: [NotMNIST](http://yaroslavvb.blogspot.ca/2011/09/notmnist-dataset.html). 
If not found on the local machine, the object downloads the dataset from the 
[original source](http://yaroslavvb.com/upload/notMNIST/). 
It contains 500k+ examples of 10 charaters using unicode fonts: *A*,*B*,*C*,*D*,*E*,*F*,*G*,*H*,*I*,*J*. Like [Mnist](#dp.Mnist), the images are of size `28x28x1`.

<a name="dp.Cifar10"/>
## Cifar10 ##
A [DataSource](#dp.DataSource) subclass wrapping the [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset. 
It is a `3x32x32` color-image set of 10 different objects. Small dataset size makes it hard to generalize from train to test set (Regime : overfitting).

<a name="dp.Cifar100"/>
## Cifar100 ##
A [DataSource](#dp.DataSource) subclass wrapping the [CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html) 
dataset. It is a `3x32x32` color-image set of 100 different objects. Small dataset (even less images 
per class than [Cifar-10](#dp.Cifar10)) size makes it hard to generalize from train to test set (Regime : overfitting). 

<a name="dp.BillionWords"/>
## BillionWords ##
A [DataSource](#dp.DataSource) subclass wrapping the corpus derived from the 
`training-monolingual.tokenized/news.20??.en.shuffled.tokenized` data distributed for [WMT11](http://statmt.org/wmt11/translation-task.html). The preprocessing suggested by 
the [Google 1-Billion Words language modeling benchmark](https://code.google.com/p/1-billion-word-language-modeling-benchmark) was used to prepare the data. 
The task consists in predicting the next word given the `n` previous ones, where `n` is the context size (it can be set in the constructor).
The data consists in approximately 30 million sentences of an average length of about 25 words.
In, there are about 800 thousand (unique) words in the vocabulary, which makes it a very memory intensive problem.
The DataSource inclues data for building hierarchical softmaxes to accelerate training.

<a name="dp.SVHN"/>
## SVHN ##
The Google Street View House Numbers dataset.

<a name="dp.Sampler"/>
## Sampler ##
A [DataSet](#dp.DataSet) iterator which qequentially samples [Batches](#dp.Batch) from a DataSet for a [Propagator](propagator.md#dp.Propagator).

<a name="dp.ShuffleSampler"/>
## ShuffleSampler ##
A subclass of [Sampler](#dp.Sampler) which iterates over [Batches](#dp.Batch) in a dataset 
by shuffling the example indices before each epoch.
