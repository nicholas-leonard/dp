# Data #

  * [BaseSet](#dp.BaseSet)
     * [DataSet](#dp.DataSet) :
      * [SentenceSet](#dp.SentenceSet)
     * [Batch](#dp.Batch)
  * [DataSource](#dp.DataSource) :
    * [Mnist](#dp.Mnist)
    * [NotMnist](#dp.NotMnist)
    * [Cifar10](#dp.Cifar10)
    * [Cifar100](#dp.Cifar100)
    * [BillionWords](#dp.BillionWords)
  * [Sampler](#dp.Sampler) :
    * [ShuffleSampler](#dp.ShuffleSampler) 

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
A subclass of [BaseSet](#dp.BaseSet). 

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
 
