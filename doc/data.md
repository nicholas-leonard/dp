# Elementary, Dear Data #
One of the most important aspects of any machine learning problem is the data. The _dp_ library provides the following data-related facilities:
  
  * [BaseSet](#dp.BaseSet) : abstract class;
     * [DataSet](#dp.DataSet) : a dataset for input and target [Views](view.md#dp.View);
       * [SentenceSet](#dp.SentenceSet) : container of sentences (used for language modeling);
       * [TextSet](#dp.TextSet) : container of text (used for language modeling);
       * [ImageClassSet](#dp.ImageClassSet) : container for large-scale image-classification datasets;
     * [Batch](#dp.Batch) : a mini-batch of inputs and targets;
  * [DataSource](#dp.DataSource) : a container of train, valid and test DataSets;
    * [Mnist](#dp.Mnist) : the ubiquitous MNIST dataset;
    * [NotMnist](#dp.NotMnist) : the lesser known NotMNIST dataset;
    * [FaceDetection](#dp.FaceDetection) : the Purdue face detection dataset;
    * [Cifar10](#dp.Cifar10) : the CIFAR-10 dataset;
    * [Cifar100](#dp.Cifar100) : the very difficult to generalize CIFAR-100 dataset;
    * [Svhn](#dp.Svhn) : the Google Street View House Numbers dataset;
    * [ImageNet](#dp.ImageNet) : the Large Scale Visual Recognition Challenge 2014 (ILSVRC2014) dataset;
    * [PennTreeBank](#dp.PennTreeBank) : the Penn Tree Bank language model dataset;
    * [BillionWords](#dp.BillionWords) : the Google 1-Billion Words language model dataset;
    * [ImageSource](#dp.ImageSource) : generic large image classification data wrapper;
    * [SmallImageSource](#dp.SmallImageSource) : generic small image classification data wrapper;
    * [TextSource](#dp.TextSource) : generic text language model dataset wrapper;
  * [Sampler](#dp.Sampler) : ordered dataset iterator;
    * [ShuffleSampler](#dp.ShuffleSampler) : shuffled dataset iterator;
    * [SentenceSampler](#dp.SentenceSampler) : samples sentences for recurrent models;
    * [RandomSampler](#dp.RandomSampler) : iterates through batches of random examples;
    * [TextSampler](#dp.TextSampler) : iterates through a text (a single contiguous sequence of words) dataset;

<a name="dp.BaseSet"></a>
## BaseSet ##
This is the base (abstract) class inherited by subclasses like [DataSet](#dp.DataSet),
[SentenceSet](#dp.SentenceSet) and [Batch](#dp.Batch). It is used for training or evaluating a model. 
It supports multiple-input and multiple-output datasets using [ListView](#dp.ListView).
In the case of multiple targets, it is useful for multi-task learning, 
or learning from hints . In the case of multiple inputs, richer inputs representations could 
be created allowing, for example, images to be combined with 
tags, text with images, etc. Multi-input/target facilities could be used with nn.ParallelTable and 
nn.ConcatTable. If the BaseSet is used for unsupervised learning, only inputs need to be provided.

<a name='dp.BaseSet.__init'></a>
### dp.BaseSet{inputs, [targets, which_set]} ###
Constructs a dataset from inputs and targets.
Arguments should be specified as key-value pairs. 
 
  * `inputs` is an instance of [View](#dp.View) or a table of these. In the latter case, they will be automatically encapsulated by a [ListView](#dp.ListView). The encapsulated Tensor is used as inputs to a `model`.
  * `targets` is an instance of `View` or a table of these. In the latter case, they will be automatically encapsulated by a `ListView`. These are used as targets for training a `Model`. The indices of examples in `targets` must be aligned with those in `inputs`. 
  * `which_set` is a string identifying the purpose of the dataset. Valid values are 
    * *train* for training, i.e. for fitting a model to a dataset; 
    * *valid* for cross-validation, i.e. for early-stopping and hyper-optimization; 
    * *test* for testing, i.e. comparing your model to the current state-of-the-art and such.

<a name="dp.BaseSet.preprocess"></a>
### preprocess([input_preprocess, target_preprocess, can_fit]) ###
Preprocesses the BaseSet.
 
  * `input_preprocess` is [Preprocess](preprocess.md#dp.Preprocess) to be applied to the input [View](view.md#dp.View) of the [BaseSet](data.md#dp.BaseSet).
  * `target_preprocess` is [Preprocess](preprocess.md#dp.Preprocess) to be applied to the target [View](view.md#dp.View) of the [BaseSet](data.md#dp.BaseSet).
  * `can_fit` is a boolean. When true, allows measuring of statistics on the View of BaseSet to initialize the Preprocess. Should normally only be done on the training set. Default is to fit the training set.

<a name="dp.BaseSet.inputs"></a>
### [inputs] inputs() ###
Returns inputs [View](view.md#dp.View).

<a name="dp.BaseSet.targets"></a>
### [targets] targets() ###
Returns targets [View](view.md#dp.View).

<a name="dp.DataSet"></a>
## DataSet ##
A subclass of [BaseSet](#dp.BaseSet). Contains input and optional target [Views](view.md#dp.View) used for training or evaluating `models`.

<a name='dp.DataSet.batch'></a>
### batch(batch_size) ###
A factory method that builds a [Batch](#dp.Batch) of size `batch_size`. It effectively 
calls [sub](#dp.DataSet.sub) with arguments `start=1` and `stop=batch_size`. This method 
reuses the DataSet's inputs and targets, such that these shouldn't be modified, unless the 
intent is to modify the original DataSet.

<a name='dp.DataSet.sub'></a>
### sub(start, stop, [new]) ###
A factory method that builds a [Batch](#dp.Batch) by calling [sub](view.md#dp.View.sub) 
with argument `start` and `stop` on the DataSet's inputs and targets.
This method reuses the DataSet's inputs and targets, such that these shouldn't be modified, unless the 
intent is to modify the original DataSet.

<a name='dp.DataSet.index'></a>
### index([batch,] indices) ###

<a name="dp.SentenceSet"></a>
## SentenceSet ##
A [DataSet](#dp.DataSet) used for language modeling. 
Takes a sequence of words stored as a tensor of word IDs and a [Tensor](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor) 
holding the start index of the sentence of its commensurate word id (the one at the same index).
Unlike DataSets, for memory efficiency reasons, this class does not store its data in [Views](view.md#dp.View).
However, the outputs of factory methods [batch](#dp.DataSet.batch), [sub](#dp.DataSet.sub), and
[index](#dp.DataSet.index) are [Batches](#dp.Batch) containing input and target [ClassViews](view.md#dp.ClassView).
The returned [batch:inputs()](#dp.BaseSet.inputs) are filled according to [Google 1-Billion Words guidelines](https://code.google.com/p/1-billion-word-language-modeling-benchmark/source/browse/trunk/README.perplexity_and_such).

<a name='dp.TextSet'></a>
## TextSet ##
A [DataSet](#dp.DataSet) used for language modeling. 
Takes a sequence of words stored as a tensor of word ids.
Contrary to [SentenceSet](#dp.SentenceSet), this wrapper assumes a continuous stream of words. 
If consecutive sentences are completely unrelated, you might be better off using SentenceSet, 
unless your model can learn to forget (like [LSTMs](https://github.com/Element-Research/rnn#rnn.LSTM)). 
Like SentenceSet, this class does not store its data in [Views](view.md#dp.View).
However, the outputs of `batch()`, `sub()`, `index()` are [Batches](#dp.Batch)
containing [ClassViews](view.md#dp.ClassView) of inputs and targets.

<a name="dp.ImageClassSet"></a>
## ImageClassSet ##
A DataSet for image classification tasked stored in a flat folder structure :
```
[data_path]/[class]/[imagename].[JPEG,png,...] 
```
Optimized for extremely large datasets (14 million images+). 
This DataSet is very memory efficient in that the images are loaded from 
disk into memory only when requested as a [Batch](#dp.Batch). It is 
used to wrap the training and validation sets of the [ImageNet](#dp.ImageNet)
DataSource.

When first initialized, the dataset needs to build an index of all image paths which it encapsulates 
into torch.CharTensor for efficieny. The index is build using some heavy command-line
magic, but this only needs to be executed once as the resulting index is cached to disk
for the next time the dataset is used. 

During queries of the dataset using [sample](#dp.ImageClassSet.sample) or [sub](#dp.DataSet.sub), 
the index is used to retrieve images from disk. This can be a major bottleneck. 
We strongly encourage storing your dataset on a Solid-State Drive (SSD). Furthermore,
if [threads-ffi](https://github.com/torch/threads-ffi/blob/master/README.md) is installed, the dataset can be used for asynchronous batch requests.
This is implemented using [multi-threading](#dp.ImageClassSet.multithread), 
which is necessary to speed up reading all those files. 

<a name="dp.ImageClassSet.__init"></a>
### dp.ImageClassSet{...} ###
ImageClassSet constructor. Arguments should be specified as key-value pairs. 

  * `data_path` is a string (or table thereof) specifying one or many paths to the data.
  * `load_size` ia a table specifying the approximate size (`nChannel x Height x Width`) for which to load the images to, initially.
  * `sample_size` is a table specifying a consistent sample size to resize the images to (or crop them). Defaults to `load_size`.
  * `verbose` is a boolean specifying whether or not to display verbose messages. Defaults to true.
  * `sample_func` is a string or function `f(self, dst, path)` that fills the `dst` Tensor with one or many images taken from the image located at `imgpath` Strings "sampleDefault",  "sampleTrain" or "sampleTest" can also be provided as they refer to existing methods. Defaults to [sampleDefault](#dp.ImageClassSet.sampleDefault). 
  * `sort_func` is a comparison function used for sorting the class directories. The order is used to assign each class and index.  Defaults to the `<` operator.
  * `cache_mode` is a string with default value "writeonce". Valid options include:
   * "writeonce" : read from cache if exists, else write to cache.
   * "overwrite" : write to cache, regardless if exists.
   * "nocache" : dont read or write from cache.
   * "readonly" : only read from cache, fail otherwise.
  * `cache_path` is a string specifiying the path of a cache file. Defaults to `[data_path[1]]/cache.th7`.

The DataSet constructor arguments also apply.

<a name="dp.ImageClassSet.sample"></a>
### [batch] sample([batch,] nSample, [sampleFunc]) ###
For `nSample` examples, uniformly samples a class, and then uniformly samples example from that class.
This keeps the class distribution balanced. Argument `sampleFunc` is a 
function or string used for sampling patches from a loaded image
(see [constructor](#dp.ImageClassSet.__init) for details). 
Defaults to whatever was passed to the constructor. The  optional `batch` argument, a [Batch](#dp.Batch) instance,
is recommended for minimizing memory allocations (see [sub](#dp.DataSet.sub) for details).
This Batch factor is called by the [RandomSampler]

Note that depending on the `sampleFunc`, the number of returned samples may 
be greater than `nSample` (see [sampleTest](#dp.ImageClassSet.sampleTest) for an example).

<a name="dp.ImageClassSet.sampleDefault"></a>
### [dst] sampleDefault([dst,] path) ###
Loads the image located at `path`. The returned `dst` Tensor will have 
size `sample_size`.

<a name="dp.ImageClassSet.sampleTrain"></a>
### [dst] sampleTrain([dst,] path) ###
Loads the image of size `load_size` located at `path`. Does a random crop
of size `sample_size` from the loaded image and returns it as `dst`.

<a name="dp.ImageClassSet.sampleTest"></a>
### [dst] sampleTest([dst,] path) ###
Loads the image of size `load_size` located at `path`. 
Does 10 crops, (center + 4 corners) and their horizontal flips.
Works with the [TopCrop](feedback.md#dp.TopCrop) feedback.

<a name="dp.ImageClassSet.multithread"></a>
### multithread([nThread, queueSize]) ###
Uses [threads-ffi](https://github.com/torch/threads-ffi/blob/master/README.md) to spawn a
[Threads](https://github.com/torch/threads-ffi/blob/master/README.md#threads.main) 
pool of `nThread` threads communicating with the current main thread through a
[queue](https://github.com/torch/threads-ffi/blob/master/README.md#worker) of size
`queueSize`. Each thread will load an ImageClassSet instance using the 
image path index cached on disk. 

A [Sampler](#dp.Sampler) or [RandomSampler](#dp.RandomSampler) can then 
be set in [async](#dp.Sampler.async) mode to query the threads for [Batches](#dp.Batch)
asynchronously. To do this, the Samplers begin by sending `queueSize` batch requests 
to the Thread pool. After that, for each batch [requested](#dp.ImageClassSet.asyncGet) from the 
dataset, another request is sent to the pool using either [subAsyncPut](#dp.ImageClassSet.subAsyncPut)
or [sampleAsyncPut](#dp.ImageClassSet.sampleAsyncPut).

<a name="dp.ImageClassSet.subAsyncPut"></a>
### subAsyncPut(batch, start, stop, callback) ###
Puts a [sub](#dp.DataSet.sub) request onto the queue such that it will be executed 
by one of the threads in the Threads pool. The `callback` is 
a function that will be executed on the resulting Batch once it returns to 
the main thread.
Must be preceeded by a call to [multithread](#dp.ImageClassSet.multithread).

<a name="dp.ImageClassSet.sampleAsyncPut"></a>
### sampleAsyncPut(batch, nSample, sampleFunc, callback) ###
Same as [subAyncPut](#dp.ImageClassSet.subAsyncPut), but for sending a
[sample](#dp.ImageClassSet.sample) request.

<a name="dp.ImageClassSet.asyncGet"></a>
### [batch] asyncGet() ###
Retrives a Batch request from the queue and returns it to the caller. 
The call must be preceded by a call to [subAyncPut](#dp.ImageClassSet.subAsyncPut)
or [sampleAyncPut](#dp.ImageClassSet.sampleAsyncPut)

<a name="dp.ImageClassSet.synchronize"></a>
### synchronize() ###
Empties the queue of asynchronous requests.

<a name="dp.Batch"></a>
## Batch ##
A subclass of [BaseSet](#dp.BaseSet). A mini-batch of input and target [Views](view.md#dp.View).
The encapsulated Tensors are to be fed into a Module and Criterion. 
The batch of examples is usually sampled from a [DataSet](#dp.DataSet) 
via a [Sampler](#dp.Sampler) iterator by calling the DataSet's different factory methods : 
[batch](#dp.DataSet.batch), [sub](#dp.DataSet.sub), and [index](#dp.DataSet.index).

<a name="dp.DataSource"></a>
## DataSource ##
Abstract class used to generate up to 3 [DataSets](#dp.DataSet) : *train*, *valid* and *test*:
 
  * *train* for training, i.e. for fitting a model to a dataset; 
  * *valid* for cross-validation, i.e. for early-stopping and hyper-optimization; 
  * *test* for testing, i.e. comparing your model to the current state-of-the-art and such.
It can also perform preprocessing using [Preprocess](preprocess.md#dp.Preprocess) on all DataSets by fitting only the training set.

<a name="dp.DataSource.__init"></a>
### dp.DataSource{...} ###
DataSource constructor. Arguments should be specified as key-value pairs. 
 
  * `train_set` is an optional [DataSet](#dp.DataSet) used for training, i.e. optimizing a `model` to minimize a `loss`
  * `valid_set` is an optional DataSet used for cross-validation, i.e. for early-stopping and hyper-optimization
  * `test_set` is an optional DataSet used to evaluate generalization performance after training (e.g. to compare different models)
  * `input_preprocess` is a [Preprocess](preprocess.md#dp.Preprocess) that will be applied to the inputs. Statistics are measured (fitted) on the `train_set` only, and then reused to preprocess all provided sets. This argument may also be provided as a list (table) of Preprocesses, in which case, they will be wrapped in the composite [Pipeline](preprocess.md#dp.Pipeline) Preprocess.
  * `target_preprocess` is like `input_preprocess`, but for preprocessing the targets.

Note that at least one of the 3 `set` arguments should be specified. If you need guidance to build your own DataSource, the [Facial Keypoint Tutorial](facialkeypointstutorial.md#facial-keypoints-tutorial) also includes a [section](facialkeypointstutorial.md#facialkeypoints) demonstrating how a DataSource can be built to wrap facial keypoint detection data.

<a name = "dp.DataSource.get"></a>
### [tensor, dataview, dataset] get(which_set, attribute, view, type) ###
This method simplifies access to tensors. This is best demonstrated with an example. 
Say you want to access the input tensor of the training set, you can call :
```lua
tensor = ds:trainSet():inputs():forward('default')
```
That is a lot of function calls. You can use the `get` method instead:
```lua
tensor = ds:get('train', 'input', 'default')
```
These are also the default arguments, so the above are equivalent to :
```lua
tensor = ds:get()
```

All arguments are optional strings :
 * `which_set` specifies which DataSet : *train*, *valid* or *test*. Defaults to *train*;
 * `attribute` specifies which attribute of the DataSet : *input* or *target*. Defaults to *inputs*;
 * `view` specifies the axis order of the tensor to get : *bwc*, *bchw*, *b*, etc. Defaults to *default*. See [Views](#dp.View);
 * `type` specifies the type of the Tensor to get : *float*, *torch.FloatTensor*, *Float*, *cuda*, etc. 

<a name = "dp.DataSource.set"></a>
### [dataview, dataset] set(which_set, attribute, view, tensor) ###
This method allows for setting one of the encapsulated Tensors.
Say you want to set the input tensor of the training set, you can call :
```lua
ds:trainSet():inputs():forward('bf', torch.randn(3,4))
```
That is a lot of function calls. You can use the `set` method instead:
```lua
ds:set('train', 'input', 'bf', torch.randn(3,4))
```

All arguments are mandatory :
 * `which_set` specifies which DataSet : *train*, *valid* or *test*;
 * `attribute` specifies which attribute of the DataSet : *input* or *target*;
 * `view` specifies the axis order of the tensor to be encapsulated in a [Views](#dp.View) : *bwc*, *bchw*, *b*, etc;
 * `tensor` is the Tensor that you want to encapsulate.

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
 
<a name="dp.TextSource"></a>
## TextSource ##
A generic text language model dataset wrapper. 
Creates a DataSource out of 1 to 3 strings or text files.
Text files are assumed to be arranged one sentence per line, each
line beginning with a space and ending with a space and a newline.
 
<a name="dp.ImageSource"></a>
## ImageSource ##
A generic version of the [ImageNet](#dp.ImageNet) DataSource used for wrapping
large image classification datasets. Each class is a directory of image files.
The images can be asynchronously loaded into memory in one ore many different threads 
to speedup training.

<a name="dp.SmallImageSource"></a>
## SmallImageSource ##
A generic DataSource used for wrapping small image classification datasets.
Each class is a directory of image files. The loaded input (images) and target (classes) 
tensors are cached to disk the first time the constuctor is called. 
The next time it is called, these cached files will be loaded instead of the directory of images,
thereby speeding up the loading process.

<a name="dp.Mnist"></a>
## Mnist ##
A [DataSource](#dp.DataSource) subclass wrapping the simple but widely used handwritten digits 
classification problem (see [MNIST](http://yann.lecun.com/exdb/mnist/)). The images are of size `28x28x1`. The classes are : 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.

<a name="dp.NotMnist"></a>
## NotMnist ##
A [DataSource](#dp.DataSource) subclass wrapping the much larger alternative to MNIST: 
[NotMNIST](http://yaroslavvb.blogspot.ca/2011/09/notmnist-dataset.html). 
If not found on the local machine, the object downloads the dataset from the 
[original source](http://yaroslavvb.com/upload/notMNIST/). 
It contains 500k+ examples of 10 charaters using unicode fonts: *A*,*B*,*C*,*D*,*E*,*F*,*G*,*H*,*I*,*J*. 
Like [Mnist](#dp.Mnist), the images are of size `28x28x1`.

<a name="dp.Cifar10"></a>
## Cifar10 ##
A [DataSource](#dp.DataSource) subclass wrapping the [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset. 
It is a `3x32x32` color-image set of 10 different objects. Small dataset size makes it hard to generalize 
from train to test set (Regime : overfitting).

<a name="dp.Cifar100"></a>
## Cifar100 ##
A [DataSource](#dp.DataSource) subclass wrapping the [CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html) 
dataset. It is a `3x32x32` color-image set of 100 different objects. Small dataset (even less images 
per class than [Cifar-10](#dp.Cifar10)) size makes it hard to generalize from train to test set (Regime : overfitting). 

<a name="dp.Svhn"></a>
## Svhn ##
The Google Street View House Numbers (SVHN) DataSource wraps 
the [originalsource](http://ufldl.stanford.edu/housenumbers/). 
If not found on the local machine, the object downloads the dataset from 
[nikopia.com](http://www.nikopia.com). 
It contains 73257 digits for training, 26032 digits for testing, and 531131 additional, 
somewhat less difficult samples, to use as extra training data. 
Like [CIFAR](#dp.Cifar10), the images are of size `3x32x32`.

<a name="dp.FaceDetection"></a>
## FaceDetection ##
A Purdue face detection dataset used in the [data tutorial](datatutorial.md).

<a name="dp.ImageNet"></a>
## ImageNet ##
Ref.: A. http://image-net.org/challenges/LSVRC/2014/download-images-5jj5.php

This DataSource wraps the Large Scale Visual Recognition Challenge 2014 (ILSVRC2014)
image classification dataset (commonly known as ImageNet). 
The dataset hasn't changed from 2012-2014.

### Requirements ###

Due to its size, the data first needs to be prepared offline.
Use [downloadimagenet.lua](https://github.com/nicholas-leonard/dp/tree/master/scripts/downloadimagenet.lua) 
to download and extract the data :
```bash
th downloadimagenet.lua --savePath '/path/to/diskspace/ImageNet'
```
The entire process requires about 360 GB of disk space to complete the download and extraction process.
This can be reduced to about 150GB if the training set is downloaded and extracted first, 
and all the `.tar` files are manually deleted. Repeat for the validation set, devkit and metadata. 
If you still don't have enough space in one partition, you can divide the data among different partitions.
We recommend a good internet connection (>60Mbs download) and a good Solid-State Drives (SSD).

Use [harmonizeimagenet.lua](https://github.com/nicholas-leonard/dp/tree/master/scripts/harmonizeimagenet.lua) 
to harmonize the train and validation sets:
```bash
th scripts/harmonizeimagenet.lua --dataPath /path/to/diskspace/ImageNet --progress --forReal
```
The sets will then contain a directory of images for each class with name `class[id]`
where `[id]` is a class index, between 1 and 1000, used for the ILVRC2014 competition.

Then we need to install [graphicsmagick](https://github.com/clementfarabet/graphicsmagick/blob/master/README.md) 
and [torchx](https://github.com/nicholas-leonard/torchx):
```bash
sudo luarocks install graphicsmagick
sudo luarocks install torchx
```

### Memory Efficiency ###

Unlike most DataSources, ImageNet doesn't read all images into memory when it is first loaded.
Instead it uses [ImageClassSet](#dp.ImageClassSet) to encapslate the different datasets; in this case 
the `train` and `valid` sets. The ImageClassSet builts a list of all images and indexes them per class. 
In this way, each [Batch](#dp.Batch) is only loaded from disk and created when requested from a 
[Sampler](#dp.Sampler), making it very memory efficient. This is also the reason why we recommend 
storing the dataset on SSD.

### Sampling ###
As in the famous [(Krizhevsky et al. 2012)](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0CCMQFjAA&url=http%3A%2F%2Fwww.cs.toronto.edu%2F~fritz%2Fabsps%2Fimagenet.pdf&ei=k1j7VIOpNoyuggTbq4SQAQ&usg=AFQjCNGDafONr3DDGBbtw5AL9B_R8AeTCg)
paper, the ImageNet training dataset samples images cropped from random 
224x224 patches from the images resizes so that the smallest dimension has 
size 256. As for the validation set, ten 224x224 patches are cropped per image,
i.e. center, four corners and their horizontal flips, and their predictions are averaged. 

### Example ###
You can test out your ImageNet datasource using the
[alexnet.lua](https://github.com/nicholas-leonard/dp/blob/master/examples/alexnet.lua)
training script.

<a name="dp.BillionWords"></a>
## BillionWords ##
A [DataSource](#dp.DataSource) subclass wrapping the corpus derived from the 
`training-monolingual.tokenized/news.20??.en.shuffled.tokenized` data distributed for [WMT11](http://statmt.org/wmt11/translation-task.html). The preprocessing suggested by 
the [Google 1-Billion Words language modeling benchmark](https://code.google.com/p/1-billion-word-language-modeling-benchmark) was used to prepare the data. 
The task consists in predicting the next word given the `n` previous ones, 
where `n` is the context size (it can be set in the constructor).
The data consists in approximately 30 million sentences of an average length of about 25 words.
In, there are about 800 thousand (unique) words in the vocabulary, which makes it a very memory intensive problem.
The DataSource inclues data for building hierarchical softmaxes to accelerate training.
As usual the actual data is downloaded automatically when not found on disk.
It is stored as a serialized `torch.Tensor` (see code for details).

<a name="dp.PennTreeBank"></a>
## Penn Tree Bank ##
A small [TextSource](#dp.TextSource) used to train and evaluate language models. 
It has a relatively small vocabulary of 10000 words. 

<a name="dp.Sampler"></a>
## Sampler ##
A [DataSet](#dp.DataSet) iterator which sequentially samples [Batches](#dp.Batch) 
from a DataSet for a [Propagator](propagator.md#dp.Propagator).
This iterator calls the [sub](#dp.DataSet.sub) Batch factory method.

<a name="dp.Sampler.__init"></a>
### dp.Sampler{batch_size, epoch_size} ###
A constructor having the following arguments:

 * `batch_size`  is a number specifying the number of examples per batch. Defaults to 1024.
 * `epoch_size` specifies the number of examples presented per epoch. When `epoch_size` is less than the size of the dataset, the sampler resumes processing the dataset from its ending position the next time [sampleEpoch](#dp.Sampler.sampleEpoch) is called. When `epoch_size` is greater, it loops through the dataset until enough samples are draw. The default (-1) is to use then entire dataset per epoch.
 * `ppf` is an optional function that preprocesses a Batch into another Batch. 

<a name="dp.Sampler.sampleEpoch"></a>
### [sampleBatch] sampleEpoch(dataset) ###
Returns an iterator over a `dataset` for one epoch. The 
returned `sampleBatch` iterator is a function taking one optional argument : `batch`, 
which is a [Batch](#dp.Batch). When this argument is provided, its encapsulated 
memory will be reused. When called, the `sampleBatch` function returns 
the next Batch of examples, until the dataset has been iterated through for the duration 
of an epoch. This method is typically overwritten by sub-classes.

<a name="dp.Sampler.async"></a>
### async() ###
Used in conjuction with a [multithreaded](#dp.ImageClassSet.multithread) dataset
to iterate through Batches asynchronously. You can test the async functionality 
via the [alexnet.lua](https://github.com/nicholas-leonard/dp/blob/master/examples/alexnet.lua)
training script.

<a name="dp.ShuffleSampler"></a>
## ShuffleSampler ##
A subclass of [Sampler](#dp.Sampler) which iterates over [Batches](#dp.Batch) in a dataset 
by shuffling the example indices before each epoch. 
This iterator calls the [index](#dp.DataSet.index) Batch factory method.

<a name="dp.ShuffleSample.__init"></a>
### dp.ShuffleSampler{batch_size, random_seed} ###
A constructor having the following arguments:
 
  * `batch_size` specifies the number of examples per sampled batches. The default is 128.
  * `random_seed` is a number used to initialize the shuffle generator.

<a name="dp.SentenceSampler"></a>
## SentenceSampler ##
A subclass of [Sampler](#dp.Sampler) which iterates over parallel 
sentences of equal size one word at a time.
The sentences sizes are iterated through randomly.
Publishes to the `"beginSequence"` [Mediator](mediator.md#dp.Mediator) 
[Channel](mediator.md#dp.Channel) before each new Sequence, which prompts 
the recurrent Modules (like [Recurrent](https://github.com/Element-Research/rnn#rnn.Recurrent)
and [LSTM](https://github.com/Element-Research/rnn#rnn.LSTM)) to forget the previous sequence of inputs.
Note that `epoch_size` only garantees the minimum number of samples per epoch (more could be sampled).
Used for [Recurrent Neural Network Language Models](https://github.com/nicholas-leonard/dp/blob/master/examples/recurrentlanguagemodel.lua).

<a name='dp.RandomSampler'></a>
## RandomSampler ##
A [DataSet](#dp.DataSet) iterator which randomly samples batches from a dataset.
This iterator calls the [sample](#dp.ImageClassSet.sample) Batch factory method.
Unlike the [ShuffleSampler](#dp.ShuffleSampler), this iterator is not garanteed
to sample each example in the dataset during a complete epoch.
You can see this Sampler in action via the [alexnet.lua](https://github.com/nicholas-leonard/dp/blob/master/examples/alexnet.lua)
training script

<a name='dp.TextSampler'></a>
## TextSampler ##
A [TextSet](#dp.TextSet) iterator which is used mostly for recurrent language models.
Successive sampled batches are contiguous to each other. 
So sample `i` of the `j`th sampled batch is the sequence of words preceding the 
sequence of words in sample `i` of the `j+1`th batch. 
When `batch_size=1`, the entire text will be sampled one sequence at a time in order of appearence in the text.
