# Data #

  * [View](#dp.View) :
    * [DataView](#dp.DataView) :
     * [ImageView](#dp.ImageView)
     * [ClassView](#dp.ClassView)
     * [SequenceView](#dp.SequenceView)
    * [ListView](#dp.ListView)
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

<a name="dp.View"/>
## View ##
Abstract class inherited by ListViews (composites) and DataViews (components).
Used to communicate Tensors between Models in a variety of formats. 
Each format is specified by a `tensor_type` and `view`. 

A `tensor_type` is a string like : _torch.FloatTensor_, _torch.DoubleTensor_, _torch.IntTensor_, _torch.LongTensor_, 
_torch.CudaTensor_, etc. The type of a Tensor provided via `forwardPut` or `backwardPut` is implicitly ascertained 
via `torch.typename(tensor)` and thus does not need to be explicitly provided as an argument.

A `view` is a string like : `bf`, `bwc`, `bhwc`, `chwb`, `b`, etc. Each character in the string identifies a type of axis.
Possible axis symbols are : 
 1. Standard Axes: 
  * _b_ : Batch/Example 
  * _f_ : Feature 
  * _t_ : Class/Index 
 2. Spatial/Temporal/Volumetric Axes: 
  * _c_ : Color/Channel
  * _h_ : Height 
  * _w_ : Width 
  * _d_ : Dept 
A `view` thus specifies the order and nature of a provided or requested tensor's axes.

A View is used at the input and output of Models. For example, in a Sequence, the first and second 
Models will share a View. The first Model's output View is the second Model's input View. These Views abstract 
away [Modules](https://github.com/torch/nn/blob/master/doc/module.md#module) like :
 * [Identity](https://github.com/torch/nn/blob/master/doc/simple.md#nn.Identity) : used for forwarding base `input` Tensor as is (or conversely, backwarding `gradOutput` Tensor); 
 * [Reshape](https://github.com/torch/nn/blob/master/doc/simple.md#nn.Reshape) : used for resizing (down-sizing) the `input` and `gradOutput` Tensors, 
 * [Transpose](https://github.com/torch/nn/blob/master/doc/simple.md#nn.Transpose) : used for tranposing axe of `input` and `gradOutput` Tensors; and 
 * [Copy](https://github.com/torch/nn/blob/master/doc/simple.md#nn.Copy) : used for forward/backward propagating a different `tensor_type`.
By using these Modules to forward/backward Tensors, Views abstract away the tedious transformations that need to be performed and tested between Models. As such, 
a Node (Model or Loss) need only specify a `view` and `tensor_type` for inputs and outputs. 

A View is also used by DataSets and DataSources to encapsulate inputs and targets. So a the first Model of a Sequence might 
share its input View with the an [indexed](#dp.View.index) or a [sub](#dp.View.sub) View of a DataSet's inputs. This also 
means that a [Preprocess](preprocess.md#dp.Preprocess) would be applied to a View, thus requiring a call to `forwardGet`.

<a name="dp.View.forwardPut"/>
### forwardPut(view, input) ###
This method should be called by a maximum of one input Model. Any
subsequent call overwrites the previous call and reinitializes the 
internal tensor cache.

It is assumed that any `input` Tensor to `forwardPut` is represented as
the most expanded size of the orignal data. For example, an image
batch would be forwarded as a 4D `tensor` and `view`, and never with 
collapses dimensions (2D).

<a name="dp.View.forwardGet"/>
### [output] forwardGet(view, [tensor_type]) ###
This method could be called from multiple output Models. Can only be called 
following a previous call to `forwardPut`. Returns the requested `view` and 
`tensor_type` of the input tensor.

<a name="dp.View.forward"/>
### [output] forward(view, [inputORtype]) ###
A convenience function that can be used as either forwardPut or forwardGet, but not 
both at the same time.

<a name="dp.View.backwardPut"/>
### backwardPut(view, gradOutput) ###
This method could be called from multiple output Models. Each call to backwardPut must have been 
preceeded by a corresponding forwardGet, i.e. one with the same `view` and `tensor_type`.

<a name="dp.View.backwardGet"/>
### [gradInput] backwardGet(view, [tensor_type]) ###
This method should be called by a maximum of one input Model.
In the case of multiple output models having called backwardPut, 
the different gradInputs must be accumulated (through summation).

<a name="dp.View.backward"/>
### [gradInput] backward(view, [gradOutputORtype]) ###
A convenience function that can be used as either backwardPut or backwardGet, but not 
both at the same time.

<a name="dp.View.index"/>
### [view] index([v,] indices) ###
Returns a sub-[View](#dp.View) of the same type as self. 

`indices` is `torch.LongTensor` of indices of the batch (_b_) axis. The returned [View](#dp.View) will be `forwardPut` with 
the same `view` as self and with an `input` indexed from self's `input`. 

When `v` is provided, it will be `forwardPut` with the same 
`view` as self, and with an `input` indexed from a subset of self's `input`. The advantage of providing or reusing `v` from batch 
to batch is that the same storage (memory) can be used.

This method is used mainly by `ShuffleSampler` for retrieving random subsets of a dataset.

<a name="dp.View.sub"/>
### [view] sub(start, stop) ###
Returns a sub-[View](#dp.View) of the same type as self. 

`start` and `stop` identify the start and stop indices to sub from the batch (_b_) axis. The returned [View](#dp.View) will be `forwardPut` with 
the same `view` as self and with an `input` indexed from self's `input`. 

<a name="dp.DataView"/>
## DataView ##
Encapsulates an `input` torch.Tensor having a given `view` and `tensor_type` through [forwardPut](#dp.View.forwardPut). 
Models can request different `views` and `tensor_types` of the `input` through [forwardGet](#dp.View.forwardGet). 
If these differ from the base `input`, then the necessary nn.Reshape, nn.Transpose and nn.Copy are combined to efficiently 
provide the requested `view` and `tensor_type`. Modules are created once they are first requested and reused from batch 
to batch. Furthermore, the resulting tensors of these `forwardGets` calls are cached so that modules which may call for the 
same forwardGet will not incur multiple Module:forwards.

For any View used in a Model or DataSet, we expect forwardPut to be called only once per batch propagation, while forwardGet can 
be called with different views and tensor_types from different Models. The converse is true for backwardPut and backwardGet: 
we expect backwardGet to be called only once, while backwardPuts can come from different Models. We 
expect any forwardGet to preceed a corresponding backwardPut, i.e. one with the same view and tensor_type. 
During a backwardGet, any tensors provided via backwardPut are first backward propagated through any Modules to get the base
view and tensor_type (provided in forwardPut), and then accumulated these with a sum.

<a name="dp.DataView.__init"/>
### dp.DataView([view, input]) ###
Constructs a dp.DataView. When both `view` and `input` are provided, passes them to [forward](#dp.View.forwardPut) to initialize the 
base Tensor.

<a name="dp.DataView.bf"/>
### [module] bf() ###
Returns a [Module](https://github.com/torch/nn/blob/master/doc/module.md#module) that can transforms an `input` Tensor from the base `view` 
(i.e. the `view` provided in [forwardPut](#dp.View.forwardPut)) to `view` _bf_. The result of forwaring `input` through this Module would 
be a Tensor with the first axis (_b_) representing a batch of examples, and the second representing a set of features (_f_). This `view` is 
commonly used by the Neural Model. This method is called by `forwardGet` when `view` _bf_ is first requested (the Module for this transformation is created only once), unless
method [flush](#dp.DataView.flush) is called. This method should be supported by all Views (currently, only [ClassView](#dp.ClassView) lacks support for it). 

<a name="dp.DataView.b"/>
### [module] b() ###
Returns a [Module](https://github.com/torch/nn/blob/master/doc/module.md#module) that can transforms an `input` Tensor from the base `view` 
(i.e. the `view` provided in [forwardPut](#dp.View.forwardPut)) to `view` _b_. If the original forwardPut `view` was _bf_, the size of axis _f_ must be 1. Otherwise, 
the original `view` should have been _b_.

<a name="dp.DataView.replace"/>
### replace(view, output) ###
Used by [Preprocess](preprocess.md#dp.Preprocess) instances to replace the `input` with a preprocessed `output` retrieved using 
[forwardGet](#dp.View.forwardGet) with argument `view`. Internally, the method backward propagates `output` as a `gradOutput` to 
get a `gradInput` of the same format as `input`, replaces the `input` with that `gradInput`, and [flushs](#dp.DataView.flush) the 
module and tensor caches.

<a name="dp.DataView.flush"/>
### flush() ###
Flushes the Module and Tensor caches.

### [input] input([input]) ###
When `input` is provided, sets the base Tensor to that value. Otherwise, returns the base Tensor (the `input`). 
This method should be used with caution since it will cause errors if the `view` of the new `input` is different than 
that of the old, in which case a `forwardPut` would be more appropriate.

<a name="dp.ImageView"/>
## ImageView ##
A DataView subclass used for providing access to a tensor of images. This is useful since it allows for automatic reshaping. For example, let us suppose that we will be using a set of 10 images with 3x3 pixels and 1 channel (black and white).  
```lua
> dt = dp.ImageView{data=torch.rand(10,3*3), axes={'b','h','w','c'}, sizes={10,3,3,1}}
DataView Warning: data:size() is different than sizes. Assuming data is appropriately contiguous. Resizing data to sizes.
```
We can use an ImageView:image() for obtaining a representation suitable for convolutions and image preprocessing:
```lua
> =dt:image()
(1,1,.,.) = 
  0.4230
  0.3545
  0.8071

(2,1,.,.) = 
  0.8478
  0.7463
  0.5556

...
(10,3,.,.) = 
  0.7421
  0.5609
  0.4971
[torch.DoubleTensor of dimension 10x3x3x1]

```
Or we can use ImageView:feature() (inherited from [DataView](#dp.DataView.feature) to obtain a representation suitable for MLPs and feature preprocessing:
```lua
> =dt:feature()
0.4230  0.3545  0.8071  0.1717  0.6072  0.9120  0.6389  0.5002  0.0237
 0.8478  0.7463  0.5556  0.7995  0.2141  0.5164  0.2037  0.2733  0.5226
 0.6114  0.3613  0.2784  0.2083  0.9485  0.5826  0.7669  0.0177  0.0550
 0.9148  0.9391  0.1449  0.4779  0.6515  0.9311  0.4179  0.1163  0.8002
 0.6517  0.3549  0.0900  0.3038  0.4123  0.0991  0.0148  0.8528  0.7237
 0.8487  0.5838  0.2006  0.0378  0.1517  0.1992  0.2076  0.3537  0.7024
 0.0856  0.4508  0.9910  0.3905  0.6099  0.9126  0.1718  0.8962  0.1037
 0.8119  0.4987  0.9008  0.2354  0.6697  0.8641  0.0031  0.8939  0.1399
 0.1546  0.5477  0.1261  0.9096  0.7459  0.6923  0.6901  0.2539  0.7569
 0.2150  0.8002  0.3193  0.1342  0.1905  0.1681  0.7421  0.5609  0.4971
[torch.DoubleTensor of dimension 10x9]
```

<a name="dp.ImageView.__init"/>
### dp.ImageView{data, [axes, sizes]} ###
Constructs a dp.ImageView out of torch.Tensor data. Arguments can also be passed as a table of key-value pairs:
```lua
> dt = dp.ImageView{data=torch.Tensor(10000,28*28), axes={'b','h','w','c'}, sizes={28,28,1}}
```

`data` is a torch.Tensor with at least 2 dimensions.  

`axes` is a table defining the order and nature of each dimension of the expanded torch.Tensor. 
It should be the most expanded version of the `data`. For example, while an individual image can be represented as a vector, in which case it takes the form of `{'b','f'}`, its expanded axes format could be `{'b','h','w','c'}`. Defaults to the latter.

`sizes` can be a table, a torch.LongTensor or a torch.LongStorage. A table or torch.LongTensor holding the `sizes` of the commensurate dimensions in `axes`. This should be supplied if the dimensions of the data is different from the number of elements in `axes`, in which case it will be used to : `data:reshape(sizes)`. Defaults to data:size().


<a name="dp.ImageView.image"/>
### [data, axes] image([inplace, contiguous]) ###
Returns a 4D-tensor of axes format : `{'b','h','w','c'}`.

`inplace` is a boolean. When true, makes `data` a contiguous view of `axes`
`{'b','h','w','c'}` for future use. Defaults to true.
 
`contiguous` is a boolean. When true, makes sure the returned data is contiguous. 
Since `inplace` makes it contiguous anyway, this parameter is only considered when `inplace=false`. Defaults to false.


<a name="dp.ImageView.feature"/>
### [data] feature([inplace, contiguous]) ###
Returns a 2D torch.Tensor of examples by features : `{'b','f'}` (see [DataView:feature()](#dp.DataView.feature)).


<a name="dp.ClassView"/>
## ClassView ##
A DataView subclass used for providing access to a tensor of classes. This is useful since it allows for automatic reshaping. For example, let us suppose that we will be using a set of 8 samples picked from the following set of 4 classes : `{0,1,2,3}`.  
```lua
> dt = dp.ClassView{data=torch.IntTensor{4,1,3,4,1,2,3,1}, classes={0,1,2,3}}
```

We can use an ClassView:class() for obtaining a representation suitable for use as targets in nn.ClassNLLCriterion:
```lua
> =dt:class()
DataView Warning: Assuming one class per example.	
 4
 1
 3
 4
 1
 2
 3
 1
[torch.IntTensor of dimension 8]
```
Or we can use ClassView:multiclass() (inherited from [DataView](#dp.DataView.feature) to obtain a representation with an extra dimension that permits each example to have multiple target classes.
```lua
> =dt:multiclass()
 4
 1
 3
 4
 1
 2
 3
 1
[torch.IntTensor of dimension 8x1]
```

<a name="dp.ClassView.__init"/>
### dp.ClassView{data, [axes, sizes]} ###
Constructs a dp.ImageView out of torch.Tensor data. Arguments can also be passed as a table of key-value pairs:
```lua
dt = dp.ClassView{data=torch.Tensor(10000), axes={'b'}, sizes={10000}}
```

`data` is a torch.Tensor with at least 1 dimensions.  

`axes` is a table defining the order and nature of each dimension of the expanded torch.Tensor. 
It should be the most expanded version of the `data`. Defaults to `{'b'}` for `#sizes==1` or `{'b','t'}` for `#sizes==2`.

`sizes` can be a table, a torch.LongTensor or a torch.LongStorage holding the `sizes` of the commensurate dimensions in `axes`. This should be supplied if the dimensions of the data is different from the number of elements in `axes`, in which case it will be used to : `data:reshape(sizes)`. Defaults to data:size().

`classes` is an optional table listing class IDs. The first index value is associated to class index 1, the second to 2, etc. For example, we could represent MNIST `data` containing classes indexed from `1,2...10` using `classes={0,1,2,3,4,5,6,7,8,9}`, supposing of course that the `0` MNIST-digits are indexed in `data` as `1`s.

<a name="dp.ClassView.class"/>
### [data, axes] class([inplace, contiguous]) ###
Returns a 1D-tensor of axes format : `{'b'}`.

`inplace` is a boolean. When true, makes `data` a contiguous view of `axes`
`{'b'}` for future use. Defaults to true.
 
`contiguous` is a boolean. When true, makes sure the returned data is contiguous. 
Since `inplace` makes it contiguous anyway, this parameter is only considered when `inplace=false`. Defaults to false.

<a name="dp.ClassView.multiclass"/>
### [data] multiclass([inplace, contiguous]) ###
Returns a 2D-tensor of axes format : `{'b','t'}`. 

A ClassView where each example has many classes can be represented by vectors of classes. The data is thus of form `{'b','t'}`. So for example, we could be using a set of 4 samples of 2 classes each picked from the following set of 4 classes : `{0,1,2,3}`.  
```lua
> dt = dp.ClassView{data=torch.IntTensor{{4,2},{3,1},{2,3},{3,4}}, classes={0,1,2,3}}
```
We can use an ClassView:multiclass() for obtaining a representation suitable for use as targets in nn.ClassNLLCriterion:
```lua
> =dt:multiclass()
 0  1
 3  0
 1  2
 3  1
[torch.IntTensor of dimension 4x2]
```
However, assuming the first index of each example vector represents the primary class, we could also call `ClassView:class()`:
```lua
> =dt:class()
 0
 3
 1
 3
[torch.IntTensor of dimension 4]
```

<a name="dp.ClassView.feature"/>
### [data] feature([inplace, contiguous]) ###
Returns a 2D torch.Tensor of examples by features : `{'b','f'}` (see [DataView:feature()](#dp.DataView.feature)).

<a name="dp.ListView"/>
## ListView ##
A composite of Views. Allows for multiple input and multiple target datasets and batches.

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

`inputs` is an instance of [View](#dp.View) or a table of these. In the latter case, they will be 
automatically encapsulated by a [ListView](#dp.ListView). These are used as inputs to a 
[Model](model.md#dp.Model).

`targets` is an instance of `View` or a table of these. In the latter case, they will be 
automatically encapsulated by a `ListView`. These are used as targets for 
training a `Model`. The indices of examples in `targets` must be aligned with those in `inputs`. 


`which_set` is a string identifying the purpose of the dataset. Valid values are 
 * *train* for training, i.e. for fitting a model to a dataset; 
 * *valid* for cross-validation, i.e. for early-stopping and hyper-optimization; 
 * *test* for testing, i.e. comparing your model to the current state-of-the-art and such.

<a name="dp.DataSet"/>
## DataSet ##
A concrete subclass of [BaseSet](#dp.BaseSet). A 

<a name="dp.DataSet.__init"/>
### dp.DataSet(which_set, inputs, [targets]) ###
Constructs a training, validation or test DataSet from [a set of] [View](#dp.View) `inputs` and `targets`.

<a name="dp.DataSet.preprocess"/>
### preprocess([input_preprocess, target_preprocess, can_fit] ###
TODO

<a name="dp.DataSet.inputs"/>
### [inputs] inputs([index]) ###

<a name="dp.DataSource"/>
## DataSource ##
TODO


