# Bird's Eye View #
Views encapsulate an `input` Tensor and one or many `gradOutput` Tensor. 
These can be forward and backward accessed (put/get) by specifying a `view` and an optional `tensor_type`: 

  * [View](#dp.View) : Abstract class inherited by all Views.
    * [DataView](#dp.DataView) : provides for views _b_ and _bf_;
      * [ImageView](#dp.ImageView) : provides views _bhwc_, _bchw_ and _chwb_;
      * [ClassView](#dp.ClassView) : provides views _b_ (overwritten) and _bt_;
      * [SequenceView](#dp.SequenceView) : provides views _bwc_ and _bcw_;
    * [ListView](#dp.ListView) : Composite of Views (work in progress).

<a name="dp.View"/>
[]()
## View ##
Abstract class inherited by ListViews (composites) and DataViews (components).
Used to communicate Tensors between Models in a variety of formats. 
Each format is specified by a `tensor_type` and `view`. 

A `tensor_type` is a string like : _torch.FloatTensor_, _torch.DoubleTensor_, _torch.IntTensor_, _torch.LongTensor_, 
_torch.CudaTensor_, etc. The type of a Tensor provided via `forwardPut` or `backwardPut` is implicitly ascertained 
via `torch.typename(tensor)` and thus does not need to be explicitly provided as an argument.

A `view` is a string like : _bf_, _bwc_, _bhwc_, _chwb_, _b_, etc. Each character in the string identifies a type of axis.
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
[]()
### forwardPut(view, input) ###
This method should be called by a maximum of one input Model. Any
subsequent call overwrites the previous call and reinitializes the 
internal tensor cache.

It is assumed that any `input` Tensor to `forwardPut` is represented as
the most expanded size of the orignal data. For example, an image
batch would be forwarded as a 4D `tensor` and `view`, and never with 
collapses dimensions (2D).

<a name="dp.View.forwardGet"/>
[]()
### [output] forwardGet(view, [tensor_type]) ###
This method could be called from multiple output Models. Can only be called 
following a previous call to `forwardPut`. Returns the requested `view` and 
`tensor_type` of the input tensor.

<a name="dp.View.forward"/>
[]()
### [output] forward(view, [inputORtype]) ###
A convenience function that can be used as either forwardPut or forwardGet, but not 
both at the same time.

<a name="dp.View.backwardPut"/>
[]()
### backwardPut(view, gradOutput) ###
This method could be called from multiple output Models. Each call to backwardPut must have been 
preceeded by a corresponding forwardGet, i.e. one with the same `view` and `tensor_type`.

<a name="dp.View.backwardGet"/>
[]()
### [gradInput] backwardGet(view, [tensor_type]) ###
This method should be called by a maximum of one input Model.
In the case of multiple output models having called backwardPut, 
the different gradInputs must be accumulated (through summation).

<a name="dp.View.backward"/>
[]()
### [gradInput] backward(view, [gradOutputORtype]) ###
A convenience function that can be used as either backwardPut or backwardGet, but not 
both at the same time.

<a name="dp.View.index"/>
[]()
### [view] index([v,] indices) ###
Returns a sub-[View](#dp.View) of the same type as self. 

`indices` is `torch.LongTensor` of indices of the batch (_b_) axis. The returned [View](#dp.View) will be `forwardPut` with 
the same `view` as self and with an `input` indexed from self's `input`. 

When View `v` is provided, it will be `forwardPut` with the same 
`view` as self, and with an `input` indexed from a subset of self's `input`. The advantage of providing or reusing `v` from batch 
to batch is that the same storage (memory) can be used. This includes any Modules (and their allocated memory) in `v`'s cache.

This method is used mainly by ShuffleSample` for retrieving random subsets of a dataset.

<a name="dp.View.sub"/>
[]()
### [view] sub([v,] start, stop) ###
Returns a sub-[View](#dp.View) of the same type as self. 

`start` and `stop` identify the start and stop indices to sub from the batch (_b_) axis. The returned [View](#dp.View) will be `forwardPut` with 
the same `view` as self and with an `input` indexed from self's `input`. 

When View `v` is provided, it will be `forwardPut` with the same 
`view` as self, and with an `input` having a copy of a subset of self's `input`. The advantage of providing or reusing `v` from batch 
to batch is that the same storage (memory) can be used. This includes any Modules (and their allocated memory) in `v`'s cache.

This method is used mainly by Sampler for retrieving random subsets of a dataset.

<a name="dp.DataView"/>
[]()
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
[]()
### dp.DataView([view, input]) ###
Constructs a dp.DataView. When both `view` and `input` are provided, passes them to [forward](#dp.View.forwardPut) to initialize the 
base Tensor.

<a name="dp.DataView.bf"/>
[]()
### [module] bf() ###
Returns a [Module](https://github.com/torch/nn/blob/master/doc/module.md#module) that transforms an `input` Tensor from the base `view` 
(i.e. the `view` provided in [forwardPut](#dp.View.forwardPut)) to `view` _bf_. The result of forwaring `input` through this Module would 
be a Tensor with the first axis (_b_) representing a batch of examples, and the second representing a set of features (_f_). If the base `input` Tensor 
has more axes than 2, the non-_b_ axes are collapsed into a single _f_ axis using a Reshape Module. This viewing method is 
commonly used by the Neural Model. This method is called by `forwardGet` when `view` _bf_ is first requested (the Module for this transformation is created only once), unless
method [flush](#dp.DataView.flush) is called. This method should be supported by all Views (currently, only [ClassView](#dp.ClassView) lacks support for it). 

<a name="dp.DataView.b"/>
[]()
### [module] b() ###
Returns a [Module](https://github.com/torch/nn/blob/master/doc/module.md#module) that transforms an `input` Tensor from the base `view` 
(i.e. the `view` provided in [forwardPut](#dp.View.forwardPut)) to `view` _b_. If the original forwardPut `view` was _bf_, the size of axis _f_ must be 1. Otherwise, 
the original `view` should have been _b_.

<a name="dp.DataView.replace"/>
[]()
### replace(view, output) ###
Used by [Preprocess](preprocess.md#dp.Preprocess) instances to replace the `input` with a preprocessed `output` retrieved using 
[forwardGet](#dp.View.forwardGet) with argument `view`. Internally, the method backward propagates `output` as a `gradOutput` to 
get a `gradInput` of the same format as `input`, replaces the `input` with that `gradInput`, and [flushs](#dp.DataView.flush) the 
module and tensor caches.

<a name="dp.DataView.flush"/>
[]()
### flush() ###
Flushes the Module and Tensor caches. This should be called before [forwardPut](#dp.View.forwardPut) is about
to be used for a new kind of base `view` than was previously used. If not, results might be different than expected.

<a name="dp.DataView.input"/>
[]()
### [input] input([input]) ###
When `input` is provided, sets the base Tensor to that value. Otherwise, returns the base Tensor (the `input`). 
This method should be used with caution since it will cause errors if the `view` of the new `input` is different than 
that of the old, in which case a `forwardPut` would be more appropriate.

<a name="dp.DataView.transpose"/>
[]()
### [module] transpose(view) ###
A generic function for transposing views. Returns a [Transpose](https://github.com/torch/nn/blob/master/doc/simple.md#nn.Transpose) Module 
that transposes the `input` such that the requested `view` would be the result. Commonly used by viewing methods like [bhwc](#dp.ImageView.bhwc), 
[bchw](#dp.ImageView.bhwc), [chwb](#dp.ImageView.chwb), [bwc](#dp.SequenceView.bwc), etc.

<a name="dp.ImageView"/>
[]()
## ImageView ##
A [DataView](#dp.DataView) subclass used for providing access to a tensor of images. This is useful since it allows for automatic reshaping, transposing and such. 
For example, let us suppose that we will be using a set of 8 images with 3x3 pixels and 1 channel (black and white):
```lua
> dv = dp.ImageView()
> dv:forwardPut('bhwc', torch.rand(8,3,3,1):double())
```
Which is equivalent to:
```lua
> dv = dp.ImageView('bhwc', torch.rand(8,3,3,1):double())
```
We can use an [forwardGet](#dp.View.forwardGet) for obtaining a `view` of the `input` suitable for CUDA convolutions, i.e. _chwb_ :
```lua
> =dv:forwardGet('chwb', 'torch.CudaTensor')
(1,1,.,.) = 
  0.0748  0.7260  0.8156  0.4645  0.3130  0.7986  0.3976  0.0998
  0.9946  0.6130  0.8277  0.5419  0.0760  0.2964  0.0085  0.7283
  0.7094  0.6541  0.0811  0.0564  0.8081  0.1823  0.4881  0.1486

(1,2,.,.) = 
  0.1323  0.1587  0.0822  0.3691  0.5855  0.8046  0.1021  0.8321
  0.5754  0.4101  0.7095  0.5695  0.2273  0.0746  0.9465  0.6109
  0.5603  0.6446  0.7345  0.2679  0.5913  0.5497  0.1150  0.6649

(1,3,.,.) = 
  0.4442  0.7090  0.1919  0.0931  0.0364  0.5544  0.7864  0.6034
  0.7611  0.1444  0.2227  0.2894  0.5571  0.2201  0.0916  0.6040
  0.4468  0.0348  0.9614  0.5323  0.5906  0.9161  0.2670  0.4789
[torch.CudaTensor of dimension 1x3x3x8]

```
Or we can use it to obtain a `view` suitable for use with `Neural` Models and feature preprocessing:
```lua
> =dv:forwardGet('bf')
 0.0748  0.9946  0.7094  0.1323  0.5754  0.5603  0.4442  0.7611  0.4468
 0.7260  0.6130  0.6541  0.1587  0.4101  0.6446  0.7090  0.1444  0.0348
 0.8156  0.8277  0.0811  0.0822  0.7095  0.7345  0.1919  0.2227  0.9614
 0.4645  0.5419  0.0564  0.3691  0.5695  0.2679  0.0931  0.2894  0.5323
 0.3130  0.0760  0.8081  0.5855  0.2273  0.5913  0.0364  0.5571  0.5906
 0.7986  0.2964  0.1823  0.8046  0.0746  0.5497  0.5544  0.2201  0.9161
 0.3976  0.0085  0.4881  0.1021  0.9465  0.1150  0.7864  0.0916  0.2670
 0.0998  0.7283  0.1486  0.8321  0.6109  0.6649  0.6034  0.6040  0.4789
[torch.DoubleTensor of dimension 8x9]
```
Note that `tensor_type` (the second argument to forwardGet) defaults to the type of the base `input` Tensor:
```lua
> dv = dp.ImageView('bhwc', torch.rand(8,3,3,1):float())
> dv:forwardGet('bf')
 0.7328  0.9736  0.4569  0.6965  0.8301  0.9852  0.8396  0.8000  0.6286
 0.9026  0.8644  0.3703  0.4951  0.1519  0.8389  0.3995  0.9385  0.1023
 0.6046  0.4231  0.5078  0.8307  0.6814  0.4599  0.5189  0.9556  0.8790
 0.5981  0.0630  0.1761  0.2252  0.3956  0.0207  0.9859  0.6281  0.3034
 0.1236  0.8075  0.6222  0.7695  0.4779  0.5241  0.0491  0.5249  0.1492
 0.3703  0.7209  0.5492  0.0131  0.6006  0.6626  0.7629  0.9105  0.5165
 0.2549  0.7451  0.3175  0.8140  0.5132  0.0446  0.0695  0.0877  0.5090
 0.9385  0.6710  0.2921  0.9387  0.0820  0.3636  0.2498  0.4257  0.7185
[torch.FloatTensor of dimension 8x9]
```
<a name="dp.ImageView.bhwc"/>
[]()
### [module] bhwc() ###
Returns a [Module](https://github.com/torch/nn/blob/master/doc/module.md#module) that transforms an `input` Tensor from the base image `view` 
(i.e. the `view` provided in [forwardPut](#dp.View.forwardPut)) to `view` _bhwc_. The result of forwaring `input` through this Module would 
be a Tensor with, in sequence, axis _b_ representing a batch of examples, axis _h_ for height, _w_ for width and _c_ for color or channels. This `view` is 
commonly used by DataSources and DataSets, as well as [Preprocesses](preprocess.md#dp.Preprocess). 
This method is called by `forwardGet` when `view` _bhwc_ is first requested (the Module for this transformation is created only once), unless
method [flush](#dp.DataView.flush) is called afterwards.

<a name="dp.ImageView.bchw"/>
[]()
### [module] bchw() ###
Like viewing method [bhwc](#dp.ImageView.bhwc), except some axes are transposed. View _bchw_ is commonly used by SpatialConvolution Modules.

<a name="dp.ImageView.chwb"/>
[]()
### [module] chwc() ###
Like viewing method [bhwc](#dp.ImageView.bhwc), except some axes are transposed. View _bchw_ is commonly used by SpatialConvolutionCUDA Modules, i.e. CudaConvNet.


<a name="dp.ClassView"/>
[]()
## ClassView ##
A [DataView](#dp.DataView) subclass used for providing access to a tensor of indices like classes and words.

```lua
> dv = dp.ClassView('b', torch.IntTensor{4,1,3,4,1,2,3,1})
> dv:setClasses({0,1,2,3})
```

We can use an ClassView:class() for obtaining a representation suitable for use as targets in nn.ClassNLLCriterion:
```lua
> =dv:forwardGet('b')	
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
Or we can forwardGet 'view' _bt_ to obtain a representation with an extra dimension that permits each example to have multiple target classes:
```lua
> =dv:forwardGet('bt')
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

<a name="dp.ClassView.b"/>
[]()
### [module] b() ###
Overwrites [DataView](#dp.DataView)'s viewing method [b](#dp.DataView.b). 
Returns a [Module](https://github.com/torch/nn/blob/master/doc/module.md#module) that transforms an `input` Tensor from the base `view` 
(i.e. the `view` provided in [forwardPut](#dp.View.forwardPut)) to `view` _b_. If the original forwardPut `view` was _bt_, 
Module [Select](https://github.com/torch/nn/blob/master/doc/simple.md#nn.Select) is used to retrieve the first column of axis _t_. This allows 
DataSets to specify more than one class per example, as long as the primary class is kept first. 
Otherwise, the original `view` should have been _b_.

<a name="dp.ClassView.bt"/>
[]()
### [data] bt() ###
Returns a [Module](https://github.com/torch/nn/blob/master/doc/module.md#module) that transforms an `input` Tensor from the base `view` 
(i.e. the `view` provided in [forwardPut](#dp.View.forwardPut)) to `view` _bt_.
View _bt_ allows each example to have many classes. So for example, we could be using a set of 4 samples of 2 classes each 
picked from the following set of 4 classes : `{0,1,2,3}`.  
```lua
> dv = dp.ClassView('bt', torch.IntTensor{{4,2},{3,1},{2,3},{3,4}})
> dv:setClasses({0,1,2,3})
```
We can use forwardGet('b') for obtaining a representation suitable for use as targets in 
[ClassNLLCriterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.ClassNLLCriterion). 
The first index of each example vector represents the primary class:
```lua
> =dv:forwardGet('b')
 4
 3
 2
 3
[torch.IntTensor of dimension 4]
```
However, this doesn't mean we can't retrieve a multiple-class view of the `input` for use in some esotheric [Criterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.Criterion):
```lua
> =dv:forwardGet('bt')
 4  2
 3  1
 2  3
 3  4
[torch.IntTensor of dimension 4x2]

```

<a name='dp.SequenceView'/>
[]()
## SequenceView ##
A [DataView](#dp.DataView) subclass used for providing access to a tensor of sequences. This is useful since it allows for automatic reshaping, transposing and such. 
For example, let us suppose that we will be using a set of 8 sequences (_b_) of width (_w_) 2 with 5 channels (_c_). 
To use Natural Language Processing parlance, we could further conceive the _w_ axis to 
represent a context of 2 words and the _c_ axis to be an an extremely small embedding of size 5:
```lua
> dv = dp.SequenceView('bwc', torch.rand(8,2,5):double())
```
We can use an [forwardGet](#dp.View.forwardGet) for obtaining a `view` of the `input` suitable for TemporalConvolutions, i.e. _bwc_ :
```lua
> =dv:forwardGet('bwc')
(1,.,.) = 
  0.4882  0.8458  0.5664  0.6242  0.2657
  0.2575  0.9267  0.1456  0.9965  0.1016

(2,.,.) = 
  0.1117  0.0561  0.8146  0.0399  0.4839
  0.3808  0.2846  0.7869  0.9603  0.4089

...
(8,.,.) = 
  0.1481  0.0953  0.0942  0.5522  0.1065
  0.7581  0.9968  0.7940  0.9442  0.8679
[torch.DoubleTensor of dimension 8x2x5]

```
Or we can use it to obtain a `view` suitable for use with `Neural` Models and feature preprocessing:
```lua
> =dv:forwardGet('bf')
 0.4882  0.8458  0.5664  0.6242  0.2657  0.2575  0.9267  0.1456  0.9965  0.1016
 0.1117  0.0561  0.8146  0.0399  0.4839  0.3808  0.2846  0.7869  0.9603  0.4089
 0.8567  0.9975  0.2096  0.2453  0.0143  0.5626  0.2677  0.4176  0.0819  0.3801
 0.9413  0.3607  0.8067  0.6665  0.7380  0.3092  0.4405  0.1196  0.3091  0.5551
 0.2282  0.9919  0.7858  0.9116  0.0001  0.7646  0.6969  0.9296  0.1911  0.5132
 0.4895  0.5126  0.2285  0.6039  0.3652  0.6351  0.8333  0.3959  0.5688  0.1336
 0.9037  0.8789  0.3657  0.1996  0.7115  0.0282  0.3251  0.9679  0.2402  0.6089
 0.1481  0.0953  0.0942  0.5522  0.1065  0.7581  0.9968  0.7940  0.9442  0.8679
[torch.DoubleTensor of dimension 8x10]

```

<a name="dp.SequenceView.bwc"/>
[]()
### [module] bwc() ###
Returns a [Module](https://github.com/torch/nn/blob/master/doc/module.md#module) that transforms an `input` Tensor from the base image `view` 
(i.e. the `view` provided in [forwardPut](#dp.View.forwardPut)) to `view` _bwc_. The result of forwaring `input` through this Module would 
be a Tensor with, in sequence (no pun intended), axis _b_ representing a batch of examples, axis _w_ for width and _c_ for channels (or embedding sizes). 
This `view` is commonly used as the output of Model Dictionary, which adapts the 
[LookupTable](https://github.com/torch/nn/blob/master/doc/convolution.md#nn.LookupTable) Module, and as input and output `view` of the Convolution1D Model, 
which wraps the [TemporalConvolution](https://github.com/torch/nn/blob/master/doc/convolution.md#nn.TemporalConvolution) and 
[TemporalMaxPooling](https://github.com/torch/nn/blob/master/doc/convolution.md#nn.TemporalMaxPooling) Modules.
This method is called by `forwardGet` when `view` _bwc_ is first requested (the Module for this transformation is created only once), unless
method [flush](#dp.DataView.flush) is called afterwards.

<a name="dp.SequenceView.bcw"/>
[]()
### [module] bcw() ###
Like viewing method [bwc](#dp.SequenceView.bhwc), except some axes are transposed. View _bcw_ isn't currently used by anything.

<a name="dp.ListView"/>
[]()
## ListView ##
A composite of Views. Allows for multiple input and multiple target datasets and batches. Work in progress.

