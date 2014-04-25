# Data #

  * [BaseTensor](#dp.BaseTensor) :
    * [DataTensor](#dp.DataTensor) :
     * [ImageTensor](#dp.ImageTensor)
     * [ClassTensor](#dp.ClassTensor)
    * [CompositeTensor](#dp.CompositeTensor)
  * [DataSet](#dp.DataSet)
  * [DataSource](#dp.DataSource) :
    * Mnist
    * NotMnist
    * Cifar10
    * Cifar100 

<a name="dp.BaseTensor"/>
## BaseTensor ##
Abstract class to allow for the use of CompositeTensors in models. Adapter (design pattern) for torch.Tensor.

<a name="dp.DataTensor"/>
## DataTensor ##
Encapsulates a torch.Tensor and provides access to it using different
viewing methods. A view may reshape the tensor inplace and render it
contiguous. Views can be used to convert data into new axes formats 
using torch.Tensor:resize(), :transpose() and :contiguous(). The 
conversions may be done in-place (default), or may be simply  
returned using the conversion methods (feature, class, image, etc.). 
A DataTensor may also holds metadata.

<a name="dp.DataTensor.__init"/>
### dp.DataTensor{data, [axes, sizes]} ###
Constructs a dp.DataTensor out of torch.Tensor data. Arguments can also be passed as a table of key-value pairs :
```dt = dp.DataTensor{data=torch.Tensor(3,4), axes={'b','f'}, sizes={3,4}}```

`data` is a torch.Tensor with at least 1 dimensions. 

`axes` is a table defining the order and nature of each dimension of a torch.Tensor. Two common examples would be the archtypical MLP input : `{'b','f'}`, 
or a common image representation : `{'b','h','w','c'}`. 
Possible axis symbols are : 
 1. Standard Axes: 
  * `'b'` : Batch/Example 
  * `'f'` : Feature 
  * `'t'` : Class 
 2. Image Axes 
  * `'c'` : Color/Channel 
  * `'h'` : Height 
  * `'w'` : Width 
  * `'d'` : Dept 

The provided `axes` should be the most expanded version of the `data` (the version with the most dimensions). For example, while an image can be represented as a vector, in which case it takes the form of `{'b','f'}`, its expanded axes format could be `{'b','h','w','c'}`. Defaults to `{'b','f'}`.

`sizes` can be a table, a torch.LongTensor or a torch.LongStorage. A table or torch.LongTensor holding the `sizes` of the commensurate dimensions in `axes`. This should be supplied if the dimensions of the data is different from the number of elements in `axes`, in which case it will be used to : `data:reshape(sizes)`. If the sizes has one less elements than `axes`, then it assumes that the missing element is the batch dimension `b` and further extrapolates its size from `data`. Defaults to data:size(). 

<a name="dp.DataTensor.feature"/>
### [data] feature([inplace, contiguous]) ###
Returns a 2D torch.Tensor of examples by features : `{'b','f'}`.

`inplace` is a boolean. When true, makes `data` a contiguous view of `axes`
`{'b','f'}` for future use. Defaults to true.
 
`contiguous` is a boolean. When true, makes sure the returned data is contiguous. 
Since `inplace` makes it contiguous anyway, this parameter is only considered when `inplace=false`. Defaults to false.


<a name="dp.ImageTensor"/>
## ImageTensor ##
A DataTensor subclass used for providing access to a tensor of images. This is useful since it allows for automatic reshaping. For example, let us suppose that we will be using a set of 10 images with 3x3 pixels and 1 channel (black and white).  
```lua
> dt = dp.ImageTensor{data=torch.rand(10,3*3), axes={'b','h','w','c'}, sizes={10,3,3,1}}
DataTensor Warning: data:size() is different than sizes. Assuming data is appropriately contiguous. Resizing data to sizes.
```
We can use an ImageTensor:image() for obtaining a representation suitable for convolutions and image preprocessing:
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
Or we can use ImageTensor:feature() (inherited from [DataTensor](#dp.DataTensor.feature) to obtain a representation suitable for MLPs and feature preprocessing:
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

<a name="dp.ImageTensor.__init"/>
### dp.ImageTensor{data, [axes, sizes]} ###
Constructs a dp.ImageTensor out of torch.Tensor data. Arguments can also be passed as a table of key-value pairs:
```lua
> dt = dp.ImageTensor{data=torch.Tensor(10000,28*28), axes={'b','h','w','c'}, sizes={28,28,1}}
```

`data` is a torch.Tensor with at least 2 dimensions.  

`axes` is a table defining the order and nature of each dimension of the expanded torch.Tensor. 
It should be the most expanded version of the `data`. For example, while an individual image can be represented as a vector, in which case it takes the form of `{'b','f'}`, its expanded axes format could be `{'b','h','w','c'}`. Defaults to the latter.

`sizes` can be a table, a torch.LongTensor or a torch.LongStorage. A table or torch.LongTensor holding the `sizes` of the commensurate dimensions in `axes`. This should be supplied if the dimensions of the data is different from the number of elements in `axes`, in which case it will be used to : `data:reshape(sizes)`. Defaults to data:size().


<a name="dp.ImageTensor.image"/>
### [data, axes] image([inplace, contiguous]) ###
Returns a 4D-tensor of axes format : `{'b','h','w','c'}`.

`inplace` is a boolean. When true, makes `data` a contiguous view of `axes`
`{'b','h','w','c'}` for future use. Defaults to true.
 
`contiguous` is a boolean. When true, makes sure the returned data is contiguous. 
Since `inplace` makes it contiguous anyway, this parameter is only considered when `inplace=false`. Defaults to false.


<a name="dp.ImageTensor.feature"/>
### [data] feature([inplace, contiguous]) ###
Returns a 2D torch.Tensor of examples by features : `{'b','f'}` (see [DataTensor:feature()](#dp.DataTensor.feature)).


<a name="dp.ClassTensor"/>
## ClassTensor ##
A DataTensor subclass used for providing access to a tensor of classes. This is useful since it allows for automatic reshaping. For example, let us suppose that we will be using a set of 8 samples picked from the following set of 4 classes : `{0,1,2,3}`.  
```lua
> dt = dp.ClassTensor{data=torch.IntTensor{4,1,3,4,1,2,3,1}, classes={0,1,2,3}}
```

We can use an ClassTensor:class() for obtaining a representation suitable for use as targets in nn.ClassNLLCriterion:
```lua
> =dt:class()
DataTensor Warning: Assuming one class per example.	
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
Or we can use ClassTensor:multiclass() (inherited from [DataTensor](#dp.DataTensor.feature) to obtain a representation with an extra dimension that permits each example to have multiple target classes.
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

<a name="dp.ClassTensor.__init"/>
### dp.ClassTensor{data, [axes, sizes]} ###
Constructs a dp.ImageTensor out of torch.Tensor data. Arguments can also be passed as a table of key-value pairs:
```lua
dt = dp.ClassTensor{data=torch.Tensor(10000), axes={'b'}, sizes={10000}}
```

`data` is a torch.Tensor with at least 1 dimensions.  

`axes` is a table defining the order and nature of each dimension of the expanded torch.Tensor. 
It should be the most expanded version of the `data`. Defaults to `{'b'}` for `#sizes==1` or `{'b','t'}` for `#sizes==2`.

`sizes` can be a table, a torch.LongTensor or a torch.LongStorage holding the `sizes` of the commensurate dimensions in `axes`. This should be supplied if the dimensions of the data is different from the number of elements in `axes`, in which case it will be used to : `data:reshape(sizes)`. Defaults to data:size().

`classes` is an optional table listing class IDs. The first index value is associated to class index 1, the second to 2, etc. For example, we could represent MNIST `data` containing classes indexed from `1,2...10` using `classes={0,1,2,3,4,5,6,7,8,9}`, supposing of course that the `0` MNIST-digits are indexed in `data` as `1`s.

<a name="dp.ClassTensor.class"/>
### [data, axes] class([inplace, contiguous]) ###
Returns a 1D-tensor of axes format : `{'b'}`.

`inplace` is a boolean. When true, makes `data` a contiguous view of `axes`
`{'b'}` for future use. Defaults to true.
 
`contiguous` is a boolean. When true, makes sure the returned data is contiguous. 
Since `inplace` makes it contiguous anyway, this parameter is only considered when `inplace=false`. Defaults to false.

<a name="dp.ClassTensor.multiclass"/>
### [data] multiclass([inplace, contiguous]) ###
Returns a 2D-tensor of axes format : `{'b','t'}`. 

A ClassTensor where each example has many classes can be represented by vectors of classes. The data is thus of form `{'b','t'}`. So for example, we could be using a set of 4 samples of 2 classes each picked from the following set of 4 classes : `{0,1,2,3}`.  
```lua
> dt = dp.ClassTensor{data=torch.IntTensor{{4,2},{3,1},{2,3},{3,4}}, classes={0,1,2,3}}
```
We can use an ClassTensor:multiclass() for obtaining a representation suitable for use as targets in nn.ClassNLLCriterion:
```lua
> =dt:multiclass()
 0  1
 3  0
 1  2
 3  1
[torch.IntTensor of dimension 4x2]
```
However, assuming the first index of each example vector represents the primary class, we could also call `ClassTensor:class()`:
```lua
> =dt:class()
 0
 3
 1
 3
[torch.IntTensor of dimension 4]
```

<a name="dp.ClassTensor.feature"/>
### [data] feature([inplace, contiguous]) ###
Returns a 2D torch.Tensor of examples by features : `{'b','f'}` (see [DataTensor:feature()](#dp.DataTensor.feature)).

<a name="dp.CompositeTensor"/>
## CompositeTensor ##
A composite of BaseTensors. Allows for multiple input and multiple target datasets and batches.


<a name="dp.DataSet"/>
## DataSet ##
Generic container for inputs and optional targets. Used for training or evaluating a model. 
Inputs/targets are tables of DataTensors, which allows for multi-input / multi-target DataSets.

If the provided inputs or targets are torch.Tensors, an attempt is 
made to convert them to [DataTensor](#dp.DataTensor) using the optionally 
provided axes and sizes (inputs), classes (targets).

Inputs and targets should be provided as instances of 
[DataTensor](#dp.DataTensor) to support conversions to other axes formats. 
Inputs and targets may also be provided as tables of 
dp.DataTensors. In the case of multiple targets, it is useful for multi-task learning, 
or learning from hints . In the case of multiple inputs, richer inputs representations could 
be created allowing, for example, images to be combined with 
tags, text with images, etc. Multi-input/target facilities could be used with nn.ParallelTable and 
nn.ConcatTable.

If the DataSet is used for unsupervised learning, only inputs need to 
be provided.

<a name="dp.DataSet.__init"/>
### dp.DataSet(which_set, inputs, [targets, axes, sizes]) ###
Constructs a training, validation or test DataSet from [a set of] [DataTensor](#dp.DataTensor) `inputs` and `targets`.

`which_set` is a string of value `'train'`, `'valid'` or `'test'` for identifying sets used for training, cross-validation and testing, respectively.

`inputs` is either a [DataTensor](#dp.DataTensor), a torch.Tensor or a table. Inputs of the DataSet taking the form of torch.Tensor with 2 dimensions, or more if topological is true. Alternatively, inputs may take the form of a table of such torch.Tensors. The first dimension of the torch.Tensor(s) should be of size number of examples.
    [targets = dp.DataTensor | torch.Tensor | table]-- Targets of the DataSet taking the form of torch.Tensor with 1-2 dimensions. Alternatively, targets may take the form of a table of such torch.Tensors. The first dimension of the torch.Tensor(s) should index examples.
    [axes = table]                      -- Optional. Used when supplied inputs is a torch.Tensor, in order to convert it to dp.DataTensor. In which case, it is a table defining the order and nature of each dimension of a tensor. Two common examples would be the archtypical MLP input : {"b", "f"}, or a common image representation :      {"b", "h", "w", "c"}.     Possible axis symbols are :      1. Standard Axes:        "b" : Batch/Example        "f" : Feature        "t" : Class      2. Image Axes        "c" : Color/Channel        "h" : Height        "w" : Width        "d" : Dept      Defaults to the dp.DataTensor default.
    [sizes = table | torch.LongTensor]  -- Optional. Used when supplied inputs is a torch.Tensor, in order to convert it to dp.DataTensor. In which case, it is a table or torch.LongTensor identifying the sizes of the commensurate dimensions in axes. This should be supplied if the dimensions of the data is different from the number of elements in the axes table, in which case it will be used to : data:resize(sizes). Defaults to dp.DataTensor default.
}

DataSet{inputs=dp.DataTensor | torch.Tensor | table, which_set=string}
DataSet(string, ...)


<a name="dp.DataSet.preprocess"/>
### preprocess([input_preprocess, target_preprocess, can_fit] ###
TODO

<a name="dp.DataSet.inputs"/>
### [inputs] inputs([index]) ###

<a name="dp.DataSource"/>
## DataSource ##
TODO


