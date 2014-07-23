# Models  #

 * [Model](#dp.Model) : abstract class inherited by Layer and Container;
 * [Layer](#dp.Layer) : abstract class inherited by component Models ;
   * [Neural](#dp.Neural) : Linear followed by a Transfer Module;
   * [Convolution1D](#dp.Convolution1D) : TemporalConvolution followed by a Transfer Module and TemporalMaxPooling;
   * [Convolution2D](#dp.Convolution2D) : SpatialConvolution followed by a Transfer Module and SpatialMaxPooling;
 * [Container](#dp.Container) : abstract class inherited by composite Models;
   * [Sequential](#dp.Sequential) : a sequence of Models.

<a name="dp.Model"/>
## Model ##
A Model is a [Node](node.md#dp.Node) sub-class. It adapts [Modules](https://github.com/torch/nn/blob/master/doc/module.md#module) to 
the *dp* library. Models can abstract multiple Modules. In fact, most Models, like [Neural](#dp.Neural) and [Convolution2D](#dp.Convolution2D)
contain 2-3 Modules. Model is the abstract class shared by component [Layer](#dp.Layer) and composite [Container](#dp.Container).

Unlike Modules, Models should be parameterized (although this isn't strictly enforced). This means that [Transfer](https://github.com/torch/nn/blob/master/doc/transfer.md) Modules 
shouldn't be specified in their own dedicated Model. Instead, we encourage that Models use Transfer Modules after parameterized Modules like 
[Linear](https://github.com/torch/nn/blob/master/doc/simple.md#nn.Linear) or [SpatialConvolution](https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialConvolution).

### dp.Model{typename, [tags, mvstate]} ###
Constructs a Model. Arguments should be specified as key-value pairs. Other then the following 
arguments, those specified in [Node](node.md#dp.Node.__init) also apply.

`typename` is a string which identifies the Model type in reports.

`tags` is a table of tags (as keys with values of true) used for determining which Visitors are allowed to visit the model.
 
`mvstate` is a table holding the Model-Visitor state. Can be used to specify arguments to Visitors that will adapt these to the Model.

<a name="dp.Model.forward"/>
### [output] forward(input, carry) ###
Forward propagates an `input` [View](view.md#dp.View) to fill and return an `output` View.

`input` is a [View](view.md#dp.View) that should have been previously filled by a [forwardPut](view.md#dp.View.forwardPut). 
The Model will call one or many [forwardGets](view.md#dp.View.forwardGet) to retrieve a Tensor in a suitable format for forward 
propagation through the Model's internal [Modules](https://github.com/torch/nn/blob/master/doc/module.md#module).

`carry` is a table that is carried throughout the graph. A Node can modify it, but should avoid deleting attributes.
This is useful when you want to forward/backward information to a later/previous Node in the graph seperated by an unknown number of [Nodes](node.md#dp.Node).

The returned `output` is a View filled using a forwardPut.

<a name="dp.Model.forward"/>
### [output] evaluate(input, carry) ###
This method is ike forward, but for evaluation purposes (valid/test).
This is useful for stochastic Modules like Dropout, which have 
different behavior for training than for evaluation. The default is to set 
`carry.evaluate = true` and to call [forward](#dp.Model.forward).

<a name="dp.Model.backward"/>
### [input] backward(output, carry) ###
Backward propagates an `output` [View](view.md#dp.View) to fill and `input` View with a gradient and return said `input`.

`output` is a [View](view.md#dp.View) that should have been previously filled by a [forwardPut](view.md#dp.View.forwardPut) 
in this Model's [forward](#dp.Model.forward) method and subsequently filled with a [backwardPut](view.md#dp.View.backwardPut)
from the next [Node](node.md#dp.Node) in the digraph. 
The Model will call [backwardGet](view.md#dp.View.forwardGet) on the `output` to retrieve a gradient Tensor in a suitable format for forward 
propagation through the Model's internal [Modules](https://github.com/torch/nn/blob/master/doc/module.md#module).

`carry` is a table that is carried throughout the graph. A Node can modify it, but should avoid deleting attributes.
This is useful when you want to forward/backward information to a later/previous Node in the graph seperated by an unknown number of Nodes.

The returned `input` is a View filled using a backwardPut. It is the same View that was passed to the previous call to this Model's [forward](#dp.Model.forward).

<a name="dp.Model.parameters"/>
### [params, gradParams [, scales]] parameters() ###
The method is used by Visitors for manipulating parameters and gradients. Return 2 to 3 tables : 
 * `params` : a table of parameters used by the Model;
 * `gradParams` : a table of parameter gradients (w.r.t Loss) used by the Model; 
 * `scales` : a table of update scales (optional).
Each param/gradParam/scale triplet must be identified by a unique key, i.e. the tensors associated to each key must be the same from batch to batch. 
This allows Visitors to use the Model's `mvstate` table to append meta-data to the triplet for later use. This is the case of Momentum which must 
accumulate `pastGrads` for each triplet.

<a name="dp.Model.accept"/>
### accept(visitor) ###
Accepts a `visitor` Visitor that will visit the Model and any of its component Models. This is how the Model's parameters are updated.

<a name="dp.Model.reset"/>
### reset() ###
Resets the parameters of the Model.

<a name='dp.Layer'/>
## Layer ##
Abstract class inherited by component [Models](#dp.Model). 
The opposite of [Container](#dp.Container) in that it doesn't contain other Models.
The Layer should be parameterized.

<a name='dp.Layer.__init'/>
### dp.Layer{input_view, output_view, output, [dropout, sparse_init]} ###
Constructs a Layer. Arguments should be specified as key-value pairs. Other then the following 
arguments, those specified in [Model](#dp.Model.__init) also apply.

`input_view` is a string specifying the `view` of the `input` [View](view.md#dp.View) like _bf_, _bhwc_, etc. 
This is usually hardcoded for each sub-class.

`output_view` is a string specifying the `view` of the `output` [View](view.md#dp.View) like _bf_, _bhwc_, etc.
This is usually hardcoded for each sub-class.

`output` is a [View](view.md#dp.View) used for communicating outputs and gradOutputs. 
This is usually hardcoded for each sub-class.
      
`dropout` is a [Dropout](https://github.com/clementfarabet/lua---nnx/blob/master/Dropout.lua) Module instance. When provided, 
it is applied to the inputs of the Model. Defaults to not using dropout.

`sparse_init` is a boolean with a default value of true. When true, applies a sparse initialization of weights. See Martens (2010), [Deep learning via Hessian-free optimization](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_Martens10.pdf). This is 
the recommended initialization for [ReLU](https://github.com/torch/nn/blob/master/doc/transfer.md#relu) Transfer Modules.

<a name='dp.Layer.inputAct'/>
### [act] inputAct() ###
Returns the result of a [forwardGet](view.md#dp.View.forwardGet) on the Layer's `input` 
using its `input_view` and `input_type`.

<a name='dp.Layer.outputGrad'/>
### [grad] outputGrad() ###
Return the result of a [backwardGet](view.md#dp.View.backwardGet) on the Layer's `output` 
using its `output_view` and `output_type`.

<a name='dp.Layer.inputGrad'/>
### inputGrad(input_grad) ###
Sets the Layer's `input` gradient by calling its [backwardPut](view.md#dp.View.backwardPut) using its `input_view` 
and the provided `input_grad` Tensor.

<a name='dp.Layer.outputAct'/>
### outputAct(output_act) ###
Sets the Layer's `output` activation by calling its [forwardPut](view.md#dp.View.forwardPut) using its `output_view` 
and the provided `output_act` Tensor.

<a name='dp.Layer.maxNorm'/>
### maxNorm(max_out_norm, max_in_norm) ###
A method called by the MaxNorm Visitor. Imposes a hard constraint on the upper bound of the norm of output and/or input
neuron weights (in a weight matrix). Has a regularization effect analogous to WeightDecay, but with easier to optimize 
hyper-parameters. Quite useful with unbounded Transfer Modules like ReLU.

Only affects 2D [parameters](#dp.Model.parameters) like the usual `weights`. 
Assumes that 2D parameters are arranged : `output_dim x input_dim`.

<a name='dp.Neural'/>
## Neural ##
[Linear](https://github.com/torch/nn/blob/master/doc/simple.md#nn.Linear) (an affine transformation) 
followed by a [Transfer](https://github.com/torch/nn/blob/master/doc/transfer.md) Module. 
Both the `input_view` and `output_view` are _bf_. 

<a name='dp.Neural.__init'/>
### dp.Neural{input_size, output_size, transfer} ###
Constructs a Neural Layer. Arguments should be specified as key-value pairs. Other then the following 
arguments, those specified in [Layer](#dp.Layer.__init) also apply.

`input_size` specifies the number of input neurons.

`output_size` specifies the Number of output neurons.

`transfer` is a [Transfer](https://github.com/torch/nn/blob/master/doc/transfer.md) Module instance 
like [Tanh](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.Tanh),
[Sigmoid](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.Sigmoid), 
[ReLU]([ReLU](https://github.com/torch/nn/blob/master/doc/transfer.md#relu), etc. If the intent is to 
use Neural as a linear affine transform (without a non-linearity), one can use an
[Identity](https://github.com/torch/nn/blob/master/doc/simple.md#nn.Identity) Module instance.

<a name='dp.Convolution1D'/>
## Convolution1D ##
[TemporalConvolution](https://github.com/torch/nn/blob/master/doc/convolution.md#temporalconvolution) (a 1D convolution) 
followed by a [Transfer](https://github.com/torch/nn/blob/master/doc/transfer.md) Module and a 
[TemporalMaxPooling](https://github.com/torch/nn/blob/master/doc/convolution.md#temporalmaxpooling). 
Both the `input_view` and `output_view` are _bwc_. 

<a name='dp.Convolution1D.__init'/>
### dp.Convolution1D{input_size, output_size, kernel_size, kernel_stride, pool_size, pool_stride, transfer} ###
Constructs a Convolution1D Layer. Arguments should be specified as key-value pairs. 
Other then the following arguments, those specified in [Layer](#dp.Layer.__init) also apply.

`input_size` specifies the number of input channels (the size of the word embedding).

`output_size` specifies the number of output channels, i.e. the `outputFrameSize`.

`kernel_size` is a number specifying the size of the temporal convolution kernel.

`kernel_stride` specifies the stride of the temporal convolution. 
Note that depending of the size of your kernel, several (of the last) 
columns of the input sequence might be lost. It is up to the user 
to add proper padding in sequences. Defaults to 1.
 
`pool_size` is a number specifying the size of the temporal max pooling.

`pool_stride` is a number specifying the stride of the temporal max pooling.

`transfer` is a transfer Module like [Tanh](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.Tanh),
[Sigmoid](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.Sigmoid), 
[ReLU]([ReLU](https://github.com/torch/nn/blob/master/doc/transfer.md#relu), etc.

<a name='dp.Convolution2D'/>
## Convolution2D ##
[SpatialConvolutionMM](https://github.com/torch/nn/blob/master/doc/convolution.md#spatialconvolution) (a 2D convolution) 
followed by a [Transfer](https://github.com/torch/nn/blob/master/doc/transfer.md) Module and a 
[SpatialMaxPooling](https://github.com/torch/nn/blob/master/doc/convolution.md#spatialmaxpooling). 
Both the `input_view` and `output_view` are _bcwh_. 

<a name='dp.Convolution2D.__init'/>
### dp.Convolution2D{input_size, output_size, kernel_size, kernel_stride, pool_size, pool_stride, transfer} ###
Constructs a Convolution2D Layer. Arguments should be specified as key-value pairs. 
Other then the following arguments, those specified in [Layer](#dp.Layer.__init) also apply.

`input_size` specifies the number of input channels (like the number of colors).

`output_size` specifies the number of output channels, i.e. the `outputFrameSize`. 
If using CUDA, should be a multiple of 8.

`kernel_size` is a table-pair specifying the size `{width,height}` of the spatial convolution kernel.

`kernel_stride` is a table-pair specifying the stride `{width,height}` of the temporal convolution. 
Note that depending of the size of your kernel, several (of 
the last) columns or rows of the input image might be lost. 
It is up to the user to add proper padding in images. Defaults to `{1,1}`
 
`pool_size` is a table-pair specifying the size `{width,height}` of the spatial max pooling.

`pool_stride` is a table-pair specifying the stride `{width,height}` of the spatial max pooling.

`transfer` is a transfer Module like [Tanh](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.Tanh),
[Sigmoid](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.Sigmoid), 
[ReLU]([ReLU](https://github.com/torch/nn/blob/master/doc/transfer.md#relu), etc.

<a name='dp.Container'/>
## Container ##
Abstract class inherited by composite [Models](#dp.Model).

<a name='dp.Container.__init'/>
### dp.Container{models} ###
Constructs a Neural. Arguments should be specified as key-value pairs. Other then the following 
arguments, those specified in [Model](#dp.Model.__init) also apply.

`models` is a table of Models. 

<a name='dp.Container.extend'/>
### extend(models) ###
Adds `models` to the end of the existing composite of models.

<a name='dp.Container.add'/>
### add(model) ###
Add `model` to the end of the existing composite of models.

<a name='dp.Container.size'/>
### [size] size() ###
Returns the number of `models` in the Container.

<a name='dp.Container.get'/>
### [model] get(index) ###
Returns the component Model at index `index`.

<a name='dp.Sequential'/>
## Sequential ##



