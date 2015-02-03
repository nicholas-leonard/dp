# Models  #

  * [Model](#dp.Model) : abstract class inherited by Layer and Container;
  * [Layer](#dp.Layer) : abstract class inherited by component Models ;
    * [Module](#dp.Module) : generic nn.Module adapter ;
    * [Neural](#dp.Neural) : Linear followed by a Transfer Module;
    * [Dictionary](#dp.Dictionary) : a LookupTable wrapper using for word embeddings;
    * [RecurrentDictionary](#dp.RecurrentDictionary) : used for building Simple RNN having a LookupTable input layer;
    * [Convolution1D](#dp.Convolution1D) : TemporalConvolution followed by a Transfer Module and TemporalMaxPooling;
    * [Convolution2D](#dp.Convolution2D) : SpatialConvolution followed by a Transfer Module and SpatialMaxPooling;
    * [SoftmaxTree](#dp.SoftmaxTree) : a hierarchy of parameterized softmaxes;
    * [MixtureOfExperts](#dp.MixtureOfExperts) : a mixture of experts using MLPs;
  * [Container](#dp.Container) : abstract class inherited by composite Models;
    * [Sequential](#dp.Sequential) : a sequence of Models.

<a name="dp.Model"/>
[]()
## Model ##
A Model is a [Node](node.md#dp.Node) sub-class. It adapts [Modules](https://github.com/torch/nn/blob/master/doc/module.md#module) to 
the *dp* library. Models can abstract multiple Modules. In fact, most Models, like [Neural](#dp.Neural) and [Convolution2D](#dp.Convolution2D)
contain 2-3 Modules. Model is the abstract class shared by component [Layer](#dp.Layer) and composite [Container](#dp.Container).

Unlike Modules, Models should be parameterized (although this isn't strictly enforced). This means that [Transfer](https://github.com/torch/nn/blob/master/doc/transfer.md) Modules 
shouldn't be specified in their own dedicated Model. Instead, we encourage that Models use Transfer Modules after parameterized Modules like 
[Linear](https://github.com/torch/nn/blob/master/doc/simple.md#nn.Linear) or [SpatialConvolution](https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialConvolution).

<a name="dp.Model.__init"/>
[]()
### dp.Model{typename, [tags, mvstate]} ###
Constructs a Model. Arguments should be specified as key-value pairs. Other then the following 
arguments, those specified in [Node](node.md#dp.Node.__init) also apply.
 
  * `typename` is a string which identifies the Model type in reports.
  * `tags` is a table of tags (as keys with values of true) used for determining which Visitors are allowed to visit the model.
  * `mvstate` is a table holding the Model-Visitor state. Can be used to specify arguments to Visitors that will adapt these to the Model.

<a name="dp.Model.forward"/>
[]()
### [output, carry] forward(input, carry) ###
Forward propagates an `input` [View](view.md#dp.View) to fill and return an `output` View.
 
  * `input` is a [View](view.md#dp.View) that should have been previously filled by a [forwardPut](view.md#dp.View.forwardPut). The Model will call one or many [forwardGets](view.md#dp.View.forwardGet) to retrieve a Tensor in a suitable format for forward propagation through the Model's internal [Modules](https://github.com/torch/nn/blob/master/doc/module.md#module).
  * `carry` is a [Carry](data.md#dp.Carry) that is carried throughout the graph. A Node can modify it, but should avoid deleting attributes. This is useful when you want to forward/backward information to a later/previous Node in the graph seperated by an unknown number of [Nodes](node.md#dp.Node).

The returned `output` is a View filled using a forwardPut.

<a name="dp.Model.evaluate"/>
[]()
### [output, carry] evaluate(input, carry) ###
This method is like [forward](#dp.Model.forward), but for evaluation purposes (valid/test).
This is useful for stochastic Modules like Dropout, which have 
different behavior for training than for evaluation. The default is to set 
`carry:putObj('evaluate', true)` and to call [forward](#dp.Model.forward).

<a name="dp.Model.backward"/>
[]()
### [input, carry] backward(output, carry) ###
Backward propagates an `output` [View](view.md#dp.View) to fill and `input` View with a gradient and return said `input`.
  
  * `output` is a [View](view.md#dp.View) that should have been previously filled by a [forwardPut](view.md#dp.View.forwardPut) in this Model's [forward](#dp.Model.forward) method and subsequently filled with a [backwardPut](view.md#dp.View.backwardPut) from the next [Node](node.md#dp.Node) in the digraph. The Model will call [backwardGet](view.md#dp.View.forwardGet) on the `output` to retrieve a gradient Tensor in a suitable format for forward propagation through the Model's internal [Modules](https://github.com/torch/nn/blob/master/doc/module.md#module).
  * `carry` is a table that is carried throughout the graph. A Node can modify it, but should avoid deleting attributes. This is useful when you want to forward/backward information to a later/previous Node in the graph seperated by an unknown number of Nodes.

The returned `input` is a View filled using a backwardPut. It is the same View that was passed to the previous call to this Model's [forward](#dp.Model.forward).

<a name="dp.Model.parameters"/>
[]()
### [params, gradParams [, scales, size]] parameters() ###
The method is used by Visitors for manipulating parameters and gradients. Return 2 to 3 tables : 

  * `params` : a table of parameters used by the Model;
  * `gradParams` : a table of parameter gradients (w.r.t Loss) used by the Model; 
  * `scales` : a table of update scales (optional);
  * `size` : a number specifying the maximum number of parameters to be returned (optional).
 
Each param/gradParam/scale triplet must be identified by a unique key, 
i.e. the tensors associated to each key must be the same from batch to batch. 
This allows Visitors to use the Model's `mvstate` table to append meta-data 
to the triplet for later use. This is the case of Momentum which must 


accumulate `pastGrads` for each triplet.

<a name="dp.Model.accept"/>
[]()
### accept(visitor) ###
Accepts a `visitor` [Visitor](visitor.md#dp.Visitor) that will visit the 
Model and any of its component Models. This is how the Model's parameters are updated.

<a name="dp.Model.reset"/>
[]()
### reset() ###
Resets the parameters (and parameter gradients) of the Model.

<a name="dp.Model.zeroGradParameters"/>
[]()
### zeroGradParameters() ###
A method called by a [Visitor](visitor.md#dp.Visitor) after a model has been visited
by all visitors. Internally, it will zero the parameter gradient vectors by calling
[nn.Module:zeroGradParameters](https://github.com/torch/nn/blob/master/doc/module.md#zerogradparameters).
In some cases, it also performs some cleanup operations (like emptying a table) 
in preparation for the next batch of `forward` and `backward` propagations.

Note that multiple calls to `forward` and `backward` between parameter updates 
(as opposed to the usual one of each) will accumulate parameter gradients from 
each `backward` until `zeroGradParameters` is called.

<a name="dp.Model.toModule"/>
[]()
### toModule([batch]) ###
Returns its contained Model [Modules](https://github.com/torch/nn/blob/master/doc/module.md#module) 
and those of it's contained [Views](view.md#dp.View) as a composite Module.
The method requires that a previous call to [forward](#dp.Model.forward) be made,
which is done automatically when argument `batch`, a [Batch](data.md#dp.Batch) instance, is provided.
This is particularly useful when you want to use the [dp](index.md) framework to train your 
Modules, but want to omit it in your production environment 
(and just use [nn](https://github.com/torch/nn/blob/master/README.md) and such instead).

<a name='dp.Layer'/>
[]()
## Layer ##
[]()
Abstract class inherited by component [Models](#dp.Model). 
The opposite of [Container](#dp.Container) in that it doesn't contain other Models.
The Layer should be parameterized.

<a name='dp.Layer.__init'/>
[]()
### dp.Layer{input_view, output_view, output, [dropout, sparse_init]} ###
Constructs a Layer. Arguments should be specified as key-value pairs. Other then the following 
arguments, those specified in [Model](#dp.Model.__init) also apply.
  
  * `input_view` is a string specifying the `view` of the `input` [View](view.md#dp.View) like _bf_, _bhwc_, etc.  This is usually hardcoded for each sub-class.
  * `output_view` is a string specifying the `view` of the `output` [View](view.md#dp.View) like _bf_, _bhwc_, etc. This is usually hardcoded for each sub-class.
  * `output` is a [View](view.md#dp.View) used for communicating outputs and gradOutputs. This is usually hardcoded for each sub-class.
  * `dropout` is a [Dropout](https://github.com/clementfarabet/lua---nnx/blob/master/Dropout.lua) Module instance. When provided, it is applied to the inputs of the Model. Defaults to not using dropout.
  * `sparse_init` is a boolean with a default value of false. When true, applies a sparse initialization of weights. See Martens (2010), [Deep learning via Hessian-free optimization](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_Martens10.pdf). This is 
the recommended initialization for non-convolution layers activated by [ReLU](https://github.com/torch/nn/blob/master/doc/transfer.md#relu) Transfer Modules.
  * `acc_update` is a boolean. When true, it uses the faster [accUpdateGradParameters](https://github.com/torch/nn/blob/master/doc/module.md#accupdategradparametersinput-gradoutput-learningrate) which performs an inplace update (no need for param gradients). However, this also means that [Momentum](visitor.md#dp.Momentum), [WeightDecay](visitor.md#dp.WeightDecay) and other such parameter gradient modifying [Visitors](visitor.md#dp.Visitor) cannot be used.

<a name='dp.Layer.inputAct'/>
[]()
### [act] inputAct() ###
Returns the result of a [forwardGet](view.md#dp.View.forwardGet) on the Layer's `input` 
using its `input_view` and `input_type` attributes.

<a name='dp.Layer.outputGrad'/>
[]()
### [grad] outputGrad() ###
Return the result of a [backwardGet](view.md#dp.View.backwardGet) on the Layer's `output` 
using its `output_view` and `output_type` attributes.

<a name='dp.Layer.inputGrad'/>
[]()
### inputGrad(input_grad) ###
Sets the Layer's `input` gradient by calling its [backwardPut](view.md#dp.View.backwardPut) using its `input_view` 
and the provided `input_grad` Tensor.

<a name='dp.Layer.outputAct'/>
[]()
### outputAct(output_act) ###
Sets the Layer's `output` activation by calling its [forwardPut](view.md#dp.View.forwardPut) using its `output_view` and the provided `output_act` Tensor.

<a name='dp.Layer.maxNorm'/>
[]()
### maxNorm(max_out_norm, max_in_norm) ###
A method called by the [MaxNorm](visitor.md#dp.MaxNorm) Visitor. Imposes a hard constraint on the upper bound of the norm of output and/or input neuron weights (in a weight matrix). 
Has a regularization effect analogous to WeightDecay, but with easier to optimize hyper-parameters. 
Quite useful with unbounded Transfer Modules like ReLU. 
Only affects 2D [parameters](#dp.Model.parameters) like the usual `weight` matrix. Assumes that 2D parameters are arranged : `output_dim x input_dim`.

<a name="dp.Module"/>
[]()
## Module ##
A generic [nn.Module](https://github.com/torch/nn/blob/master/doc/module.md#module) adapter. 
Not to be confused with nn.Module. Use this to quickly wrap a nn.Module into a [Model](#dp.Model). 
For all intents and purposes, it should do a great 
job of integrating your existing Modules into dp. Just wrap them using this Model. 
However, some dp.Visitors expect each param/gradParam to be identified by a 
unique key that stays the same from batch to batch.
This wont be true for modules like nnx.SoftMaxTree or nnx.LookupTable (so be weary).

<a name="dp.Module.__init"/>
[]()
### dp.Module{module[, input_view]} ###
Module constructor. Other then the following 
arguments, those specified in [Layer](#dp.Layer.__init) also apply:
  
  * `module` is a nn.Module instance.
  * `input_view` is a string specifying the `view` of the `input` [View](view.md#dp.View) like _bf_, _bhwc_, etc. Defaults to `default` axis view.

<a name='dp.Neural'/>
[]()
## Neural ##
[Linear](https://github.com/torch/nn/blob/master/doc/simple.md#nn.Linear) (an affine transformation) 
followed by a [Transfer](https://github.com/torch/nn/blob/master/doc/transfer.md) Module. Both the `input_view` and `output_view` are _bf_. 

<a name='dp.Neural.__init'/>
[]()
### dp.Neural{input_size, output_size, transfer} ###
Constructs a Neural Layer. Arguments should be specified as key-value pairs. Other then the following 
arguments, those specified in [Layer](#dp.Layer.__init) also apply.
 
  * `input_size` specifies the number of input neurons.
  * `output_size` specifies the Number of output neurons.
  * `transfer` is a [Transfer](https://github.com/torch/nn/blob/master/doc/transfer.md) Module instance like [Tanh](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.Tanh), [Sigmoid](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.Sigmoid), 
[ReLU]([ReLU](https://github.com/torch/nn/blob/master/doc/transfer.md#relu), etc. If the intent is to use Neural as a linear affine transform (without a non-linearity), one can use an [Identity](https://github.com/torch/nn/blob/master/doc/simple.md#nn.Identity) Module instance.

<a name='dp.Dictionary'/>
[]()
## Dictionary ##
Adapts a [LookupTable](https://github.com/torch/nn/blob/master/doc/convolution.md#nn.LookupTable). 
Used primarily for learning word embeddings. This Model can only be 
situated at the begining of a digraph as `LookupTable:backward` produces no `gradInput` Tensor.

<a name='dp.Dictionary.__init'/>
[]()
### dp.Dictionary ###
Constructs a Dictionary Layer. Arguments should be specified as key-value pairs. 
Other than the following arguments, those specified in [Layer](#dp.Layer.__init) also apply.
 
  * `dict_size` specifies the number of entries in the dictionary (e.g. number of words).
  * `output_size` specifies the number of neurons per entry. This is also known as the embedding size.

<a name='dp.RecurrentDictionary'/>
[]()
## RecurrentDictionary ##
Adapts a [Recurrent](https://github.com/clementfarabet/lua---nnx#nnx.Recurrent) 
Module encapsulating a [LookupTable](https://github.com/torch/nn/blob/master/doc/convolution.md#nn.LookupTable) 
`input` layer, a [Linear](https://github.com/torch/nn/blob/master/doc/simple.md#nn.Linear) `feedback` layer
and a non-linear `transfer` layer. In effect, this implements everything but the `output` layer 
of a Simple Recurrent Neural Network (SRNN), specifically an SRNN used for modeling 
of temporal sequences, typically language models. 

Due to its recurrent nature, this model has strong ties with other objects,
including the SentenceSampler and the RecurrentVisitorChain. 
An experiment combining all three components is examplified in the
[recurrentneuralnetwork.lua](../examples/recurrentlanguagemodel.lua) script.

<a name='dp.RecurrentDictionary.__init'/>
[]()
### dp.RecurrentDictionary{...} ###
Constructs a RecurrentDictionary Layer. Arguments should be specified as key-value pairs. 
Other than the following arguments, those specified in [Layer](#dp.Layer.__init) also apply.
 
  * `dict_size` specifies the number of entries in the `input` layer dictionary, i.e. the vocabulary size of the LookupTable.
  * `output_size` specifies the number of neurons per entry. Also the input and output size of the `feedback` layer.
  * `rho` specifies the number of time-steps back in time to back-propagate through (refer to [Recurrent](https://github.com/clementfarabet/lua---nnx#nnx.Recurrent)). Defaults to 5.
  * `transfer` is a transfer Module like [Tanh](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.Tanh),
[Sigmoid](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.Sigmoid), [ReLU]([ReLU](https://github.com/torch/nn/blob/master/doc/transfer.md#relu), etc. Defaults to nn.Sigmoid (recommended for RNNs).

<a name='dp.Convolution1D'/>
[]()
## Convolution1D ##
[TemporalConvolution](https://github.com/torch/nn/blob/master/doc/convolution.md#temporalconvolution) (a 1D convolution) followed by a [Transfer](https://github.com/torch/nn/blob/master/doc/transfer.md) Module and a 
[TemporalMaxPooling](https://github.com/torch/nn/blob/master/doc/convolution.md#temporalmaxpooling). 
Both the `input_view` and `output_view` are _bwc_. 

<a name='dp.Convolution1D.__init'/>
[]()
### dp.Convolution1D{...} ###
Constructs a Convolution1D Layer. Arguments should be specified as key-value pairs. 
Other than the following arguments, those specified in [Layer](#dp.Layer.__init) also apply.
 
  * `input_size` specifies the number of input channels (the size of the word embedding).
  * `output_size` specifies the number of output channels, i.e. the `outputFrameSize`.
  * `kernel_size` is a number specifying the size of the temporal convolution kernel.
  * `kernel_stride` specifies the stride of the temporal convolution. Note that depending of the size of your kernel, several (of the last) columns of the input sequence might be lost. It is up to the user to add proper padding in sequences. Defaults to 1.
  * `pool_size` is a number specifying the size of the temporal max pooling.
  * `pool_stride` is a number specifying the stride of the temporal max pooling.
  * `transfer` is a transfer Module like [Tanh](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.Tanh),
[Sigmoid](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.Sigmoid), [ReLU]([ReLU](https://github.com/torch/nn/blob/master/doc/transfer.md#relu), etc.

<a name='dp.Convolution2D'/>
[]()
## Convolution2D ##
[SpatialConvolutionMM](https://github.com/torch/nn/blob/master/doc/convolution.md#spatialconvolution) (a 2D convolution) followed by a [Transfer](https://github.com/torch/nn/blob/master/doc/transfer.md) Module and a 
[SpatialMaxPooling](https://github.com/torch/nn/blob/master/doc/convolution.md#spatialmaxpooling). 
Both the `input_view` and `output_view` are _bcwh_. 

<a name='dp.Convolution2D.__init'/>
[]()
### dp.Convolution2D{input_size, output_size, kernel_size, kernel_stride, pool_size, pool_stride, transfer} ###
Constructs a Convolution2D Layer. Arguments should be specified as key-value pairs. 
Other then the following arguments, those specified in [Layer](#dp.Layer.__init) also apply.
 
  * `input_size` specifies the number of input channels (like the number of colors).
  * `output_size` specifies the number of output channels, i.e. the `outputFrameSize`. If using CUDA, should be a multiple of 8.
  * `kernel_size` is a table-pair specifying the size `{width,height}` of the spatial convolution kernel.
  * `kernel_stride` is a table-pair specifying the stride `{width,height}` of the temporal convolution. Note that depending of the size of your kernel, several (of the last) columns or rows of the input image might be lost. It is up to the user to add proper padding in images. Defaults to `{1,1}`
  * `pool_size` is a table-pair specifying the size `{width,height}` of the spatial max pooling.
  * `pool_stride` is a table-pair specifying the stride `{width,height}` of the spatial max pooling.
  * `padding` specifies the number of zero-padding to add to the input before performing the convolution (allows the `height x width` of output to be larger).
  * `transfer` is a transfer Module like [Tanh](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.Tanh),
[Sigmoid](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.Sigmoid), [ReLU](https://github.com/torch/nn/blob/master/doc/transfer.md#relu), etc.

<a name='dp.SoftmaxTree'/>
[]()
## SoftmaxTree ##
A hierarchy of parameterized softmaxes. Used for computing the likelihood of a leaf class. 
Should be used with [TreeNLL](loss.md#dp.TreeNLL) Loss. Requires a tensor mapping one `parent_id` to many `child_id`. 
Greatly accelerates learning and testing for language models with large vocabularies. 
A vocabulary hierarchy is provided via the [BillionWords](data.md#dp.BillionWords) DataSource.

<a name='dp.SoftmaxTree.__init'/>
[]()
### dp.SoftmaxTree{...} ###
Constructs a SoftmaxTree Layer. Arguments should be specified as key-value pairs. 
Other then the following arguments, those specified in [Layer](#dp.Layer.__init) also apply.
 
  * `input_size` specifies the number of input neurons, also known as the output embedding size.
  * `hierarchy` is a table mapping integer `parent_ids` to a tensor of `child_ids`.
  * `root_id` is an integer specifying the `id` of the root of the tree. Defaults to 1.

<a name='dp.MixtureOfExperts'/>
[]()
## MixtureOfExperts ##
A mixture of MLP experts gated by an MLP gater.
    
<a name='dp.MixtureOfExperts.__init'/>
[]()
### dp.MixtureOfExperts{...} ###
Constructs a MixtureOfExperts Layer. Arguments should be specified as key-value pairs. 
Other then the following arguments, those specified in [Layer](#dp.Layer.__init) also apply.
 
  * `input_size` specifies the number of input neurons.
  * `n_expert` specifies the number of experts.
  * `expert_size` is a table outlining number of neurons per expert hidden layer.
  * `expert_act` is a transfer Module like [Tanh](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.Tanh) (the default) that will be used to activate the expert hidden layer(s).
  * `gater_size` specifies the number of neurons in gater hidden layers.
  * `gater_act` is a transfer Module like [Tanh](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.Tanh) (the default) that will be used to activate the gater's hidden layer(s).
  * `output_size` specifies the output size of the Model.
  * `output_act` specifies the output activation Module. Defaults to [LogSoftMax](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.LogSoftMax).
 
<a name='dp.Container'/>
[]()
## Container ##
Abstract class inherited by composite [Models](#dp.Model) (see [Composite Design Pattern](https://en.wikipedia.org/wiki/Composite_pattern)).

<a name='dp.Container.__init'/>
[]()
### dp.Container{models} ###
Container Constructor. Arguments should be specified as key-value pairs. Other then the following 
arguments, those specified in [Model](#dp.Model.__init) also apply:
 
  * `models` is a table of Models. 

<a name='dp.Container.extend'/>
[]()
### extend(models) ###
Adds `models` to the end of the existing composite of models.

<a name='dp.Container.add'/>
[]()
### add(model) ###
Add `model` to the end of the existing composite of models.

<a name='dp.Container.size'/>
[]()
### [size] size() ###
Returns the number of `models` in the Container.

<a name='dp.Container.get'/>
[]()
### [model] get(index) ###
Returns the component Model at index `index`.

<a name='dp.Sequential'/>
[]()
## Sequential ##
This Container is used for building multi-layer perceptrons (MLP). It has a 
similar interface to [nn.Sequential](https://github.com/torch/nn/blob/master/doc/containers.md#nn.Sequential), 
but doesn't use it internally, i.e. dp.Sequential does not adapt nn.Sequential.

