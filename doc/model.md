# Models #  
 * [Model](#dp.Model)
   * [Layer](#dp.Layer)
    * [Neural](#dp.Neural)
    * [Convolution1D](#dp.Convolution1D)
   * [Container](#dp.Container)
    * [Sequential](#dp.Sequential)

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

<a name='dp.Model.reset"/>
### reset() ###
Resets the parameters of the Model.
